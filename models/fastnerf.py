import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        in_channels: number of input channels (3 for both xyz and direction)
        """
        super(Embedding, self).__init__()
        self.N_freqs = N_freqs
        self.in_channels = in_channels
        self.funcs = [torch.sin, torch.cos]
        self.out_channels = in_channels*(len(self.funcs)*N_freqs+1)

        if logscale:
            self.freq_bands = 2**torch.linspace(0, N_freqs-1, N_freqs)
        else:
            self.freq_bands = torch.linspace(1, 2**(N_freqs-1), N_freqs)

    def forward(self, x):
        """
        Embeds x to (x, sin(2^k x), cos(2^k x), ...) 
        Different from the paper, "x" is also in the output
        See https://github.com/bmild/nerf/issues/12

        Inputs:
            x: (B, self.in_channels)

        Outputs:
            out: (B, self.out_channels)
        """
        out = [x]
        for freq in self.freq_bands:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class FastNeRF(nn.Module):
    def __init__(self,
                 D=8, W=256,
                 in_channels_xyz=63, in_channels_dir=27, 
                 skips=[4]):
        """
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)
        skips: add skip connection in the Dth layer
        """
        super(FastNeRF, self).__init__()
        self.D = D
        self.W = W
        self.in_channels_xyz = in_channels_xyz
        self.in_channels_dir = in_channels_dir
        self.skips = skips

        # xyz encoding layers
        layers1 = []
        for i in range(4):
            if i == 0:
                layers1.append(nn.Linear(in_channels_xyz, W))
            else:
                layers1.append(nn.Linear(W, W))
        self.xyz_endocing_1 = nn.Sequential(*layers1, nn.ReLU(True))
        
        layers2 = []
        for i in range(4):
            if i == 0:
                layers2.append(nn.Linear(W+in_channels_xyz, W))
            else:
                layers2.append(nn.Linear(W, W))
        self.xyz_endocing_2 = nn.Sequential(*layers2, nn.ReLU(True))
        
        self.xyz_encoding_final = nn.Linear(W, W//2*3)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                                nn.Linear(in_channels_dir, W//2),
                                nn.ReLU(True))

        # output layers
        self.sigma = nn.Linear(W, 1)
        self.rgb = nn.Sequential(
                        nn.Linear(W//2, 3),
                        nn.Sigmoid())

    def forward(self, x, sigma_only=False):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py

        Inputs:
            x: (B, self.in_channels_xyz(+self.in_channels_dir))
               the embedded vector of position and direction
            sigma_only: whether to infer sigma only. If True,
                        x is of shape (B, self.in_channels_xyz)

        Outputs:
            if sigma_ony:
                sigma: (B, 1) sigma
            else:
                out: (B, 4), rgb and sigma
        """
        if not sigma_only:
            input_xyz, input_dir = \
                torch.split(x, [self.in_channels_xyz, self.in_channels_dir], dim=-1)
        else:
            input_xyz = x

        xyz_ = input_xyz
        xyz_ = self.xyz_endocing_1(xyz_)
        xyz_ = self.xyz_endocing_2(torch.cat([xyz_, input_xyz], -1))

        sigma = self.sigma(xyz_)
        if sigma_only:
            return sigma

        xyz_encoding_final = self.xyz_encoding_final(xyz_)
        xyz_encoding_final = torch.reshape(xyz_encoding_final, shape=(-1, self.W//2, 3))

        dir_encoding = self.dir_encoding(input_dir)
        dir_encoding_final = torch.reshape(dir_encoding, shape=(-1, 1, self.W//2))


        rgb = torch.bmm(dir_encoding_final, xyz_encoding_final).squeeze()

        out = torch.cat([rgb, sigma], -1)


        return out