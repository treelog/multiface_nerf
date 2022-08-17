from .blender import BlenderDataset
from .llff import LLFFDataset
from .multiface import MultiFaceDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'multiface': MultiFaceDataset}