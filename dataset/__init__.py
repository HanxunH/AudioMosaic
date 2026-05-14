import warnings
warnings.filterwarnings(
    "ignore",
    message=r".*torchcodec\.decoders\.AudioDecoder.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*StreamingMediaDecoder has been deprecated.*",
    category=UserWarning,
)

import mlconfig
from .audioset import DistributedSampler, WeightedRandomSampler, DistributedSamplerWrapper
from .dataset_manager import DatasetManager, RayDatasetManager

mlconfig.register(DatasetManager)
mlconfig.register(RayDatasetManager)