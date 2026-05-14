from .audioset import AudiosetDataset, AudiosetCLRDataset
from audiomentations import Compose, AddGaussianNoise, Trim, BitCrush, TimeMask, TanhDistortion, AdjustDuration, BandStopFilter, BandPassFilter
from audiomentations import PolarityInversion, Gain, PitchShift, HighPassFilter, LowPassFilter, Normalize, AddGaussianSNR, Limiter, Aliasing, TimeStretch
from audiomentations import Shift, Clip

    
class AudioMosaicPretrain(object):
    def __call__(self, x):
        augment = Compose([
            PolarityInversion(p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.7),
            AddGaussianSNR(p=0.5),
            Gain(p=0.3),
            HighPassFilter(p=0.3),
            BandStopFilter(p=0.5),
            PitchShift(p=0.6),
        ])
        x = augment(x, sample_rate=16000)
        return x


# Ablation: no augmentation
class AudioMosaicPretrain_NoAug(object):
    def __call__(self, x):
        return x


# Ablation: +PolarityInversion, +TimeStretch
class AudioMosaicPretrain_Aug2(object):
    def __call__(self, x):
        augment = Compose([
            PolarityInversion(p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.7),
        ])
        x = augment(x, sample_rate=16000)
        return x


# Ablation: +AddGaussianSNR, +Gain
class AudioMosaicPretrain_Aug4(object):
    def __call__(self, x):
        augment = Compose([
            PolarityInversion(p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.7),
            AddGaussianSNR(p=0.5),
            Gain(p=0.3),
        ])
        x = augment(x, sample_rate=16000)
        return x


# Ablation: +HighPassFilter, +BandStopFilter
class AudioMosaicPretrain_Aug6(object):
    def __call__(self, x):
        augment = Compose([
            PolarityInversion(p=0.5),
            TimeStretch(min_rate=0.7, max_rate=1.3, p=0.7),
            AddGaussianSNR(p=0.5),
            Gain(p=0.3),
            HighPassFilter(p=0.3),
            BandStopFilter(p=0.5),
        ])
        x = augment(x, sample_rate=16000)
        return x

dataset_options = {
    "AudiosetDataset": lambda path, transform, is_test, kwargs:
    AudiosetDataset(
        dataset_json_file=path,
        audio_conf=transform,
        roll_mag_aug=not is_test,
        mode='eval' if is_test else 'train',
        **kwargs
    ),
    "AudiosetCLRDataset": lambda path, transform, is_test, kwargs:
    AudiosetCLRDataset(
        dataset_json_file=path,
        audio_conf=transform,
        roll_mag_aug=not is_test,
        mode='eval' if is_test else 'train',
        **kwargs
    ),
}

transform_options = {
    "AudioSetFinetune": {
        "train_transform":  {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 48,
            'timem': 192,
            'mixup': 0.8,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        },
        "test_transform": {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioSetFinetune2M": {
        "train_transform":  {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 25,
            'timem': 200,
            'mixup': 0.8,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        },
        "test_transform": {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "ESCFinetune": {
        "train_transform":  {
            'num_mel_bins': 128, 
            'target_length': 512, 
            'freqm': 24,
            'timem': 96,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -6.6268077,
            'std': 5.358466,
            'noise': False,
            'multilabel': False
        },
        "test_transform": {
            'num_mel_bins': 128, 
            'target_length': 512, 
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -6.6268077,
            'std': 5.358466,
            'noise': False,
            'multilabel': False
        }
    },
    "SPCFinetune": {
        "train_transform":  {
            'num_mel_bins': 128, 
            'target_length': 128, 
            'freqm': 48,
            'timem': 48,
            'mixup': 0.8,
            'dataset': "SPC",
            'mode':'train',
            'roll_mag_aug': False,
            'mean': -6.845978,
            'std': 5.5654526,
            'noise': True,
            'multilabel': True
        },
        "test_transform": {
            'num_mel_bins': 128, 
            'target_length': 128, 
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "SPC",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -6.845978,
            'std': 5.5654526,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioMosaicPretrain": {
        "train_transform":  {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0.0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True,
            'clr_augment': AudioMosaicPretrain(),
        },
        "test_transform": {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioMosaicPretrain_NoAug": {
        "train_transform":  {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0.0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True,
            'clr_augment': AudioMosaicPretrain_NoAug(),
        },
        "test_transform": {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioMosaicPretrain_Aug2": {
        "train_transform":  {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0.0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True,
            'clr_augment': AudioMosaicPretrain_Aug2(),
        },
        "test_transform": {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioMosaicPretrain_Aug4": {
        "train_transform":  {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0.0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True,
            'clr_augment': AudioMosaicPretrain_Aug4(),
        },
        "test_transform": {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "AudioMosaicPretrain_Aug6": {
        "train_transform":  {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0.0,
            'dataset': "audioset",
            'mode':'train',
            'roll_mag_aug': True,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True,
            'clr_augment': AudioMosaicPretrain_Aug6(),
        },
        "test_transform": {
            'num_mel_bins': 128,
            'target_length': 1024,
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "audioset",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': True
        }
    },
    "EnvSDDFinetune": {
        "train_transform":  {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 48,
            'timem': 192,
            'mixup': 0.0,
            'dataset': "EnvSDD",
            'mode':'train',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': False
        },
        "test_transform": {
            'num_mel_bins': 128, 
            'target_length': 1024, 
            'freqm': 0,
            'timem': 0,
            'mixup': 0,
            'dataset': "EnvSDD",
            'mode':'val',
            'roll_mag_aug': False,
            'mean': -4.2677393,
            'std': 4.5689974,
            'noise': False,
            'multilabel': False
        }
    },
}


