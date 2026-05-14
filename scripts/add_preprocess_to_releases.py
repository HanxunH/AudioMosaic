"""Inject a self-contained preprocess() into modeling.py + update config.json + load_model.py for all HF releases."""
import json
import os

ROOT = "/Users/hanxunh/Desktop/Research/Code/AudioMosaic-ICML26-CR"
HF_DIR = f"{ROOT}/hf_release"

# Per-variant audio-preprocessing defaults
AUDIOSET = {"target_length": 1024, "num_mel_bins": 128, "norm_mean": -4.2677393, "norm_std": 4.5689974, "sample_rate": 16000, "frame_shift": 10.0}
ESC      = {"target_length": 512,  "num_mel_bins": 128, "norm_mean": -6.6268077, "norm_std": 5.358466,  "sample_rate": 16000, "frame_shift": 10.0}
SPC      = {"target_length": 128,  "num_mel_bins": 128, "norm_mean": -6.845978,  "norm_std": 5.5654526, "sample_rate": 16000, "frame_shift": 10.0}

PREPROCESSING = {
    "AudioMosaic-vit-b16-pretrained":                       AUDIOSET,
    "AudioMosaic-vit-b16-finetune-as20k":                   AUDIOSET,
    "AudioMosaic-vit-b16-finetune-as2m":                    AUDIOSET,
    "AudioMosaic-vit-b16-linear-prob-as20k":                AUDIOSET,
    "AudioMosaic-vit-b16-linear-prob-as20k-attentive":      AUDIOSET,
    "AudioMosaic-vit-b16-finetune-envsdd-ata":              AUDIOSET,
    "AudioMosaic-vit-b16-finetune-envsdd-tta":              AUDIOSET,
    "AudioMosaic-vit-b16-finetune-esc-split1":              ESC,
    "AudioMosaic-vit-b16-finetune-esc-split2":              ESC,
    "AudioMosaic-vit-b16-finetune-esc-split3":              ESC,
    "AudioMosaic-vit-b16-finetune-esc-split4":              ESC,
    "AudioMosaic-vit-b16-finetune-esc-split5":              ESC,
    "AudioMosaic-vit-b16-finetune-spc1":                    SPC,
    "AudioMosaic-vit-b16-finetune-spc2":                    SPC,
    "AudioMosaic-vit-b16-ltu-stage4":                       AUDIOSET,
}


PREPROCESS_FN = '''

# -----------------------------------------------------------------------------
# Self-contained audio preprocessing
# -----------------------------------------------------------------------------
def preprocess(audio, sample_rate=16000, target_length=1024, num_mel_bins=128,
               norm_mean=-4.2677393, norm_std=4.5689974, frame_shift=10.0):
    """Turn raw audio into a log-mel spectrogram suitable for AudioMosaic models.

    Args:
        audio: path to an audio file (str), a (waveform, sr) tuple, or a waveform tensor [C, T].
        sample_rate: target sample rate (default 16000). Input is resampled if different.
        target_length: number of mel frames (pad/truncate). 1024 for AudioSet/EnvSDD,
                       512 for ESC-50, 128 for Speech Commands.
        num_mel_bins: number of mel filterbank bins. Default 128.
        norm_mean, norm_std: per-bin normalization constants matching the model's training set.
                             AudioSet: (-4.2677393, 4.5689974); ESC-50: (-6.6268077, 5.358466);
                             SPC: (-6.845978, 5.5654526).
        frame_shift: frame shift in milliseconds. Default 10.

    Returns:
        Tensor of shape [1, 1, target_length, num_mel_bins], ready to feed to model(x).
    """
    import torchaudio
    if isinstance(audio, str):
        waveform, sr = torchaudio.load(audio)
    elif isinstance(audio, tuple):
        waveform, sr = audio
    else:
        waveform, sr = audio, sample_rate
    if waveform.shape[-1] < sample_rate:
        waveform = torch.nn.functional.pad(waveform, (0, sample_rate - waveform.shape[-1]))
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform, htk_compat=True, sample_frequency=sample_rate,
        use_energy=False, window_type='hanning',
        num_mel_bins=num_mel_bins, dither=0.0, frame_shift=frame_shift,
    )
    n_frames = fbank.shape[0]
    pad_n = target_length - n_frames
    if pad_n > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, pad_n))
    elif pad_n < 0:
        fbank = fbank[:target_length, :]
    fbank = (fbank - norm_mean) / norm_std
    return fbank.unsqueeze(0).unsqueeze(0)
'''


def update_modeling(repo_dir):
    path = os.path.join(repo_dir, "modeling.py")
    src = open(path).read()
    if "def preprocess(" in src:
        # Already injected; replace block (between the marker and the next top-level def/class or EOF)
        marker = "# Self-contained audio preprocessing"
        i = src.find(marker)
        if i != -1:
            src = src[:src.rfind("# " + "-" * 70, 0, i)]
        else:
            src = src[:src.find("def preprocess(")]
    src = src.rstrip() + PREPROCESS_FN
    with open(path, "w") as f:
        f.write(src)


def update_config(repo_dir, defaults):
    path = os.path.join(repo_dir, "config.json")
    cfg = json.load(open(path))
    cfg["audio_preprocessing"] = defaults
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def update_load_model(repo_dir, is_ltu=False, is_pretrain=False):
    path = os.path.join(repo_dir, "load_model.py")
    src = open(path).read()
    if "def preprocess(" in src:
        return  # already updated
    # Append a preprocess() wrapper that uses config defaults
    wrapper = '''


def preprocess(audio, repo_dir: str = None, **overrides):
    """Convenience wrapper: read audio_preprocessing defaults from config.json and call modeling.preprocess.

    Pass `repo_dir` if calling from outside the snapshot, otherwise auto-detected.
    Use `**overrides` to override any default (e.g., target_length=2048).
    """
    if repo_dir is None:
        repo_dir = os.path.dirname(os.path.abspath(__file__))
    import json as _json
    with open(os.path.join(repo_dir, "config.json")) as f:
        cfg = _json.load(f)
    defaults = cfg.get("audio_preprocessing", {})
    defaults.update(overrides)
    from modeling import preprocess as _preprocess
    return _preprocess(audio, **defaults)
'''
    src = src.rstrip() + wrapper + "\n"
    with open(path, "w") as f:
        f.write(src)


if __name__ == "__main__":
    for name, defaults in PREPROCESSING.items():
        d = os.path.join(HF_DIR, name)
        if not os.path.isdir(d):
            print(f"  ⚠ skipping (no dir): {name}")
            continue
        update_modeling(d)
        update_config(d, defaults)
        update_load_model(d, is_ltu="ltu" in name, is_pretrain="pretrained" in name)
        print(f"  ✅ {name}: tgt_len={defaults['target_length']}, mean={defaults['norm_mean']:.3f}, std={defaults['norm_std']:.3f}")
    print("\nDone. Re-push with the existing release scripts to upload changes.")
