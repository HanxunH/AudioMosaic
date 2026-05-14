"""
End-to-end smoke test for all AudioMosaic HF releases.

For each repo: snapshot_download -> import modeling -> instantiate -> load weights ->
run a forward pass on a dummy log-mel spectrogram of the configured size.
"""
import json
import os
import sys
import time

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file


# (repo_id, spec_size, expected_output_check)
VISION_REPOS = [
    ("hanxunh/AudioMosaic-vit-b16-pretrained",                       [1024, 128], "dict"),
    ("hanxunh/AudioMosaic-vit-b16-finetune-as20k",                   [1024, 128], (2, 527)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-as2m",                    [1024, 128], (2, 527)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-spc1",                    [128, 128],  (2, 11)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-spc2",                    [128, 128],  (2, 35)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-esc-split1",              [512, 128],  (2, 50)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-esc-split2",              [512, 128],  (2, 50)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-esc-split3",              [512, 128],  (2, 50)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-esc-split4",              [512, 128],  (2, 50)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-esc-split5",              [512, 128],  (2, 50)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-envsdd-ata",              [1024, 128], (2, 2)),
    ("hanxunh/AudioMosaic-vit-b16-finetune-envsdd-tta",              [1024, 128], (2, 2)),
    ("hanxunh/AudioMosaic-vit-b16-linear-prob-as20k",                [1024, 128], (2, 527)),
    ("hanxunh/AudioMosaic-vit-b16-linear-prob-as20k-attentive",      [1024, 128], (2, 527)),
]


def test_vision(repo_id, spec_size, expected):
    t0 = time.time()
    local = snapshot_download(repo_id)

    if local not in sys.path:
        sys.path.insert(0, local)

    # Force fresh import per repo (each has its own modeling.py)
    for mod_name in ("modeling", "load_model"):
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    cfg = json.load(open(os.path.join(local, "config.json")))
    cls_name = cfg.pop("model_class")

    import importlib
    modeling = importlib.import_module("modeling")
    model_cls = getattr(modeling, cls_name)
    model = model_cls(**cfg)
    msg = model.load_state_dict(load_file(os.path.join(local, "model.safetensors")), strict=False)

    model.eval()
    x = torch.randn(2, 1, *spec_size)
    with torch.no_grad():
        out = model(x)

    sys.path.remove(local)
    elapsed = time.time() - t0

    # Validate
    if expected == "dict":
        ok = isinstance(out, dict)
        shape_str = f"dict[{','.join(out.keys())}]" if ok else f"got {type(out).__name__}"
    else:
        ok = hasattr(out, "shape") and tuple(out.shape) == expected
        shape_str = f"{tuple(out.shape) if hasattr(out, 'shape') else type(out).__name__}"

    return {
        "repo": repo_id, "ok": ok and len(msg.missing_keys) == 0 and len(msg.unexpected_keys) == 0,
        "missing": len(msg.missing_keys), "unexpected": len(msg.unexpected_keys),
        "class": cls_name, "shape": shape_str, "elapsed": elapsed,
    }


def test_ltu():
    """Lighter check for LTU: download + import + sanity-check structures without loading 9.3GB Llama."""
    t0 = time.time()
    repo = "hanxunh/AudioMosaic-vit-b16-ltu-stage4"
    local = snapshot_download(repo, allow_patterns=[
        "*.json", "*.py", "tokenizer.model", "transformers_vendored/**",
        "extra_weights.bin",  # 372 MB but useful to inspect
    ])
    sys.path.insert(0, local)
    for m in list(sys.modules):
        if m.startswith(("modeling", "transformers_vendored", "load_model")):
            del sys.modules[m]

    import modeling
    import transformers_vendored
    import modeling_audiomosaic_ltu

    extras = torch.load(os.path.join(local, "extra_weights.bin"),
                        map_location="cpu", weights_only=False)
    audio_keys = sum(1 for k in extras if "audio_encoder" in k)
    proj_keys = sum(1 for k in extras if "audio_proj" in k)
    lora_keys = sum(1 for k in extras if "lora" in k.lower())

    cfg = json.load(open(os.path.join(local, "config.json")))
    ok = (
        hasattr(modeling, "AudioMosaicTransformer")
        and hasattr(modeling_audiomosaic_ltu, "AudioMosaicLlamaForCausalLM")
        and cfg.get("model_class") == "AudioMosaicLlamaForCausalLM"
        and audio_keys == 154 and proj_keys == 2 and lora_keys == 128
    )

    sys.path.remove(local)
    elapsed = time.time() - t0
    return {
        "repo": repo, "ok": ok,
        "class": "AudioMosaicLlamaForCausalLM",
        "shape": f"extras: audio_encoder={audio_keys}, proj={proj_keys}, lora={lora_keys}",
        "elapsed": elapsed,
    }


def main():
    results = []
    for repo, spec, expected in VISION_REPOS:
        try:
            r = test_vision(repo, spec, expected)
            mark = "✅" if r["ok"] else "❌"
            print(f"  {mark} {repo.split('/')[-1]:50s} [{r['class']}] -> {r['shape']}  ({r['elapsed']:.1f}s, miss={r['missing']}, unexp={r['unexpected']})")
            results.append(r)
        except Exception as e:
            print(f"  ❌ {repo.split('/')[-1]:50s} ERROR: {e}")
            results.append({"repo": repo, "ok": False, "error": str(e)})

    print()
    print("[LTU end-to-end repo]")
    try:
        r = test_ltu()
        mark = "✅" if r["ok"] else "❌"
        print(f"  {mark} {r['repo'].split('/')[-1]:50s} [{r['class']}] -> {r['shape']}  ({r['elapsed']:.1f}s)")
        results.append(r)
    except Exception as e:
        print(f"  ❌ ltu-stage4 ERROR: {e}")

    print()
    ok = sum(1 for r in results if r.get("ok"))
    print(f"=== {ok}/{len(results)} passed ===")


if __name__ == "__main__":
    main()
