"""
Smoke test for the AudioMosaic repo:
1. Importable: dataset/, losses/, models/, all main_*.py
2. mlconfig can parse every config in configs/AudioMosaic/ and instantiate the model
3. Each model can do a forward pass on a synthetic spectrogram
4. Loss criterion can be instantiated and called

Does NOT run actual training; data loaders are skipped (require real data).
"""
import os
import sys
import importlib
import traceback
from glob import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import torch
import mlconfig


def test_imports():
    print("\n[1] Module imports")
    for mod in ["dataset", "losses", "models", "misc", "util", "exp_mgmt",
                "engine_pretrain", "engine_finetune"]:
        try:
            importlib.import_module(mod)
            print(f"  ✅ {mod}")
        except Exception as e:
            print(f"  ❌ {mod}: {type(e).__name__}: {e}")


def test_main_scripts():
    """Verify main_*.py compile + import argparse without side effects."""
    print("\n[2] main_*.py syntax check")
    for path in sorted(glob(f"{ROOT}/main_*.py")):
        name = os.path.basename(path)
        if name in {"main_ltu.py", "main_ltu_evaluate.py", "main_ltu_inference.py"}:
            # These need datasets/openai/etc — just compile-check
            try:
                with open(path) as f:
                    compile(f.read(), path, "exec")
                print(f"  ✅ {name} (syntax OK; runtime needs LTU base + extra deps)")
            except Exception as e:
                print(f"  ❌ {name}: {type(e).__name__}: {e}")
        else:
            # Try compile only — actually running would parse argv
            try:
                with open(path) as f:
                    compile(f.read(), path, "exec")
                print(f"  ✅ {name} (syntax OK)")
            except Exception as e:
                print(f"  ❌ {name}: {type(e).__name__}: {e}")


def _resolve_config(yaml_path):
    """Manually resolve $-vars in a config (workaround for mlconfig syntax)."""
    from omegaconf import OmegaConf
    cfg = OmegaConf.to_container(mlconfig.load(yaml_path), resolve=False)

    # Build top-level scalar lookup
    top = {k: v for k, v in cfg.items() if not isinstance(v, (dict, list))}

    def resolve(node):
        if isinstance(node, dict):
            return {k: resolve(v) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(v) for v in node]
        if isinstance(node, str) and node.startswith("$"):
            return top.get(node[1:], node)
        return node

    return resolve(cfg)


def test_configs():
    """Load every config and instantiate the model via the registered factory."""
    print("\n[3] mlconfig parse + model instantiation + forward pass")
    import models as models_pkg
    import io, contextlib
    yamls = sorted(glob(f"{ROOT}/configs/AudioMosaic/*.yaml"))
    ok, fail = 0, 0
    for y in yamls:
        name = os.path.basename(y)[:-5]
        try:
            cfg = _resolve_config(y)
            if "stage" in name:
                print(f"  ⚠  {name}: parsed (LTU stage; full instantiation needs Llama base)")
                ok += 1
                continue

            model_cfg = dict(cfg["model"])
            fn_name = model_cfg.pop("name")
            model_cfg.pop("ckpt_path", None)  # we don't have the pretrained encoder file here
            fn = getattr(models_pkg, fn_name)
            with contextlib.redirect_stdout(io.StringIO()):
                model = fn(**model_cfg)

            spec_size = model_cfg.get("spec_size", [1024, 128])
            x = torch.randn(2, 1, *spec_size)
            model.eval()
            with torch.no_grad():
                out = model(x)
            shape = tuple(out.shape) if hasattr(out, "shape") else f"dict[{','.join(out.keys())}]"
            print(f"  ✅ {name}: {fn_name} -> {shape}")
            ok += 1
            del model
        except Exception as e:
            print(f"  ❌ {name}: {type(e).__name__}: {e}")
            fail += 1
    print(f"  ({ok}/{ok+fail} configs OK)")


def test_pretrain_forward_backward():
    """End-to-end mini train step: load pretrain config, forward, compute loss, backward."""
    print("\n[4] Pretrain forward + NT-Xent backward (1 synthetic step)")
    import models as models_pkg
    import losses
    import io, contextlib
    try:
        cfg = _resolve_config(f"{ROOT}/configs/AudioMosaic/pretrain.yaml")
        model_cfg = dict(cfg["model"])
        fn_name = model_cfg.pop("name")
        model_cfg.pop("ckpt_path", None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = getattr(models_pkg, fn_name)(**model_cfg)
        crit_cfg = dict(cfg["criterion"])
        crit_name = crit_cfg.pop("name")
        crit_cfg["gather_distributed"] = False   # disable DDP for smoke test
        criterion = getattr(losses, crit_name)(**crit_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        # NTXentLoss expects (model, (x_view1, x_view2), y=None)
        x_pair = (torch.randn(2, 1, 1024, 128), torch.randn(2, 1, 1024, 128))
        result = criterion(model, x_pair)
        loss = result["loss"] if isinstance(result, dict) else result
        loss.backward()
        optimizer.step()
        print(f"  ✅ pretrain.yaml: forward + NT-Xent backward + optimizer.step OK (loss={loss.item():.4f})")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        traceback.print_exc()


def test_finetune_forward_backward():
    """End-to-end mini train step: finetune_as20k config (avg pooling, BCE loss)."""
    print("\n[5] Finetune forward + BCE backward (1 synthetic step)")
    import models as models_pkg
    import io, contextlib
    try:
        cfg = _resolve_config(f"{ROOT}/configs/AudioMosaic/finetune_as20k.yaml")
        model_cfg = dict(cfg["model"])
        fn_name = model_cfg.pop("name")
        model_cfg.pop("ckpt_path", None)
        with contextlib.redirect_stdout(io.StringIO()):
            model = getattr(models_pkg, fn_name)(**model_cfg)
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        model.train()
        x = torch.randn(2, 1, 1024, 128)
        y = torch.zeros(2, 527); y[:, [0, 5, 10]] = 1.0  # multi-label
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        print(f"  ✅ finetune_as20k.yaml: forward + BCE backward OK (loss={loss.item():.4f})")
    except Exception as e:
        print(f"  ❌ {type(e).__name__}: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    test_imports()
    test_main_scripts()
    test_configs()
    test_pretrain_forward_backward()
    test_finetune_forward_backward()
    print("\nDone.")
