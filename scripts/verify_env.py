"""Verify the training environment before Day 2.

Run this in a Kaggle notebook after installing requirements.txt. It checks the
GPU is visible, the pinned libraries imported at the versions we pinned, and
that persistence paths exist. Prints a block to paste into PROGRESS.md's
frozen decisions.

    python scripts/verify_env.py
"""

import importlib
import os
import platform
import sys
from pathlib import Path

PINNED = {
    "transformers": "4.57.6",
    "datasets": "3.2.0",
    "accelerate": "1.2.1",
    "peft": "0.14.0",
    "bitsandbytes": "0.45.0",
    "trl": "0.13.0",
}

KAGGLE_WORKING = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")


def on_kaggle():
    return KAGGLE_WORKING.exists() or "KAGGLE_KERNEL_RUN_TYPE" in os.environ


def check_versions():
    print("Pinned libraries")
    print("-" * 52)
    ok = True
    found = {}
    for name, want in PINNED.items():
        try:
            mod = importlib.import_module(name)
            got = getattr(mod, "__version__", "unknown")
            found[name] = got
            if got == want:
                print(f"  ok    {name:<16} {got}")
            else:
                print(f"  DRIFT {name:<16} {got}  (pinned {want})")
                ok = False
        except ImportError:
            print(f"  MISS  {name:<16} not installed")
            found[name] = None
            ok = False
    return ok, found


def check_gpu():
    print("\nGPU")
    print("-" * 52)
    try:
        import torch
    except ImportError:
        print("  MISS  torch not installed")
        return False, {}

    print(f"  torch            {torch.__version__}")
    if not torch.cuda.is_available():
        print("  FAIL  no CUDA device")
        print("        Kaggle: Settings > Accelerator > GPU P100 (or T4 x2)")
        return False, {"torch": torch.__version__}

    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  devices          {count} x {name}")
    print(f"  memory           {total:.1f} GiB each")

    info = {"torch": torch.__version__, "gpu": f"{count}x {name}", "vram_gib": round(total, 1)}

    # Two 4-bit models plus LoRA is tight on a single 16 GiB card.
    # Worth knowing on Day 1, not mid-run on Day 4.
    if count >= 2:
        print(f"  note             {count} GPUs — Qwen and Llama CAN run in parallel")
    elif total < 20:
        print(f"  note             run Qwen and Llama SEQUENTIALLY on {total:.0f} GiB")

    if "P100" in name:
        print("  note             P100 has no bf16 — use fp16 compute dtype")
        info["compute_dtype"] = "fp16"
    else:
        info["compute_dtype"] = "bf16"

    return True, info


def check_bnb(compute_dtype="fp16"):
    print("\n4-bit quantization")
    print("-" * 52)
    try:
        import torch
        from transformers import BitsAndBytesConfig

        dtype = torch.float16 if compute_dtype == "fp16" else torch.bfloat16
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        print(f"  ok    BitsAndBytesConfig constructs (nf4, {compute_dtype} compute)")
        return True
    except Exception as exc:  # noqa: BLE001 - surface whatever went wrong
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        return False


def check_persistence():
    print("\nPersistence")
    print("-" * 52)

    if not on_kaggle():
        print("  note  not on Kaggle (expected when running locally)")
        return True

    print(f"  ok    working dir  {KAGGLE_WORKING}")
    print("  WARN  /kaggle/working is WIPED when the session ends.")
    print("        Push adapters to a Kaggle Dataset before the session closes:")
    print("        python scripts/push_artifacts.py --name qwen-v1 --path /kaggle/working/qwen-v1")

    if KAGGLE_INPUT.exists():
        attached = sorted(p.name for p in KAGGLE_INPUT.iterdir())
        if attached:
            print(f"  ok    attached inputs: {', '.join(attached)}")
        else:
            print("  note  no datasets attached yet")

    # Kaggle credentials — presence only, never print the value.
    cred = Path.home() / ".kaggle" / "kaggle.json"
    if cred.exists() or "KAGGLE_KEY" in os.environ:
        print("  ok    Kaggle API credentials found")
    else:
        print("  MISS  no Kaggle API credentials — needed to push artifacts")
        print("        Add via notebook Secrets, or place ~/.kaggle/kaggle.json")

    return True


def check_session_budget():
    """Kaggle gives 30 GPU-hours/week. Eight training runs must fit."""
    print("\nSession budget")
    print("-" * 52)
    print("  Kaggle free tier: 30 GPU-hours/week, 9h max per session")
    print("  Sprint needs 8 training runs (2 models x 4 versions)")
    print("  Budget ~3h/run to stay inside the weekly quota")
    print("  Check your remaining quota at kaggle.com/settings")


def main():
    print(f"Python {platform.python_version()} on {platform.system()}")
    print(f"Platform: {'Kaggle' if on_kaggle() else 'local'}\n")

    versions_ok, found = check_versions()
    gpu_ok, gpu_info = check_gpu()
    bnb_ok = check_bnb(gpu_info.get("compute_dtype", "fp16"))
    check_persistence()
    check_session_budget()

    print("\n" + "=" * 52)
    if versions_ok and gpu_ok and bnb_ok:
        print("Environment verified. Record this in PROGRESS.md:\n")
        for k, v in found.items():
            print(f"  {k}=={v}")
        for k, v in gpu_info.items():
            print(f"  {k}: {v}")
        return 0

    print("Environment NOT ready. Fix the FAIL/MISS/DRIFT lines above.")
    print("Do not start Day 2 on a drifted environment.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
