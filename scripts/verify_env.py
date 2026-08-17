"""Verify the training environment before Day 2.

Run this in Colab after installing requirements.txt. It checks that the GPU is
visible, the pinned libraries actually imported at the versions we pinned, and
Drive is mounted. Prints a block to paste into PROGRESS.md's frozen decisions.

    python scripts/verify_env.py
"""

import importlib
import platform
import sys

PINNED = {
    "transformers": "4.57.6",
    "datasets": "3.2.0",
    "accelerate": "1.2.1",
    "peft": "0.14.0",
    "bitsandbytes": "0.45.0",
    "trl": "0.13.0",
}


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
        print("  FAIL  no CUDA device — enable the GPU runtime in Colab")
        return False, {"torch": torch.__version__}

    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  device           {name}")
    print(f"  memory           {total:.1f} GiB")

    # Two 4-bit models plus LoRA is tight on a 16 GiB card. Worth knowing on
    # Day 1 rather than discovering it mid-run on Day 4.
    if total < 20:
        print(f"  note             run Qwen and Llama SEQUENTIALLY on {total:.0f} GiB")
    return True, {"torch": torch.__version__, "gpu": name, "vram_gib": round(total, 1)}


def check_bnb():
    print("\n4-bit quantization")
    print("-" * 52)
    try:
        import torch
        from transformers import BitsAndBytesConfig

        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        print("  ok    BitsAndBytesConfig constructs (nf4, bf16 compute)")
        return True
    except Exception as exc:  # noqa: BLE001 - surface whatever went wrong
        print(f"  FAIL  {type(exc).__name__}: {exc}")
        return False


def check_drive():
    print("\nDrive")
    print("-" * 52)
    from pathlib import Path

    p = Path("/content/drive/MyDrive")
    if p.exists():
        print(f"  ok    mounted at {p}")
        return True
    print("  note  not mounted (expected if running outside Colab)")
    return True


def main():
    print(f"Python {platform.python_version()} on {platform.system()}\n")

    versions_ok, found = check_versions()
    gpu_ok, gpu_info = check_gpu()
    bnb_ok = check_bnb()
    check_drive()

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
