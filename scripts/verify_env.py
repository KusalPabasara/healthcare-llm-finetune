"""Verify the training environment before Day 2.

Run this in Colab after installing requirements.txt. It checks the GPU is
visible, the pinned libraries imported at the versions we pinned, and that
Drive is mounted. Prints a block to paste into PROGRESS.md's frozen decisions.

    python scripts/verify_env.py
"""

import importlib
import os
import platform
import sys
from pathlib import Path

DRIVE = Path("/content/drive/MyDrive/healthcare-llm")

# Packages whose exact version affects training results. Versions are read
# from requirements.txt rather than duplicated here: a hardcoded copy drifts
# the moment a pin changes, and then the checker reports a failure against a
# version nobody uses any more.
CRITICAL = ("transformers", "datasets", "accelerate", "peft", "bitsandbytes", "trl")

REQUIREMENTS = Path(__file__).resolve().parent.parent / "requirements.txt"


def load_pins(path: Path = REQUIREMENTS) -> dict:
    """Parse `name==version` lines from requirements.txt."""
    if not path.exists():
        sys.exit(f"error: {path} not found — run from the repo, not a copy of this script")

    pins = {}
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if "==" not in line:
            continue
        name, _, version = line.partition("==")
        name = name.strip().lower()
        if name in CRITICAL:
            pins[name] = version.strip()

    absent = [p for p in CRITICAL if p not in pins]
    if absent:
        sys.exit(f"error: {path.name} has no pin for: {', '.join(absent)}")
    return pins


PINNED = load_pins()


def on_colab():
    return Path("/content").exists() or "COLAB_GPU" in os.environ


def check_versions():
    print("Pinned libraries")
    print("-" * 52)
    ok = True
    found = {}
    for name, want in PINNED.items():
        try:
            mod = importlib.import_module(name)
        except ImportError:
            print(f"  MISS  {name:<16} not installed")
            found[name] = None
            ok = False
            continue
        except Exception as exc:  # noqa: BLE001
            # bitsandbytes raises non-ImportError when its CUDA binary is
            # missing. Reporting that as "not installed" sends you looking
            # for the wrong problem.
            print(f"  ERROR {name:<16} imported but broken: {type(exc).__name__}")
            found[name] = None
            ok = False
            continue

        got = getattr(mod, "__version__", "unknown")
        found[name] = got
        if got == want:
            print(f"  ok    {name:<16} {got}")
        else:
            print(f"  DRIFT {name:<16} {got}  (pinned {want})")
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
    if torch.version.cuda:
        print(f"  cuda             {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("  FAIL  no CUDA device")
        print("        Colab: Runtime > Change runtime type > T4 GPU")
        return False, {"torch": torch.__version__}

    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  devices          {count} x {name}")
    print(f"  memory           {total:.1f} GiB each")

    info = {"torch": torch.__version__, "gpu": f"{count}x {name}", "vram_gib": round(total, 1)}

    # Two 4-bit models plus LoRA is tight on a single 16 GiB card.
    # Worth knowing on Day 1, not mid-run on Day 4.
    if total < 20:
        print(f"  note             run Qwen and Llama SEQUENTIALLY on {total:.0f} GiB")

    # bf16 needs Ampere or newer. The T4 is Turing (compute 7.5), so fp16.
    supports_bf16 = torch.cuda.get_device_capability(0)[0] >= 8
    info["compute_dtype"] = "bf16" if supports_bf16 else "fp16"
    if not supports_bf16:
        print(f"  note             {name} has no bf16 — use fp16 compute dtype")

    return True, info


def check_bnb(compute_dtype="fp16"):
    """Actually quantize on the GPU.

    Constructing a BitsAndBytesConfig proves nothing — it is a dataclass and
    succeeds even when bitsandbytes has no CUDA binary. The only honest check
    is to put a real tensor through a 4-bit layer on the device.
    """
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
        print(f"  ok    BitsAndBytesConfig constructs (nf4, {compute_dtype})")
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  config: {type(exc).__name__}: {exc}")
        return False

    try:
        import bitsandbytes as bnb
        import torch
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL  bitsandbytes unusable: {type(exc).__name__}: {exc}")
        return False

    print(f"  ok    bitsandbytes    {getattr(bnb, '__version__', 'unknown')}")

    if not torch.cuda.is_available():
        print("  SKIP  no GPU — cannot verify quantization actually runs")
        return False

    try:
        linear = bnb.nn.Linear4bit(
            64, 64, bias=False, compute_dtype=torch.float16, quant_type="nf4"
        ).cuda()
        out = linear(torch.randn(2, 64, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        if out.shape != (2, 64):
            print(f"  FAIL  unexpected output shape {tuple(out.shape)}")
            return False
        print("  ok    4-bit matmul runs on GPU (nf4 Linear4bit verified)")
        return True
    except Exception as exc:  # noqa: BLE001
        msg = str(exc).split("\n")[0][:120]
        print(f"  FAIL  4-bit matmul: {type(exc).__name__}: {msg}")
        print("        bitsandbytes has no CUDA binary for this torch build.")
        print("        Fix: pip install -U 'bitsandbytes>=0.46.1'")
        return False


def check_persistence():
    print("\nPersistence")
    print("-" * 52)

    if not on_colab():
        print("  note  not on Colab (expected when running locally)")
        return True

    mount = Path("/content/drive/MyDrive")
    if not mount.exists():
        print("  FAIL  Drive is not mounted — nothing will survive a disconnect")
        print("        from google.colab import drive; drive.mount('/content/drive')")
        return False

    print(f"  ok    Drive mounted at {mount}")

    if DRIVE.exists():
        print(f"  ok    project dir  {DRIVE}")
        raw = DRIVE / "data" / "raw"
        if raw.exists():
            sets = sorted(p.name for p in raw.iterdir() if p.is_dir())
            if sets:
                print(f"  ok    staged data: {', '.join(sets)}")
    else:
        print(f"  note  {DRIVE} not created yet")

    print("  WARN  /content is wiped on disconnect. Checkpoint to Drive:")
    print(f"        output_dir='{DRIVE}/checkpoints/<artifact>', save_steps=100")
    return True


def check_session_budget():
    """Colab free tier disconnects. Plan around it rather than being surprised."""
    print("\nSession limits")
    print("-" * 52)
    print("  Colab free tier: ~4h typical, can disconnect sooner")
    print("  Sprint needs 8 training runs (2 models x 4 versions)")
    print("  Run models SEQUENTIALLY and checkpoint every ~100 steps to Drive")
    print("  On a drop, resume with resume_from_checkpoint=True")


def main():
    print(f"Python {platform.python_version()} on {platform.system()}")
    print(f"Platform: {'Colab' if on_colab() else 'local'}\n")

    versions_ok, found = check_versions()
    gpu_ok, gpu_info = check_gpu()
    bnb_ok = check_bnb(gpu_info.get("compute_dtype", "fp16"))
    check_persistence()
    check_session_budget()

    print("\n" + "=" * 52)

    # A VM reset restores Colab's stock packages: several pins drift together
    # and the ones Colab does not ship vanish. That looks alarming but the fix
    # is just reinstalling — distinguish it from a genuine partial install.
    stock_reset = sum(1 for v in found.values() if v is None) >= 2 and any(
        v is not None and v != PINNED[k] for k, v in found.items()
    )
    if stock_reset and not versions_ok:
        print("This looks like a fresh/reset Colab VM — stock packages are back.")
        print("Nothing is broken and Drive data is untouched. Reinstall:\n")
        print("  !pip install -r requirements.txt")
        print("\nThen restart the session and re-run this check.")
        print("=" * 52)
        return 1

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
