from __future__ import annotations

try:
    import torch
except ImportError as exc:
    raise SystemExit("PyTorch is not installed in this environment.") from exc

print(f"PyTorch: {torch.__version__}")
print(f"GPU available through torch.cuda: {torch.cuda.is_available()}")
print(f"ROCm/HIP build: {getattr(torch.version, 'hip', None)}")
if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    for index in range(torch.cuda.device_count()):
        print(f"Device {index}: {torch.cuda.get_device_name(index)}")
