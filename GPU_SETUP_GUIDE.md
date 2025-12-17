# GPU Setup Guide for Interactive Robotics

## Current Status
- **GPU Detected:** NVIDIA GeForce RTX 4060 Max-Q / Mobile
- **PyTorch Version:** 2.8.0+cu128 (CUDA 12.8 support built-in)
- **Problem:** CUDA drivers not installed, so GPU is not accessible

## Why GPU Matters
- **Habitat-Sim:** Can use GPU for faster rendering (10-100x speedup)
- **PyTorch Models:** Neural networks will run much faster
- **Real-time Simulation:** Smoother interactive demos

---

## Installation Steps

### Option 1: Install NVIDIA Drivers via Ubuntu Package Manager (Recommended)

1. **Check available drivers:**
   ```bash
   ubuntu-drivers devices
   ```

2. **Install recommended driver automatically:**
   ```bash
   sudo ubuntu-drivers autoinstall
   ```

   OR manually install a specific version (e.g., 550):
   ```bash
   sudo apt install nvidia-driver-550
   ```

3. **Reboot your system:**
   ```bash
   sudo reboot
   ```

4. **Verify installation:**
   ```bash
   nvidia-smi
   ```
   You should see GPU information and driver version.

### Option 2: Install Latest Drivers from NVIDIA (Advanced)

1. **Add NVIDIA repository:**
   ```bash
   sudo add-apt-repository ppa:graphics-drivers/ppa
   sudo apt update
   ```

2. **Install latest driver:**
   ```bash
   sudo apt install nvidia-driver-560  # or latest available
   ```

3. **Reboot and verify:**
   ```bash
   sudo reboot
   nvidia-smi
   ```

---

## Verify GPU in PyTorch

After installing drivers and rebooting, test GPU access:

```bash
conda activate home-robot
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA Available: True
GPU Name: NVIDIA GeForce RTX 4060 Max-Q / Mobile
```

---

## Configure Habitat-Sim for GPU

After GPU is working, Habitat-Sim will automatically use it for rendering.

Test with:
```bash
conda activate home-robot
python furnished_house_robot_demo.py
```

To force GPU usage in Habitat-Sim:
```python
import habitat_sim

sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.gpu_device_id = 0  # Use first GPU
sim_cfg.enable_physics = True
```

---

## Troubleshooting

### Issue: "nvidia-smi not found" after installation
- **Solution:** Reboot required after driver installation

### Issue: CUDA version mismatch
- **Current PyTorch:** Built with CUDA 12.8
- **Driver compatibility:** Need NVIDIA driver >= 525.60.13 for CUDA 12.8
- **Check:** `nvidia-smi` shows compatible CUDA version in top right

### Issue: GPU detected but PyTorch can't use it
```bash
# Reinstall PyTorch with CUDA support
conda activate home-robot
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### Issue: Out of memory errors
- **Reduce batch size** in your scripts
- **Monitor GPU usage:** `watch -n 1 nvidia-smi`
- RTX 4060 Mobile has 8GB VRAM (sufficient for most tasks)

---

## Performance Comparison

| Task | CPU Only | With GPU (RTX 4060) |
|------|----------|---------------------|
| Habitat-Sim Rendering | ~5-10 FPS | ~60-120 FPS |
| Neural Network Inference | ~100-500ms | ~10-50ms |
| Training (small model) | Hours | Minutes |

---

## Quick Check Script

Save as `test_gpu.py`:
```python
#!/usr/bin/env python3
import torch
import habitat_sim

print("=" * 60)
print("GPU Status Check")
print("=" * 60)

# PyTorch
print(f"\n1. PyTorch")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Device Count: {torch.cuda.device_count()}")

# Habitat-Sim
print(f"\n2. Habitat-Sim")
try:
    backend = habitat_sim.SimulatorBackend()
    print(f"   Backend: {backend}")
    print(f"   GPU Rendering: Supported")
except:
    print(f"   Status: Check installation")

print("\n" + "=" * 60)
```

Run with: `python test_gpu.py`

---

## Summary

1. **Install NVIDIA drivers:** `sudo ubuntu-drivers autoinstall`
2. **Reboot:** `sudo reboot`
3. **Verify:** `nvidia-smi` and `python test_gpu.py`
4. **Run demo:** `python furnished_house_robot_demo.py`

Your RTX 4060 is excellent for this work - you'll see significant speedups once drivers are installed!
