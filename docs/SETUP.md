# Setup & Troubleshooting Guide

## GPU Setup (Optional - for faster rendering)

### Current Status
- **GPU:** NVIDIA GeForce RTX 4060 Max-Q / Mobile
- **PyTorch:** 2.8.0+cu128 (CUDA 12.8 support)
- **Benefits:** 10-100x faster rendering with GPU acceleration

### Quick Install
```bash
# Install NVIDIA drivers automatically
sudo ubuntu-drivers autoinstall

# Reboot
sudo reboot

# Verify installation
nvidia-smi
```

For detailed GPU setup instructions, see the original `GPU_SETUP_GUIDE.md` in project history.

---

## SophusPy Compatibility (Resolved)

The project uses `sophuspy==1.2.0` for 3D transformations. If you encounter import errors:

### Symptoms
```
ModuleNotFoundError: No module named 'sophus'
```

### Solution
```bash
# Update to newer version
pip install sophuspy==1.2.0

# Or use the import pattern:
import sophuspy as sp  # Not 'sophus'
```

The environment is pre-configured correctly in `environment_backup.yml`.

---

## Common Issues

### 1. Missing Data Files
```bash
# Run the install script to download all datasets
./install.sh
```

### 2. CUDA Out of Memory
- Reduce scene complexity
- Close other GPU applications
- Use CPU rendering (automatic fallback)

### 3. Import Errors
```bash
# Verify conda environment is activated
conda activate interactive-robotics

# Reinstall dependencies
pip install -r requirements.txt
```

---

## System Requirements

**Minimum:**
- Ubuntu 20.04+
- Python 3.9
- 8GB RAM
- 10GB disk space

**Recommended:**
- NVIDIA GPU (RTX series)
- 16GB RAM
- 50GB disk space (with full datasets)

---

## Environment Setup

The project uses conda for environment management:

```bash
# Create environment
conda env create -f environment_backup.yml

# Activate
conda activate interactive-robotics

# Install additional packages
pip install -r requirements.txt
```

For automated setup, just run:
```bash
./install.sh
```
