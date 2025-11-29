# Interactive Robotics Simulation Environment

🤖 **A comprehensive interactive robotics simulation setup using Habitat-Sim, Habitat-Lab, and Home-Robot for research and development.**

## 🎯 Overview

This repository provides a complete interactive robotics simulation environment that combines:
- **Habitat-Sim**: Photorealistic 3D simulation engine with GPU acceleration
- **Habitat-Lab**: High-level framework for embodied AI tasks and benchmarks  
- **Home-Robot**: Robotics abstractions and real-world integration
- **PyBullet**: Physics-based interactive simulation (alternative/complementary)

## 🏗️ Architecture

```
Interactive Robotics Demo
        ↓
┌─────────────────────┐
│   Home-Robot        │ ← High-level robotics abstractions
│   (Repository)      │
└─────────────────────┘
        ↓
┌─────────────────────┐
│   Habitat-Lab       │ ← Task environments & RL training
│   (Repository)      │   
└─────────────────────┘
        ↓
┌─────────────────────┐
│   Habitat-Sim       │ ← Core 3D simulation engine
│   (Pip Package)     │
└─────────────────────┘
        ↓
┌─────────────────────┐
│   GPU Rendering     │ ← NVIDIA/AMD GPU acceleration
│   (Hardware)        │
└─────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- **OS**: Linux (Ubuntu 18.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA 11.7+ (or AMD GPU with ROCm)
- **Python**: 3.9+
- **Conda**: Miniconda/Anaconda installed

### Installation

1. **Clone the Home-Robot repository**:
   ```bash
   git clone --recurse-submodules https://github.com/facebookresearch/home-robot.git
   cd home-robot
   export HOME_ROBOT_ROOT=$(pwd)
   ```

2. **Create conda environment**:
   ```bash
   conda create -n home-robot python=3.9 cmake=3.14.0 -y
   conda activate home-robot
   ```

3. **Install Habitat-Sim** (headless for servers, with display for desktop):
   ```bash
   # For desktop with display:
   mamba install habitat-sim -c conda-forge -c aihabitat -y
   
   # For headless servers:
   mamba install habitat-sim headless -c conda-forge -c aihabitat -y
   ```

4. **Install Habitat-Lab and dependencies**:
   ```bash
   git submodule update --init --recursive src/third_party/habitat-lab
   pip install -e src/third_party/habitat-lab/habitat-lab
   pip install -e src/third_party/habitat-lab/habitat-baselines
   ```

5. **Install Home-Robot packages**:
   ```bash
   pip install -e src/home_robot
   pip install -e src/home_robot_hw
   pip install -e src/home_robot_sim  # If using simulation features
   ```

6. **Install additional dependencies**:
   ```bash
   pip install sophuspy==1.2.0 opencv-python numpy matplotlib
   ```

7. **Download datasets** (Complete research-grade setup):
   ```bash
   # Set up Git LFS for large file downloads
   git lfs install
   
   # Download comprehensive datasets using official script
   cd home-robot
   export HOME_ROBOT_ROOT=$(pwd)
   ./download_data.sh --yes
   
   # This downloads (~22GB total):
   # - HSSD photorealistic home scenes (20GB, 168 scenes)
   # - Interactive objects dataset (1.8GB, 3K+ objects) 
   # - OVMM task episodes (746MB, 41 episode files)
   # - Stretch robot model (47MB, official URDF + meshes)
   # - Basic test scenes (107MB, apartment/castle/van-gogh)
   ```
   
   **Alternative: Minimal setup** (if you want basic functionality only):
   ```bash
   # Download only test scenes for basic demos
   python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path home-robot/data/
   ```

## 🎮 Running the Demo

**Interactive Habitat Demo**:
```bash
cd /path/to/InteractiveRobotics
conda activate home-robot
python interactive_habitat_demo.py
```

**Controls**:
- `W/A/S/D`: Move forward/left/backward/right
- `Q/E`: Turn left/right  
- `R`: Move up
- `F`: Move down
- `ESC`: Exit

## 🔧 System Requirements

### Minimum Requirements
- **RAM**: 8GB+ (16GB+ recommended for large scenes)
- **GPU**: 4GB VRAM+ (8GB+ for complex scenes)
- **Storage**: 30GB+ free space (for complete dataset installation)
- **CPU**: Multi-core processor
- **Network**: Stable internet for 22GB+ download

### Recommended Requirements
- **RAM**: 16GB+ (32GB for intensive research)
- **GPU**: 8GB+ VRAM (RTX 3060+, RTX 4060+, or equivalent AMD)
- **Storage**: 100GB+ free space (datasets + workspace + future data)
- **CPU**: 8+ cores (for parallel processing)
- **Network**: Fast connection (datasets are large)

### Tested Hardware
- **GPU**: NVIDIA RTX 4060, AMD Radeon 780M
- **OS**: Ubuntu 20.04, Ubuntu 22.04
- **Python**: 3.9, 3.10, 3.11

## 📁 Project Structure

```
InteractiveRobotics/
├── README.md                      # This file
├── .gitignore                     # Prevents uploading large datasets
├── check_installation.py          # Installation verification script
├── interactive_habitat_demo.py    # Main interactive demo
└── home-robot/                    # Home-Robot repository
    ├── src/home_robot/           # Core robotics package
    ├── src/home_robot_sim/       # Simulation interfaces
    ├── src/home_robot_hw/        # Hardware interfaces
    ├── src/third_party/          # Third-party dependencies
    │   └── habitat-lab/          # Habitat-Lab framework
    └── data/                     # Datasets (22GB+, git-ignored)
        ├── hssd-hab/             # Photorealistic home scenes (20GB)
        │   ├── scenes/           # 168 house scene files
        │   └── stages/           # Scene geometry files
        ├── objects/              # Interactive objects (1.8GB)
        │   └── train_val/        # 3K+ household objects
        ├── datasets/ovmm/        # Task episodes (746MB)
        │   ├── train/            # Training episodes
        │   └── val/              # Validation episodes
        ├── robots/hab_stretch/   # Robot model (47MB)
        │   ├── urdf/             # Robot description files
        │   └── meshes/           # 3D robot meshes
        └── data/scene_datasets/  # Basic test scenes
            └── habitat-test-scenes/  # Apartment, castle, van-gogh
```

## 🛠️ Troubleshooting

### Common Issues

**1. ImportError: sophuspy version mismatch**
```bash
pip install sophuspy==1.2.0 --force-reinstall
```

**2. Habitat-Sim installation fails**
```bash
# Try conda-forge channel:
mamba install habitat-sim -c conda-forge -c aihabitat -y

# Or build from source:
git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim
pip install -r requirements.txt
python setup.py install --headless
```

**3. GPU not detected**
```bash
# Check CUDA installation:
nvidia-smi

# Check PyTorch CUDA support:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**4. Scene data not found**
```bash
# Download required scenes:
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path home-robot/data/
```

### Performance Optimization

**For better performance**:
- Use headless Habitat-Sim on servers
- Ensure GPU drivers are up to date
- Close unnecessary applications during simulation
- Use SSD storage for scene data

## 🧪 Verification

Run the installation checker to verify your setup:
```bash
python check_installation.py
```

This will test:
- ✅ Python environment
- ✅ Required packages and versions
- ✅ GPU acceleration
- ✅ Habitat-Sim functionality
- ✅ Scene data availability
- ✅ Interactive demo readiness

## 🗂️ **Dataset Overview**

After running `./download_data.sh`, you'll have:

| Dataset | Size | Count | Purpose |
|---------|------|-------|--------|
| **HSSD Scenes** | 20GB | 168 scenes | Photorealistic home environments for navigation |
| **Objects** | 1.8GB | 3,139 objects | Interactive household items for manipulation |
| **Episodes** | 746MB | 41 files | Pre-defined navigation and manipulation tasks |
| **Robot Model** | 47MB | 1 robot | Official Hello Robot Stretch URDF + meshes |
| **Test Scenes** | 107MB | 3 scenes | Basic apartment, castle, van-gogh environments |
| **Total** | **~22GB** | **38K+ assets** | **Complete robotics research environment** |

### **What Each Dataset Enables:**
- 🏠 **Navigation Research**: 168 photorealistic homes + basic test scenes
- 🎯 **Object Manipulation**: 3K+ realistic household objects
- 📋 **Task Learning**: Structured episodes for RL training  
- 🤖 **Robot Simulation**: Accurate Stretch robot with physics
- 🔬 **Benchmarking**: Standard datasets for research comparison

## 📚 Key Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
| habitat-sim | 0.3.0+ | 3D simulation engine |
| habitat-lab | 0.2.5+ | AI task framework |
| home-robot | 0.1.0+ | Robotics abstractions |
| sophuspy | 1.2.0 | Geometry transformations |
| opencv-python | Latest | Computer vision |
| numpy | Latest | Numerical computing |
| torch | Latest | ML framework (with CUDA) |

## 🔬 Research Applications

This environment supports:
- **Navigation Research**: PointNav, ObjectNav, ImageNav
- **Manipulation Tasks**: Pick & Place, Object Interaction
- **Embodied AI**: Vision-Language Navigation, Instruction Following
- **Multi-Agent Systems**: Collaborative robotics
- **Sim-to-Real Transfer**: Real robot deployment

## 📈 Next Steps

1. **Explore Embodiments**: Try different robot types (humanoid, quadruped, mobile manipulator)
2. **Add Tasks**: Implement custom navigation or manipulation tasks
3. **Integrate RL**: Add reinforcement learning training loops
4. **Real Robot**: Connect to actual hardware (Stretch, Spot, etc.)
5. **Custom Scenes**: Import your own 3D environments

## 🤝 Contributing

This is a research environment. Feel free to:
- Add new simulation scenarios
- Implement additional robot embodiments  
- Create custom tasks and benchmarks
- Optimize performance and add features

## 📄 License

Based on Home-Robot (MIT License) and Habitat (MIT License).

## 🙏 Acknowledgments

- **Meta AI Research** for Habitat-Sim and Habitat-Lab
- **Facebook Research** for Home-Robot
- **AI Habitat Community** for datasets and tools

---

🚀 **Happy Robot Simulation!** 🤖
