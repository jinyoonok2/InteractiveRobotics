# Interactive Robotics Simulation Environment

🤖 **An interactive robotics simulation environment featuring humanoid avatars and robots in photorealistic environments with physics-based animation, intelligent collision detection, and full manipulation capabilities.**

## ✨ Key Features

✅ **Humanoid Avatar Control**: SMPL-X humanoid models with walking animation and LERP interpolation  
✅ **Smart Wall Collision**: NavMesh-based collision with sliding along walls for smooth navigation  
✅ **HSSD High-Fidelity Rendering**: PBR materials + HBAO ambient occlusion for photorealistic visuals  
✅ **Robot Manipulation**: Stretch & Spot robots with fine joint control (keys 1-5) and snap-to-gripper  
✅ **Multi-Scene Support**: Choose from 8 curated environments or any of 168 HSSD scenes  
✅ **Dual Camera Modes**: First-person and third-person views with independent camera rotation  

## 🎯 Project Overview

This project demonstrates **humanoid avatar control and mobile manipulation** in photorealistic simulation environments. It combines:

- **Characters**: SMPL-X humanoid avatars with motion capture walking animations
- **Robots**: Stretch Robot (Hello Robot) and Spot Robot (Boston Dynamics) with full articulation
- **Simulation Engine**: Habitat-Sim 0.3.3 with Bullet Physics for realistic object dynamics
- **Environments**: HSSD dataset providing 168 photorealistic 3D-scanned home environments
- **Physics**: NavMesh collision detection with wall sliding and distance-based animation

### What We've Built

**1. Humanoid Avatar System**
- SMPL-X humanoid models (male_1, female_0, neutral_0) with 216 DOF (54 joints × 4 quaternions)
- Walking motion capture data (130 frames @ 30 FPS) with smooth LERP interpolation
- Distance-based animation speed matching: `frames += distance_moved × 30.0`
- Freeze-on-idle behavior (animation pauses when stationary)
- Procedural left arm swing workaround (motion capture issue)
- Proper height adjustment: NavMesh + 0.9m body center offset

**2. Advanced Physics & Collision**
- NavMesh-based collision detection with `try_step()` for wall sliding
- Characters slide along walls instead of stopping abruptly
- Proper height maintenance on uneven terrain
- Smart object and robot positioning at navigable points

**3. HSSD High-Fidelity Rendering**
- PBR (Physically-Based Rendering) material support
- HBAO (Horizon-Based Ambient Occlusion) for realistic shadows
- `override_scene_light_defaults=True` for improved lighting quality
- Optimized rendering for dark materials (robot visibility)

**4. Robot Control & Manipulation**
- Fine-grained joint control: Keys 1-5 for joints 0-4 (Shift+1-5 reverse)
- Alternative controls: J/K (lift), U/I (arm extension)
- Snap-to-gripper object manipulation (Habitat OVMM standard)
- Stretch and Spot robots placed as static scene objects
- Full 6-DOF movement with corrected coordinate system (W=forward +Z)

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

### 🎯 **Option A: Use This Repository (Recommended)**

⚠️ **IMPORTANT: You MUST use `--recurse-submodules` or the setup will be broken!**

**Clone this InteractiveRobotics repository with all dependencies:**
```bash
# ✅ CORRECT: Clone with home-robot submodule included
git clone --recurse-submodules https://github.com/jinyoonok2/InteractiveRobotics.git
cd InteractiveRobotics

# The home-robot directory is automatically included as a submodule
export HOME_ROBOT_ROOT=$(pwd)/home-robot
```

**❌ DON'T do this (will give you an empty home-robot directory):**
```bash
# This will NOT work - missing --recurse-submodules
git clone https://github.com/jinyoonok2/InteractiveRobotics.git  
```

**🔧 If you already cloned without `--recurse-submodules`, fix it:**
```bash
cd InteractiveRobotics
git submodule update --init --recursive
```

### 🔧 **Option B: Manual Home-Robot Setup**

1. **Clone the original Home-Robot repository**:
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

3. **Install Habitat-Sim WITH BULLET PHYSICS** (CRITICAL for robot visualization):
   ```bash
   # IMPORTANT: Use 'withbullet' for robot URDF loading support
   conda install habitat-sim withbullet -c aihabitat -c conda-forge -y
   
   # Verify Bullet Physics is enabled:
   python -c "import habitat_sim; print(f'Bullet Physics: {habitat_sim.built_with_bullet}')"
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

## 🎮 Running the Demos

### Humanoid Exploration Demo (humanoid_exploration_demo.py) - **PRIMARY**

**The main demo featuring humanoid avatar control with walking animation and scene exploration.**

```bash
conda activate interactive-robotics  # or home-robot if you used the old install
python humanoid_exploration_demo.py
```

**Features:**
- SMPL-X humanoid avatar control with realistic walking animation
- NavMesh collision detection with wall sliding
- Distance-based animation speed matching
- First-person and third-person camera modes (toggle with C)
- Stretch & Spot robots placed in the scene as static objects
- Scene selection menu (8 curated environments + custom option)

### Robot Interaction Demo (robot_interaction_demo.py)

**Dedicated robot manipulation demo with fine joint control.**

```bash
conda activate interactive-robotics
python robot_interaction_demo.py
```

**Features:**
- Controllable Stretch robot with full articulation
- Fine joint control (keys 1-5, Shift+1-5 reverse)
- Object spawning and snap-to-gripper manipulation
- Third-person follow camera with independent rotation
- Movement direction fixes (W=forward, S=backward)

### Verify Installation

```bash
python check_installation.py  # System verification
```

## 🕹️ Controls

### Humanoid Exploration Demo

**Movement (Humanoid Avatar):**
- `W` - Move forward (with wall sliding)
- `S` - Move backward
- `A` - Turn left
- `D` - Turn right
- `Q` - Strafe left
- `E` - Strafe right

**Camera:**
- `C` - Toggle camera mode (1st person / 3rd person)
- `Z` - Rotate camera left (3rd person only)
- `X` - Rotate camera right (3rd person only)

**Objects & Spawning:**
- `O` - Spawn object near player
- `H` - Spawn additional humanoid avatar
- `P` - Pick/drop object (when near static robots)

**System:**
- `ESC` - Exit demo

### Robot Interaction Demo

**Movement (Robot):**
- `W` - Move forward
- `S` - Move backward
- `A` - Turn left
- `D` - Turn right
- `Q` - Strafe left
- `E` - Strafe right

**Camera:**
- `C` - Toggle camera mode
- `Z` - Rotate camera left (3rd person)
- `X` - Rotate camera right (3rd person)

**Fine Joint Control:**
- `1-5` - Move joints 0-4 forward
- `Shift+1-5` - Move joints 0-4 reverse
- `J` - Lift arm down
- `K` - Lift arm up
- `U` - Retract arm
- `I` - Extend arm

**Manipulation:**
- `O` - Spawn new object
- `P` - Pick nearest object (<0.8m) / Drop held object
- `G` - Toggle gripper (open/close)
- `R` - Reset robot joints

**System:**
- `ESC` - Exit demo

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
├── README.md                           # This file
├── install.sh                          # Automated installation script
├── .gitignore                          # Prevents uploading large datasets
├── check_installation.py               # Installation verification script
├── humanoid_exploration_demo.py        # Main humanoid demo (PRIMARY)
├── robot_interaction_demo.py           # Robot manipulation demo
├── furnished_house_robot_demo.py       # Original reference demo
├── data/                               # Project data folder (git-ignored)
│   ├── scenes/                         # HSSD scenes
│   │   └── hssd-hab/                   # 168 photorealistic homes
│   ├── humanoids/                      # SMPL-X humanoid models
│   │   ├── male_1/                     # Male humanoid (URDF + meshes)
│   │   ├── female_0/                   # Female humanoid
│   │   └── walking_motion_processed_smplx.pkl  # Motion capture data
│   ├── robots/                         # Robot models
│   │   ├── hab_stretch/                # Stretch robot (URDF + meshes)
│   │   └── hab_spot_arm/               # Spot robot (URDF + meshes)
│   └── objects/                        # Interactive objects
└── home-robot/                         # Home-Robot repository (submodule)
    ├── src/home_robot/                 # Core robotics package (NOT USED)
    ├── src/home_robot_sim/             # Simulation interfaces (NOT USED)
    ├── assets/                         # Robot and object assets (REFERENCE)
    └── data/                           # Original data location (MOVED to ../data/)
```
        │   ├── scenes/                 # 168 house scene files (.scene_instance.json)
        │   └── stages/                 # Scene geometry files (.glb)
        ├── objects/                    # Interactive objects (1.8GB)
        │   └── train_val/              # 3K+ household objects
        ├── datasets/ovmm/              # Task episodes (746MB)
        │   ├── train/                  # Training episodes
        │   └── val/                    # Validation episodes
        ├── robots/hab_stretch/         # Robot model (47MB)
        │   ├── urdf/                   # Robot description files
        │   └── meshes/                 # 3D robot meshes
        └── scene_datasets/             # Basic test scenes
            └── habitat-test-scenes/    # Apartment, castle, van-gogh
```

## 🛠️ Technical Implementation

### Architecture
```
Robot Manipulation Demo
        ↓
┌─────────────────────┐
│   Home-Robot        │ ← Robot abstractions & URDF assets
│   (Submodule)       │
└─────────────────────┘
        ↓
┌─────────────────────┐
│   Habitat-Sim       │ ← 3D simulation engine + Bullet Physics
│   (with Bullet)     │
└─────────────────────┘
        ↓
┌─────────────────────┐
│   HSSD Dataset      │ ← 168 photorealistic scenes
│   (22GB data)       │
└─────────────────────┘
        ↓
┌─────────────────────┐
│   GPU Rendering     │ ← CUDA/ROCm acceleration
│   (Hardware)        │
└─────────────────────┘
```

### Key Components

**Lighting System:**
- 3-point balanced lighting (ambient 50%, sun 120%, fill 60%)
- Optimized to preserve robot's dark gray materials (Kd 0.326)
- Custom setup for HSSD scenes (which lack built-in lighting)

**Robot Control:**
- Stretch robot: 43 links, 14 controllable joints
- KINEMATIC motion type for predictable movement
- Gripper link ID: 36, Head link ID: 8

**Camera System:**
- Third-person follow camera
- 2.5m distance behind robot, 0.8m height
- Adjustable pitch range: -1.0 to 1.5 radians
- Uses Magnum quaternion math for smooth rotation

**Manipulation System:**
- Snap-to-gripper picking (Habitat OVMM research standard)
- Object tracking via gripper link absolute_translation
- KINEMATIC motion while held, DYNAMIC when dropped
- 0.8m detection radius for object picking

**Object Spawning:**
- Uses Habitat-Sim primitive shapes (spheres, cubes)
- Fallback to URDF block assets if primitives fail
- DYNAMIC motion type with mass=0.5kg

## 🚀 Current Progress

### ✅ Completed Features

**Humanoid Animation System:**
- [x] SMPL-X humanoid models (male_1, female_0, neutral_0) loaded
- [x] Walking motion capture data (130 frames @ 30 FPS)
- [x] LERP interpolation for smooth 60Hz→30Hz animation sync
- [x] Distance-based animation speed: `frames += distance × 30.0`
- [x] Freeze-on-idle behavior (animation pauses when stationary)
- [x] Proper height offset (NavMesh + 0.9m body center)
- [x] Procedural left arm swing (workaround for frozen mocap data)

**Environment & Physics:**
- [x] NavMesh-based collision detection with `try_step()`
- [x] Wall sliding (characters slide along walls, not stop)
- [x] HSSD high-fidelity rendering (PBR + HBAO)
- [x] Smart object/robot positioning at navigable points
- [x] 8 curated scenes + custom scene ID support

**Robot Control:**
- [x] Fine joint control (keys 1-5, Shift+1-5 reverse)
- [x] Alternative controls (J/K for lift, U/I for arm)
- [x] Movement direction fixes (W=forward +Z, S=backward -Z)
- [x] Snap-to-gripper object manipulation
- [x] Stretch & Spot robots as static scene objects

**Camera System:**
- [x] First-person and third-person modes
- [x] Independent camera rotation (Z/X keys)
- [x] Smooth follow camera for humanoid/robot
- [x] Toggle between modes (C key)

### 🔄 Known Issues

**Humanoid Animation:**
- ⚠️ Left arm frozen in motion capture data
- ✓ Workaround: Procedural quaternion-based arm swing (DOF 64-67)
- ⚠️ KinematicHumanoid helper incompatible (version mismatch - abandoned)
- ✓ Using manual LERP pose updates instead

### 📋 Future Work

**Advanced Features:**
- [ ] Integration with Habitat OVMM tasks
- [ ] Navigation planning (A* or similar)
- [ ] Semantic scene understanding
- [ ] Multi-robot scenarios
- [ ] Real-world robot deployment bridge

**Visual Improvements:**
- [ ] Higher quality rendering settings
- [ ] Shadow improvements
- [ ] Post-processing effects
- [ ] Better material rendering

**User Experience:**
- [ ] GUI-based scene selection
- [ ] In-scene object browser
- [ ] Save/load robot states
- [ ] Replay/recording system

## 📊 Scene Recommendations

**Best for Presentations:**
- **Modern Apartment (102344280)**: Optimal lighting, compact layout, best robot visibility
- **Luxury House (103997792)**: Well-lit, spacious, good for navigation demos

**Good for Testing:**
- **Office Space (102344049)**: Larger area, tests lighting coverage
- **Contemporary Villa (102816036)**: Complex layout, multi-room navigation

**Challenging Environments:**
- **Modern Studio (102816318)**: Open space, minimal furniture
- **Coastal Home (104348276)**: Large windows, natural lighting challenges

## 🎯 Robot Visualization Breakthrough 

**⚠️ CRITICAL SOLUTION**: Robot visibility requires **Bullet Physics** enabled in Habitat-Sim!

### The Solution (Working Code)
```python
# 1. Enable Bullet Physics in simulator configuration
sim_cfg.enable_physics = True  # MUST be True for URDF loading

# 2. Use Articulated Object Manager for robot loading
ao_mgr = self.sim.get_articulated_object_manager()
robot_obj = ao_mgr.add_articulated_object_from_urdf(
    filepath="home-robot/data/robots/hab_stretch/urdf/hab_stretch.urdf",
    fixed_base=False,
    maintain_link_order=False,
    force_reload=True
)

# 3. Position and configure robot
robot_obj.translation = mn.Vector3(2.0, 0.0, 1.5)
robot_obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
```

### Key Requirements
- ✅ **Habitat-Sim with Bullet Physics**: `conda install habitat-sim withbullet -c aihabitat`
- ✅ **Physics Enabled**: `sim_cfg.enable_physics = True` in configuration
- ✅ **URDF File Access**: Proper path to `hab_stretch.urdf` and mesh files
- ✅ **Articulated Object Manager**: Use `add_articulated_object_from_urdf()` method

### Success Indicators
```
✅ Robot loaded successfully!
   Type: <class 'habitat_sim._ext.habitat_sim_bindings.ManagedBulletArticulatedObject'>
   Position: [2.0, 0.0, 1.5]
   🎯 Robot is now VISIBLE in the simulation!
```

## 🛠️ Troubleshooting

### Robot Visualization Issues

**1. "Not implemented in base PhysicsManager" Error**
```bash
# Install Bullet Physics version:
conda install habitat-sim withbullet -c aihabitat -c conda-forge
# Verify: python -c "import habitat_sim; print(habitat_sim.built_with_bullet)"
```

**2. URDF Loading Fails**
```bash
# Check URDF file exists:
ls home-robot/data/robots/hab_stretch/urdf/hab_stretch.urdf
# Check mesh files:
ls home-robot/data/robots/hab_stretch/meshes/
```

### Common Issues

**3. ImportError: sophuspy version mismatch**
```bash
pip install sophuspy==1.2.0 --force-reinstall
```

**4. Habitat-Sim installation fails**
```bash
# Install with Bullet Physics support:
conda install habitat-sim withbullet -c aihabitat -c conda-forge -y
```

**5. GPU not detected**
```bash
# Check CUDA installation:
nvidia-smi
# Check PyTorch CUDA support:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**6. Scene data not found**
```bash
# Download required scenes:
./download_data.sh  # Or manual download
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

## 🔬 Research Context

### Habitat OVMM (Open Vocabulary Mobile Manipulation)

This project's manipulation system is inspired by the **Habitat OVMM** benchmark, which uses:
- **Snap-to-gripper**: "Magic" grasping where objects teleport to gripper (research standard)
- **KINEMATIC attachment**: Object motion type changes while held
- **Action space**: SNAP_OBJECT and DESNAP_OBJECT discrete actions

**Why "magic" grasping?**
- Focuses on high-level planning and navigation (not low-level motor control)
- Eliminates physics-based grasping failures
- Standard in embodied AI research for reproducibility
- Sim-to-real transfer handled separately in real robot deployment

### Research Applications

This environment supports:
- **Navigation Research**: PointNav, ObjectNav, ImageNav tasks
- **Manipulation Tasks**: Pick & Place, Object Rearrangement
- **Embodied AI**: Vision-Language Navigation, Instruction Following
- **Multi-Agent Systems**: Collaborative robotics scenarios
- **Sim-to-Real Transfer**: Bridge to real Stretch robots

## 📈 Performance Metrics

**Tested Configuration:**
- **GPU**: NVIDIA RTX 4060 Laptop (8GB VRAM)
- **Driver**: 580.95.05
- **FPS**: 60+ with Modern Apartment scene
- **RAM Usage**: ~4-6GB
- **VRAM Usage**: ~2-3GB

**Scene Complexity Impact:**
- Small scenes (apartments): 60+ FPS
- Large scenes (offices, villas): 45-60 FPS
- With multiple objects: Minimal FPS impact

## 📚 Key Dependencies

| Package | Version | Purpose |
|---------|---------|--------|
| habitat-sim | 0.3.3 | 3D simulation engine with Bullet Physics |
| habitat-lab | 0.2.5+ | AI task framework |
| home-robot | 0.1.0+ | Robotics abstractions and Stretch URDF |
| magnum | Latest | 3D graphics and math library |
| sophuspy | 1.2.0 | Lie group transformations |
| opencv-python | Latest | Computer vision and display |
| numpy | Latest | Numerical computing |
| torch | 2.8.0+ | ML framework with CUDA support |

## 🎓 Learning Resources

**Habitat Documentation:**
- [Habitat-Sim Docs](https://aihabitat.org/docs/habitat-sim/)
- [Habitat-Lab Docs](https://aihabitat.org/docs/habitat-lab/)

**Home-Robot:**
- [GitHub Repository](https://github.com/facebookresearch/home-robot)
- [Documentation](https://facebookresearch.github.io/home-robot/)

**Research Papers:**
- Habitat 2.0: Training Home Assistants to Rearrange Objects
- OVMM: Open Vocabulary Mobile Manipulation

## 📈 Next Steps

**Immediate Improvements:**
1. **Enhanced Lighting**: Dynamic adjustment for large scenes
2. **More Objects**: Custom URDF models and realistic household items
3. **Better Physics**: Collision-based grasping option
4. **GUI Controls**: Scene selection and object browser in-simulation

**Advanced Features:**
1. **Task Integration**: Implement Habitat OVMM tasks
2. **Navigation Planning**: A* pathfinding integration
3. **Semantic Mapping**: 3D semantic scene understanding
4. **Multi-Robot**: Collaborative manipulation scenarios
5. **Real Robot Bridge**: Deploy to actual Stretch robot

## 🤝 Contributing

Contributions welcome! Areas of interest:
- Additional robot models (UR5, Fetch, etc.)
- Custom manipulation tasks
- Lighting improvements for large scenes
- Performance optimizations
- Real robot deployment examples

## 📄 License

Based on:
- **Home-Robot**: MIT License (Facebook Research)
- **Habitat**: MIT License (Meta AI Research)
- **HSSD Dataset**: CC BY 4.0

## 🙏 Acknowledgments

- **Meta AI Research** - Habitat-Sim and Habitat-Lab frameworks
- **Facebook Research** - Home-Robot robotics platform
- **Hello Robot Inc.** - Stretch robot design and URDF
- **AI Habitat Community** - HSSD datasets and research tools

---

**For questions or issues**, please open an issue on GitHub.

🚀 **Happy Robot Simulation!** 🤖
