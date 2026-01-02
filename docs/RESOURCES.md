# Resources & Assets Guide

## Datasets

### HSSD (Habitat Synthetic Scenes Dataset)
Current project uses HSSD for high-quality 3D environments.

**Features:**
- 168 total scenes available
- PBR materials with realistic lighting
- Modern residential environments
- Pre-configured in this project

**Location:** `data/scenes/hssd-hab/`

**Official Links:**
- GitHub: https://github.com/3dlg-hcvc/hssd
- Dataset: https://github.com/3dlg-hcvc/hssd-models
- Paper: "HSSD: Habitat Synthetic Scenes Dataset"

---

## Robot Models

### Stretch Robot
Official Facebook/Meta Stretch robot model for Habitat-Sim.

**Location:** `data/robots/hab_stretch/`

**Contents:**
- URDF: Robot definition file
- Meshes: 3D geometry (*.obj, *.STL)
- Materials: Texture definitions (*.mtl)

**Verification:**
✅ Official version (MD5: c1964edf5b97e77315361feb17295e46)

### Spot Robot
Boston Dynamics Spot robot with arm attachment.

**Location:** `data/robots/hab_spot_arm/`

---

## Humanoid Models

### SMPL-X Avatars
Parametric human body models with realistic proportions.

**Location:** `data/humanoids/`

**Available Models:**
- male_0, male_1
- female_0, female_1
- neutral_0

**Structure:**
- 54 joints
- 216 DOF (4 quaternion values per joint)
- Walking motion capture data (130 frames @ 30 FPS)

**Motion Data:** `data/humanoids/walking_motion_processed_smplx.pkl`

---

## Alternative Datasets

### ReplicaCAD (Optional)
High-quality photogrammetry scenes with pre-configured lighting.

**Download:**
```bash
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
```

**Features:**
- Pre-configured lighting layouts
- Apartment-style scenes
- ~2GB download

**GitHub:** https://github.com/facebookresearch/habitat-sim

### Matterport3D (Optional)
Real-world scanned environments.

**Note:** Requires academic license and registration.

---

## Adding Custom Assets

### Custom Scenes
Place scene dataset config in `data/scenes/` and update scene_id in demo code.

### Custom Robots
1. Create URDF file with meshes
2. Place in `data/robots/your_robot/`
3. Update robot path in demo configuration

### Custom Objects
Add URDF files to `data/objects/` for spawnable objects.

---

## External Resources

**Habitat-Sim Documentation:**
- https://aihabitat.org/docs/habitat-sim/

**Habitat-Lab (RL Framework):**
- https://github.com/facebookresearch/habitat-lab

**OVMM Challenge:**
- https://ovmm.github.io/

**Home-Robot Project:**
- https://github.com/facebookresearch/home-robot
