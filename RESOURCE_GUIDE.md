# Resource Guide: Lighting & Robot Assets

## Where to Get Native Lighting Configurations

### 1. **HSSD (Habitat Synthetic Scenes Dataset)**
The HSSD scenes you're using don't include pre-configured lighting layouts by design - they expect you to add custom lighting.

**Official Source:**
- **GitHub:** https://github.com/3dlg-hcvc/hssd
- **Dataset:** https://github.com/3dlg-hcvc/hssd-models
- **Paper:** "HSSD: Habitat Synthetic Scenes Dataset"

**Note:** HSSD focuses on geometry and materials. Lighting is typically added programmatically (like we did).

### 2. **ReplicaCAD - Alternative with Lighting**
ReplicaCAD includes pre-configured lighting setups.

**Download:**
```bash
# From Habitat-Sim examples
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset --data-path data/
```

**Features:**
- Pre-configured lighting layouts
- High-quality photogrammetry
- Apartment-style scenes
- ~2GB download

**GitHub:** https://github.com/facebookresearch/habitat-sim/tree/main/src_python/habitat_sim/utils/datasets_download

### 3. **HM3D (Habitat-Matterport 3D)**
Production-quality scans with semantic annotations.

**Access:**
- Requires Matterport3D license
- Academic use free: https://matterport.com/habitat-matterport-3d-research-dataset
- Commercial use: Contact Matterport

**Features:**
- Real-world scanned environments
- Semantic segmentation
- Better lighting textures baked in

---

## Where to Get Better Robot Models

### 1. **Habitat-Sim Robot Assets (Current)**
You're using this already! Located at:
```
home-robot/data/robots/hab_stretch/
```

**To improve lighting on these:**
Create `.object_config.json` files for each mesh:

```bash
cd home-robot/data/robots/hab_stretch/meshes
```

Create `link_gripper_fingertip_right.object_config.json`:
```json
{
  "render_asset": "link_gripper_fingertip_right.obj",
  "requires_textures": true,
  "requires_lighting": true,
  "shader_type": "phong",
  "units_to_meters": 1.0
}
```

Repeat for all mesh files with warnings.

### 2. **Hello Robot Official Models**
**Source:** https://github.com/hello-robot/stretch_description

**Features:**
- Official Stretch robot URDF
- Updated regularly
- Better materials and textures

**Install:**
```bash
cd ~/Downloads
git clone https://github.com/hello-robot/stretch_description.git
# Copy to your project
cp -r stretch_description/urdf/* ~/InteractiveRobotics/home-robot/data/robots/
```

### 3. **Habitat 2.0/3.0 Robot Models**
Newer versions with better rendering support.

**GitHub:** https://github.com/facebookresearch/habitat-lab/tree/main/data/robots

**Available Robots:**
- Fetch robot (mobile manipulator)
- Franka Panda (arm)
- Spot (Boston Dynamics)
- Updated Stretch models

**Download specific robots:**
```bash
wget https://dl.fbaipublicfiles.com/habitat/robots/fetch_robot.zip
unzip fetch_robot.zip -d data/robots/
```

### 4. **Drake Robotics Models**
High-quality robot models with materials.

**Source:** https://github.com/RobotLocomotion/drake

**Models Include:**
- Boston Dynamics Spot
- UR10 arms
- Allegro hands
- Many others with proper PBR materials

---

## Alternative Complete Scene Datasets

### 1. **Gibson Dataset**
- Real scanned environments
- Pre-processed for navigation
- 572 buildings

**Access:** http://gibsonenv.stanford.edu/database/

### 2. **Replica Dataset**
- High-quality reconstruction
- 18 scenes
- Good lighting/textures

**Download:**
```bash
python -m habitat_sim.utils.datasets_download --uids replica_dataset --data-path data/
```

### 3. **MP3D (Matterport3D)**
- Premium scanned environments
- 90 buildings
- Requires license

**More info:** https://niessner.github.io/Matterport/

---

## Creating Your Own Lighting Configurations

### Option 1: JSON Configuration Files
Create `{scene_id}.lighting_config.json`:

```json
{
  "lights": {
    "ambient": {
      "intensity": 0.6,
      "color": [1.0, 1.0, 1.0]
    },
    "directional": [
      {
        "intensity": 2.0,
        "color": [1.0, 0.95, 0.85],
        "direction": [-1.0, -1.5, -0.5]
      }
    ]
  }
}
```

### Option 2: Procedural (What We Did)
Keep using Python code for dynamic lighting - gives you most control!

---

## Recommended Setup for Production

**Best Quality:**
1. **Scenes:** ReplicaCAD or HM3D (if licensed)
2. **Robots:** Hello Robot official + Drake models
3. **Lighting:** Custom programmatic (like we implemented)

**Current Setup (Excellent for Development):**
- ✅ HSSD scenes (free, high-quality)
- ✅ Hab Stretch robot (functional)
- ✅ Custom 4-point lighting (professional)

Your current setup is **great for research and development**. The lighting warnings are minor cosmetic issues.

---

## Quick Commands

**Download ReplicaCAD with lighting:**
```bash
cd home-robot
python -m habitat_sim.utils.datasets_download --uids replica_cad_dataset
```

**Get updated Stretch robot:**
```bash
cd ~/Downloads
git clone https://github.com/hello-robot/stretch_description.git
```

**Check available Habitat datasets:**
```bash
python -m habitat_sim.utils.datasets_download --list
```

---

## Resources & Documentation

- **Habitat-Sim Docs:** https://aihabitat.org/docs/habitat-sim/
- **Lighting Tutorial:** https://aihabitat.org/docs/habitat-sim/lighting.html
- **Robot Assets:** https://github.com/facebookresearch/habitat-lab/tree/main/data
- **HSSD Paper:** https://arxiv.org/abs/2306.08949
- **Hello Robot:** https://hello-robot.com/

---

## Bottom Line

**For your current project:**
- Your setup is excellent ✅
- The 3 robot mesh warnings are cosmetic
- HSSD quality matches Facebook examples with our lighting

**To eliminate all warnings:**
1. Add `.object_config.json` files to robot meshes
2. Or use updated robot models from Hello Robot official repo

**For future projects:**
- Consider ReplicaCAD for pre-configured lighting
- Explore HM3D for photorealistic scans
- Check Drake for high-quality robot models
