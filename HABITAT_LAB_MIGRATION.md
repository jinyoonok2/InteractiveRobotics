# Habitat-Lab Framework Migration

## Overview

This project now supports **two approaches** for humanoid control:

### 1. **Standalone Habitat-Sim** (`humanoid_exploration_demo.py`)
- ✅ Simple, lightweight
- ✅ Direct control, minimal setup
- ✅ Good for prototyping and visualization
- ❌ Not designed for RL policy training
- ❌ Manual animation system

### 2. **Habitat-Lab Framework** (`humanoid_habitat_lab.py`) ⭐ **Recommended**
- ✅ Proper task-based architecture
- ✅ Ready for policy learning (RL/IL)
- ✅ Action/observation framework
- ✅ Episode management
- ✅ Extensible for multi-agent scenarios
- ⚠️  More complex setup

## Quick Start

### Running the Habitat-Lab Demo

```bash
conda activate home-robot
python humanoid_habitat_lab.py
```

**Controls:**
- W/S - Move forward/backward
- A/D - Turn left/right  
- ESC - Exit

## Architecture

### Habitat-Lab Components

```
HumanoidLabDemo
├── Task (HumanoidExplorationTask)
│   ├── Actions (HumanoidMoveAction with animation)
│   ├── Sensors (RGB, Depth, etc.)
│   └── Measurements (distance, collision, etc.)
├── Environment (habitat.Env)
│   ├── Simulator (habitat-sim)
│   └── Dataset (scene configurations)
└── Agent (KinematicHumanoid)
    ├── SMPL-X body model
    └── Motion capture data
```

### Custom Action: HumanoidMoveAction

Located in `humanoid_habitat_lab.py`, this action:
- Moves humanoid with velocity control
- Automatically updates walking animation
- Handles NavMesh collision
- LERP interpolates between motion frames
- Fixes left arm with procedural swing

**Usage in policies:**
```python
action = {
    "action": "humanoid_move",
    "action_args": {
        "lin_vel": 0.1,  # -1.0 to 1.0
        "ang_vel": 0.0   # -1.0 to 1.0
    }
}
obs = env.step(action)
```

## For Policy Learning

### Adding RL Training

The habitat-lab structure supports standard RL libraries:

**1. Add observation space:**
```python
# In HumanoidExplorationTask
def get_observation(self):
    return {
        "rgb": self.sensor_observations["head_rgb_sensor"],
        "position": self.humanoid.translation,
        "rotation": self.humanoid.rotation,
    }
```

**2. Add reward function:**
```python
# In HumanoidExplorationTask  
def calculate_reward(self):
    # Distance to goal
    # Collision penalty
    # Smoothness reward
    return reward
```

**3. Connect to RL library:**
```python
import stable_baselines3 as sb3

# Wrap habitat env
gym_env = habitat.gym.make("HumanoidExploration-v0")
model = sb3.PPO("MultiInputPolicy", gym_env)
model.learn(total_timesteps=1000000)
```

### Multi-Agent Extension

Add robot collaboration:

```python
config.habitat.simulator.agents.update({
    "robot_agent": {
        "articulated_agent_urdf": "./data/robots/hab_stretch/urdf/hab_stretch.urdf",
        "articulated_agent_type": "MobileManipulator",
    }
})
```

## Migration Path

### From `humanoid_exploration_demo.py` → `humanoid_habitat_lab.py`

**Step 1:** Test current setup
```bash
python humanoid_habitat_lab.py
```

**Step 2:** Verify fallback mode works (it should automatically handle missing datasets)

**Step 3:** For full task mode, create episode dataset:
```python
# Create minimal episode file
episodes = [{
    "episode_id": "0",
    "scene_id": "102344280",
    "start_position": [0, 0, 0],
    "start_rotation": [0, 0, 0, 1],
}]

with gzip.open("data/episodes/humanoid_exploration.json.gz", "wt") as f:
    json.dump({"episodes": episodes}, f)
```

**Step 4:** Update config to use your dataset

## Next Steps

### 1. **Add More Actions**
```python
@registry.register_task_action
class HumanoidPickAction(SimulatorTaskAction):
    def step(self, object_id: int, **kwargs):
        # Implement pick logic
        pass
```

### 2. **Add Sensors**
```python
config.habitat.simulator.agents.main_agent.sim_sensors.update({
    "depth_sensor": {
        "type": "HabitatSimDepthSensor",
        "height": 256,
        "width": 256,
    }
})
```

### 3. **Add Measurements**
```python
@registry.register_measure
class DistanceToGoal(Measure):
    def update_metric(self, *args, **kwargs):
        current_pos = self._sim.humanoid.translation
        goal_pos = self._task.goal_position
        self._metric = (goal_pos - current_pos).length()
```

### 4. **Train Policies**
- Use PPO/SAC from stable-baselines3
- Or custom policy networks with PyTorch
- Leverage habitat-baselines for distributed training

## File Organization

```
InteractiveRobotics/
├── humanoid_exploration_demo.py      # Standalone (legacy)
├── humanoid_habitat_lab.py           # Framework (current)
├── humanoid_lab_step1.py             # Development step
├── robot_interaction_demo.py         # Robot-only demo
└── data/
    ├── humanoids/                     # SMPL-X models
    │   ├── male_1/
    │   └── walking_motion_processed_smplx.pkl
    └── scenes/                        # HSSD environments
```

## Troubleshooting

**Issue:** "Could not find dataset RearrangementDataset-v0"
- **Solution:** Demo automatically falls back to simplified mode ✅

**Issue:** Animation not smooth
- **Solution:** Check motion data file exists at `data/humanoids/walking_motion_processed_smplx.pkl`

**Issue:** Humanoid floating/falling
- **Solution:** NavMesh is being computed automatically, wait for "NavMesh computed" message

## Benefits of Habitat-Lab Framework

### For Research
- ✅ Reproducible experiments
- ✅ Standard evaluation metrics
- ✅ Episode management
- ✅ Multi-agent coordination

### For Development  
- ✅ Modular action/sensor system
- ✅ Easy to add new tasks
- ✅ Built-in logging and metrics
- ✅ Compatible with standard RL libraries

### For Deployment
- ✅ Sim-to-real transfer tools
- ✅ Policy export
- ✅ Batch evaluation
- ✅ Distributed training support

## Resources

- **Habitat-Lab Docs:** https://aihabitat.org/docs/habitat-lab/
- **KinematicHumanoid Tutorial:** https://aihabitat.org/docs/habitat-lab/humanoids.html
- **Social Rearrangement:** https://ai.meta.com/research/publications/habitat-3-0/
- **Our Docs:** [docs/SETUP.md](docs/SETUP.md), [docs/RESOURCES.md](docs/RESOURCES.md)
