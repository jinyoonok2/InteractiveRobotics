#!/usr/bin/env python3
"""
Humanoid Exploration Demo - Habitat-Lab Framework
Uses proper agent configuration and KinematicHumanoid for realistic animation
"""

import os
import numpy as np
import cv2
import magnum as mn
from typing import Any, Dict, Optional

import habitat
from habitat.core.env import Env
from habitat_sim.physics import MotionType
from habitat.articulated_agents.humanoids import KinematicHumanoid
from omegaconf import DictConfig, OmegaConf
import habitat_sim

class HumanoidExplorationEnv(Env):
    """Custom Habitat-Lab environment for humanoid exploration"""
    
    def __init__(self, config: DictConfig, dataset: Optional[Any] = None):
        super().__init__(config, dataset)
        self.humanoid_controller = None
        self.camera_mode = 'third_person'
        self.camera_yaw = np.pi  # Behind humanoid
        
    def reset(self):
        """Reset environment and initialize humanoid"""
        observations = super().reset()
        
        # Get the humanoid agent
        if hasattr(self._sim, 'agents_mgr'):
            for agent_idx in range(len(self._sim.agents_mgr)):
                agent = self._sim.agents_mgr[agent_idx]
                agent_config = self._config.agents[agent_idx]
                
                # Check if this is a humanoid agent
                if 'humanoid' in agent_config.articulated_agent_type.lower():
                    # Create KinematicHumanoid controller
                    self.humanoid_controller = KinematicHumanoid(
                        agent_config,
                        self._sim,
                        limit_robo_joints=False,
                        fixed_base=False
                    )
                    print(f"✅ KinematicHumanoid controller initialized")
                    break
        
        return observations
    
    def step(self, action: Dict[str, Any]) -> Dict:
        """Execute action and return observations"""
        observations = super().step(action)
        return observations


def create_humanoid_config(scene_id: str = "102344280") -> DictConfig:
    """Create habitat-lab configuration for humanoid exploration"""
    
    # Base habitat config
    config = DictConfig({
        "habitat": {
            "seed": 42,
            "env_task": "GymHabitatEnv",
            "env_task_gym_dependencies": [],
            "simulator": {
                "type": "Sim-v0",
                "action_space_config": "v0",
                "forward_step_size": 0.25,
                "turn_angle": 10,
                "habitat_sim_v0": {
                    "gpu_device_id": 0,
                    "allow_sliding": True,
                    "enable_physics": True,
                    "physics_config_file": "./data/default.physics_config.json",
                },
                "agents": [{
                    "agent_0": {
                        "height": 1.5,
                        "radius": 0.3,
                        "articulated_agent_urdf": "./data/humanoids/male_1/male_1.urdf",
                        "articulated_agent_type": "KinematicHumanoid",
                        "ik_arm_urdf": "./data/humanoids/male_1/male_1.urdf",
                        "motion_data_path": "./data/humanoids/walking_motion_processed_smplx.pkl",
                        "auto_update_sensor_transform": True,
                        "sensors": {
                            "head_rgb": {
                                "type": "HabitatSimRGBSensor",
                                "height": 720,
                                "width": 1280,
                                "position": [0.0, 1.6, 0.0],
                                "orientation": [0.0, 0.0, 0.0],
                            },
                            "third_rgb": {
                                "type": "HabitatSimRGBSensor", 
                                "height": 720,
                                "width": 1280,
                                "position": [0.0, 2.5, 2.5],
                                "orientation": [-45.0, 0.0, 0.0],
                            }
                        }
                    }
                }],
            },
            "task": {
                "type": "RearrangementTask-v0",
                "reward_measure": "distance_to_goal",
                "success_measure": "spl",
                "success_reward": 2.5,
                "slack_reward": -0.01,
                "measurements": {},
                "actions": {
                    "base_velocity": {
                        "type": "BaseVelocityAction",
                        "lin_speed": 10.0,
                        "ang_speed": 10.0,
                        "allow_dyn_slide": True,
                        "allow_back": True,
                    }
                },
            },
            "dataset": {
                "type": "RearrangementDataset-v0",
                "split": "train",
                "data_path": f"data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json",
                "scenes_dir": "data/scenes/",
            }
        }
    })
    
    return config


class InteractiveHumanoidDemo:
    """Interactive demo using habitat-lab framework"""
    
    def __init__(self, scene_id: str = "102344280"):
        self.scene_id = scene_id
        self.env = None
        self.running = True
        
    def setup(self):
        """Initialize the habitat-lab environment"""
        print("🤖 Setting up Habitat-Lab Humanoid Demo...")
        
        # Create configuration
        config = create_humanoid_config(self.scene_id)
        
        # Create environment
        print("  📦 Creating environment...")
        try:
            self.env = HumanoidExplorationEnv(config=config)
            print("  ✅ Environment created")
        except Exception as e:
            print(f"  ❌ Failed to create environment: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Reset environment
        print("  🔄 Resetting environment...")
        try:
            obs = self.env.reset()
            print("  ✅ Environment reset")
            print(f"  📊 Observation keys: {list(obs.keys())}")
        except Exception as e:
            print(f"  ❌ Failed to reset environment: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def run(self):
        """Main interaction loop"""
        if not self.setup():
            print("❌ Setup failed!")
            return
        
        print("\n" + "="*60)
        print("🎮 Interactive Controls:")
        print("   W/S - Move forward/backward")
        print("   A/D - Turn left/right")
        print("   Q/E - Strafe left/right")
        print("   C - Toggle camera")
        print("   ESC - Exit")
        print("="*60 + "\n")
        
        while self.running:
            # Get current observations
            obs = self.env._sim.get_sensor_observations()
            
            # Display observation
            if "head_rgb" in obs:
                frame = obs["head_rgb"]
            elif "third_rgb" in obs:
                frame = obs["third_rgb"]
            else:
                # Fallback - get any RGB sensor
                rgb_keys = [k for k in obs.keys() if 'rgb' in k.lower()]
                if rgb_keys:
                    frame = obs[rgb_keys[0]]
                else:
                    print("⚠️ No RGB sensor found")
                    break
            
            # Convert to BGR for OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add UI
            cv2.putText(frame, "Habitat-Lab Humanoid Demo", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Humanoid Exploration", frame)
            
            # Handle input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.running = False
            elif key == ord('w'):
                # Move forward
                action = {"action": "base_velocity", "action_args": {"base_vel": [0.1, 0.0]}}
                self.env.step(action)
            elif key == ord('s'):
                # Move backward
                action = {"action": "base_velocity", "action_args": {"base_vel": [-0.1, 0.0]}}
                self.env.step(action)
            elif key == ord('a'):
                # Turn left
                action = {"action": "base_velocity", "action_args": {"base_vel": [0.0, 0.1]}}
                self.env.step(action)
            elif key == ord('d'):
                # Turn right
                action = {"action": "base_velocity", "action_args": {"base_vel": [0.0, -0.1]}}
                self.env.step(action)
        
        cv2.destroyAllWindows()
        self.env.close()
        print("\n✅ Demo completed!")


def main():
    print("🚀 Starting Habitat-Lab Humanoid Exploration Demo...")
    print("This version uses the full habitat-lab framework")
    print("with proper KinematicHumanoid support\n")
    
    # Scene selection
    SCENES = {
        '1': {'id': '102344280', 'name': 'Modern Apartment'},
        '2': {'id': '102344250', 'name': 'Luxury House'},
        '3': {'id': '102344049', 'name': 'Office Space'},
        '4': {'id': '102344403', 'name': 'Large Villa'},
    }
    
    print("="*60)
    print("🏠 SELECT ENVIRONMENT:")
    print("="*60)
    for key, scene in SCENES.items():
        print(f"   {key}. {scene['name']} (ID: {scene['id']})")
    print("="*60)
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice in SCENES:
        scene_id = SCENES[choice]['id']
        print(f"\n✅ Selected: {SCENES[choice]['name']}")
    else:
        print("\n⚠️ Invalid choice, using default (Modern Apartment)")
        scene_id = '102344280'
    
    try:
        demo = InteractiveHumanoidDemo(scene_id=scene_id)
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
