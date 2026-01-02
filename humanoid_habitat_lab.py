#!/usr/bin/env python3
"""
Humanoid Exploration with Full Habitat-Lab Framework
Proper task-based architecture for policy learning and complex interactions
"""

import os
import numpy as np
import cv2
import pickle
import magnum as mn
from typing import Any, Dict, List, Optional

import habitat
import habitat_sim
from habitat.core.registry import registry
from habitat.core.simulator import Observations
from habitat.core.embodied_task import SimulatorTaskAction
from habitat.tasks.rearrange.rearrange_task import RearrangeTask
from habitat.articulated_agents.humanoids import KinematicHumanoid
from habitat.config.default_structured_configs import (
    ThirdRGBSensorConfig,
    HabitatSimV0Config,
)
from omegaconf import DictConfig, OmegaConf
from dataclasses import dataclass, field


@dataclass
class HumanoidAgentConfig:
    """Configuration for humanoid agent"""
    articulated_agent_urdf: str = "./data/humanoids/male_1/male_1.urdf"
    articulated_agent_type: str = "KinematicHumanoid"
    motion_data_path: str = "./data/humanoids/walking_motion_processed_smplx.pkl"
    ik_arm_urdf: str = "./data/humanoids/male_1/male_1.urdf"
    auto_update_sensor_transform: bool = True
    sim_sensors: Dict = field(default_factory=dict)
    height: float = 1.5
    radius: float = 0.3


@registry.register_task_action
class HumanoidMoveAction(SimulatorTaskAction):
    """Action for moving humanoid with walking animation"""
    
    def __init__(self, *args, config, sim: habitat_sim.Simulator, **kwargs):
        super().__init__(*args, config=config, sim=sim, **kwargs)
        self.lin_speed = getattr(config, "lin_speed", 0.25)
        self.ang_speed = getattr(config, "ang_speed", 10.0)
        self.humanoid = None
        self.motion_data = None
        self.motion_frame = 0.0
        self.num_frames = 0
        self.prev_position = None
        
    def _load_motion_data(self):
        """Load walking motion capture data"""
        if self.motion_data is not None:
            return
        
        motion_path = "./data/humanoids/walking_motion_processed_smplx.pkl"
        if os.path.exists(motion_path):
            with open(motion_path, 'rb') as f:
                data = pickle.load(f)
                if 'walk_motion' in data and 'joints_array' in data['walk_motion']:
                    self.motion_data = data['walk_motion']['joints_array']
                    self.num_frames = len(self.motion_data)
                    print(f"  ✅ Loaded motion data: {self.num_frames} frames")
    
    def _get_humanoid(self):
        """Get the humanoid articulated object"""
        if self.humanoid is not None:
            return self.humanoid
        
        ao_mgr = self._sim.get_articulated_object_manager()
        for i in range(ao_mgr.get_num_objects()):
            ao = ao_mgr.get_object_by_id(i)
            # Check if this is a humanoid (has many joints)
            if ao.num_links > 50:  # Humanoids have 54 links
                self.humanoid = ao
                self._load_motion_data()
                return ao
        return None
    
    def _update_animation(self, distance_moved: float):
        """Update walking animation based on movement"""
        if self.motion_data is None or distance_moved < 0.001:
            return
        
        # Advance animation proportional to distance
        frames_to_advance = distance_moved * 30.0
        self.motion_frame = (self.motion_frame + frames_to_advance) % self.num_frames
        
        # LERP between frames
        frame_a = int(np.floor(self.motion_frame))
        frame_b = int(np.ceil(self.motion_frame)) % self.num_frames
        alpha = self.motion_frame - frame_a
        
        pose_a = self.motion_data[frame_a].flatten()
        pose_b = self.motion_data[frame_b].flatten()
        blended_pose = (1.0 - alpha) * pose_a + alpha * pose_b
        
        # Apply pose to humanoid
        new_positions = list(self.humanoid.joint_positions)
        num_to_update = min(len(blended_pose), len(new_positions))
        for i in range(num_to_update):
            new_positions[i] = float(blended_pose[i])
        
        # Override left arm with procedural swing
        swing_amplitude = 0.5
        swing = np.sin(self.motion_frame * 0.2) * swing_amplitude
        left_shoulder_start = 64
        if left_shoulder_start + 3 < len(new_positions):
            angle = swing
            new_positions[left_shoulder_start] = np.sin(angle / 2)
            new_positions[left_shoulder_start + 1] = 0.0
            new_positions[left_shoulder_start + 2] = 0.0
            new_positions[left_shoulder_start + 3] = np.cos(angle / 2)
        
        self.humanoid.joint_positions = new_positions
    
    def step(self, lin_vel: float, ang_vel: float, **kwargs):
        """Execute movement with animation"""
        humanoid = self._get_humanoid()
        if humanoid is None:
            return {}
        
        # Calculate movement
        forward = humanoid.rotation.transform_vector(mn.Vector3(0, 0, 1))
        new_pos = humanoid.translation + forward * lin_vel * self.lin_speed
        
        # Apply rotation
        if abs(ang_vel) > 0.01:
            humanoid.rotation = humanoid.rotation * mn.Quaternion.rotation(
                mn.Rad(ang_vel * self.ang_speed * 0.01), mn.Vector3(0, 1, 0))
        
        # Collision detection with NavMesh
        if self._sim.pathfinder.is_loaded:
            final_pos = self._sim.pathfinder.try_step(humanoid.translation, new_pos)
            final_pos.y = final_pos.y + 0.9  # Body center offset
        else:
            final_pos = new_pos
        
        # Calculate distance moved
        current_pos = humanoid.translation
        if self.prev_position is None:
            self.prev_position = current_pos
        distance_moved = (final_pos - self.prev_position).length()
        self.prev_position = final_pos
        
        # Update position
        humanoid.translation = final_pos
        
        # Update animation if moving
        if distance_moved > 0.001:
            self._update_animation(distance_moved)
        
        return {}


@registry.register_task(name="HumanoidExploration-v0")
class HumanoidExplorationTask(RearrangeTask):
    """Custom task for humanoid exploration"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        print("  ✅ HumanoidExplorationTask initialized")


def create_humanoid_config(scene_id: str = "102344280") -> DictConfig:
    """Create full habitat-lab configuration"""
    
    config = OmegaConf.create({
        "habitat": {
            "seed": 42,
            "environment": {
                "max_episode_steps": 1000,
            },
            "simulator": {
                "type": "Sim-v0",
                "action_space_config": "v0",
                "forward_step_size": 0.25,
                "turn_angle": 10.0,
                "habitat_sim_v0": {
                    "gpu_device_id": 0,
                    "allow_sliding": True,
                    "enable_physics": True,
                },
                "scene": scene_id,
                "scene_dataset": "./data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json",
                "agents": {
                    "main_agent": {
                        "height": 1.5,
                        "radius": 0.3,
                        "articulated_agent_urdf": "./data/humanoids/male_1/male_1.urdf",
                        "articulated_agent_type": "KinematicHumanoid",
                        "sim_sensors": {
                            "head_rgb_sensor": {
                                "type": "HabitatSimRGBSensor",
                                "height": 720,
                                "width": 1280,
                                "position": [0.0, 1.6, 0.0],
                                "orientation": [0.0, 0.0, 0.0],
                            },
                            "third_rgb_sensor": {
                                "type": "HabitatSimRGBSensor",
                                "height": 720,
                                "width": 1280,
                                "position": [0.0, 2.5, 2.5],
                                "orientation": [-30.0, 0.0, 0.0],
                            },
                        },
                    },
                },
            },
            "task": {
                "type": "HumanoidExploration-v0",
                "lab_sensors": {},
                "measurements": {},
                "actions": {
                    "humanoid_move": {
                        "type": "HumanoidMoveAction",
                        "lin_speed": 0.1,
                        "ang_speed": 10.0,
                    },
                },
            },
            "dataset": {
                "type": "RearrangementDataset-v0",
                "split": "train",
                "data_path": "data/datasets/ovmm/ovmm_val.json.gz",
                "scenes_dir": "data/scenes/",
            },
        }
    })
    
    return config


class HumanoidLabDemo:
    """Full habitat-lab framework demo"""
    
    def __init__(self, scene_id: str = "102344280"):
        self.scene_id = scene_id
        self.env = None
        self.running = True
        self.camera_mode = 'third_person'
        
    def setup(self):
        """Initialize habitat-lab environment"""
        print("\n🤖 Setting up Habitat-Lab Humanoid Environment...")
        
        # Create config
        print("  📝 Creating configuration...")
        config = create_humanoid_config(self.scene_id)
        
        # Create environment
        print("  📦 Creating environment...")
        try:
            self.env = habitat.Env(config=config)
            print("  ✅ Environment created")
        except Exception as e:
            print(f"  ⚠️  Standard Env failed: {e}")
            print("  🔄 Falling back to simplified setup...")
            # Fall back to basic simulator if full env fails
            return self._setup_fallback()
        
        # Reset
        print("  🔄 Resetting environment...")
        try:
            obs = self.env.reset()
            print("  ✅ Environment ready!")
            print(f"  📊 Observations: {list(obs.keys())}")
            return True
        except Exception as e:
            print(f"  ⚠️  Reset failed: {e}")
            return self._setup_fallback()
    
    def _setup_fallback(self):
        """Fallback to basic simulator setup"""
        print("  🔧 Using simplified simulator...")
        
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = "./data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json"
        sim_cfg.scene_id = self.scene_id
        sim_cfg.enable_physics = True
        sim_cfg.requires_textures = True
        sim_cfg.override_scene_light_defaults = True
        sim_cfg.enable_hbao = True
        
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [720, 1280]
        rgb_sensor.position = [0.0, 1.6, 0.0]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor]
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        
        # Load humanoid
        ao_mgr = self.sim.get_articulated_object_manager()
        self.humanoid = ao_mgr.add_articulated_object_from_urdf(
            filepath="./data/humanoids/male_1/male_1.urdf",
            fixed_base=False
        )
        
        if self.sim.pathfinder.is_loaded:
            spawn_point = self.sim.pathfinder.get_random_navigable_point()
        else:
            spawn_point = mn.Vector3(0, 0, 0)
            self.sim.recompute_navmesh(self.sim.pathfinder, habitat_sim.NavMeshSettings())
            if self.sim.pathfinder.is_loaded:
                spawn_point = self.sim.pathfinder.get_random_navigable_point()
        
        spawn_point.y += 0.9
        self.humanoid.translation = spawn_point
        self.humanoid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        
        # Load motion data
        self.motion_data = None
        self.motion_frame = 0.0
        motion_path = "./data/humanoids/walking_motion_processed_smplx.pkl"
        if os.path.exists(motion_path):
            with open(motion_path, 'rb') as f:
                data = pickle.load(f)
                if 'walk_motion' in data:
                    self.motion_data = data['walk_motion']['joints_array']
                    self.num_frames = len(self.motion_data)
        
        self.fallback_mode = True
        print("  ✅ Fallback simulator ready")
        return True
    
    def _update_fallback_animation(self, distance: float):
        """Update animation in fallback mode"""
        if self.motion_data is None or distance < 0.001:
            return
        
        self.motion_frame = (self.motion_frame + distance * 30.0) % self.num_frames
        
        frame_a = int(np.floor(self.motion_frame))
        frame_b = int(np.ceil(self.motion_frame)) % self.num_frames
        alpha = self.motion_frame - frame_a
        
        pose_a = self.motion_data[frame_a].flatten()
        pose_b = self.motion_data[frame_b].flatten()
        blended_pose = (1.0 - alpha) * pose_a + alpha * pose_b
        
        new_positions = list(self.humanoid.joint_positions)
        for i in range(min(len(blended_pose), len(new_positions))):
            new_positions[i] = float(blended_pose[i])
        
        # Left arm fix
        swing = np.sin(self.motion_frame * 0.2) * 0.5
        if 67 < len(new_positions):
            new_positions[64] = np.sin(swing / 2)
            new_positions[65] = 0.0
            new_positions[66] = 0.0
            new_positions[67] = np.cos(swing / 2)
        
        self.humanoid.joint_positions = new_positions
    
    def run(self):
        """Main interaction loop"""
        if not self.setup():
            print("❌ Setup failed!")
            return
        
        print("\n" + "="*60)
        print("🎮 Habitat-Lab Humanoid Exploration")
        print("   W/S - Move forward/backward")
        print("   A/D - Turn left/right")
        print("   C - Toggle camera")
        print("   ESC - Exit")
        print("="*60 + "\n")
        
        using_env = hasattr(self, 'env') and self.env is not None
        prev_pos = None
        
        while self.running:
            # Get observations
            if using_env:
                try:
                    obs = self.env._sim.get_sensor_observations()
                    frame = obs.get("head_rgb_sensor", obs.get("third_rgb_sensor", obs.get(list(obs.keys())[0])))
                except:
                    using_env = False
                    continue
            else:
                # Fallback mode - update camera manually
                agent = self.sim.initialize_agent(0)
                agent_state = agent.get_state()
                h_pos = self.humanoid.translation
                
                camera_offset = mn.Vector3(0, 1.5, 2.5)
                agent_state.position = np.array([h_pos.x, h_pos.y + camera_offset.y, h_pos.z + camera_offset.z])
                
                look_at = h_pos + mn.Vector3(0, 0.5, 0)
                forward = (look_at - mn.Vector3(*agent_state.position)).normalized()
                right = mn.math.cross(forward, mn.Vector3(0, 1, 0)).normalized()
                up = mn.math.cross(right, forward).normalized()
                rot_mat = mn.Matrix3x3(right, up, -forward)
                rot_quat = mn.Quaternion.from_matrix(rot_mat)
                agent_state.rotation = np.quaternion(rot_quat.scalar, *rot_quat.vector)
                agent.set_state(agent_state)
                
                obs = self.sim.get_sensor_observations()
                frame = obs["color_sensor"]
            
            # Convert frame
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # UI
            mode_text = "Habitat-Lab Task Mode" if using_env else "Fallback Sim Mode"
            cv2.putText(frame, f"Humanoid Exploration ({mode_text})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow("Humanoid Demo", frame)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.running = False
            elif key in [ord('w'), ord('s'), ord('a'), ord('d')]:
                if using_env:
                    # Use habitat-lab actions
                    lin_vel = 0.1 if key == ord('w') else (-0.1 if key == ord('s') else 0.0)
                    ang_vel = 0.1 if key == ord('a') else (-0.1 if key == ord('d') else 0.0)
                    
                    try:
                        action = {"action": "humanoid_move", "action_args": {"lin_vel": lin_vel, "ang_vel": ang_vel}}
                        self.env.step(action)
                    except Exception as e:
                        print(f"Action failed: {e}")
                else:
                    # Fallback controls
                    if prev_pos is None:
                        prev_pos = self.humanoid.translation
                    
                    if key == ord('w'):
                        forward = self.humanoid.rotation.transform_vector(mn.Vector3(0, 0, 1))
                        new_pos = self.humanoid.translation + forward * 0.1
                    elif key == ord('s'):
                        backward = self.humanoid.rotation.transform_vector(mn.Vector3(0, 0, -1))
                        new_pos = self.humanoid.translation + backward * 0.1
                    elif key == ord('a'):
                        self.humanoid.rotation = self.humanoid.rotation * mn.Quaternion.rotation(mn.Rad(0.1), mn.Vector3(0, 1, 0))
                        new_pos = self.humanoid.translation
                    else:  # 'd'
                        self.humanoid.rotation = self.humanoid.rotation * mn.Quaternion.rotation(mn.Rad(-0.1), mn.Vector3(0, 1, 0))
                        new_pos = self.humanoid.translation
                    
                    if self.sim.pathfinder.is_loaded and key in [ord('w'), ord('s')]:
                        final_pos = self.sim.pathfinder.try_step(self.humanoid.translation, new_pos)
                        final_pos.y += 0.9
                    else:
                        final_pos = new_pos
                    
                    distance = (final_pos - prev_pos).length()
                    self.humanoid.translation = final_pos
                    prev_pos = final_pos
                    
                    self._update_fallback_animation(distance)
                    self.sim.step_physics(1.0/60.0)
        
        cv2.destroyAllWindows()
        if using_env:
            self.env.close()
        else:
            self.sim.close()
        print("\n✅ Demo completed!")


def main():
    print("🚀 Habitat-Lab Framework Humanoid Demo")
    print("Full task-based architecture for policy learning\n")
    
    try:
        demo = HumanoidLabDemo(scene_id="102344280")
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
