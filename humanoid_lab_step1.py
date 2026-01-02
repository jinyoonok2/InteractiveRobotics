#!/usr/bin/env python3
"""
Humanoid Demo with Habitat-Lab Framework
Proper integration for future policy learning and complex interactions
"""

import os
import numpy as np
import cv2
import magnum as mn
from typing import Any, Dict

import habitat
import habitat_sim
from habitat.config.default_structured_configs import HabitatConfigPlugin
from habitat.core.registry import registry
from omegaconf import DictConfig


print("🔍 Checking habitat-lab installation...")
print(f"  Habitat version: {habitat.__version__}")
print(f"  Habitat-Sim version: {habitat_sim.__version__}")

# First, let's create a minimal working example
# We'll build up to KinematicHumanoid step by step

def create_minimal_sim_config(scene_id="102344280"):
    """Create a minimal habitat-sim configuration"""
    
    base_path = os.path.dirname(__file__)
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_dataset_config_file = os.path.join(
        base_path, "data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json"
    )
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = True
    sim_cfg.requires_textures = True
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.enable_hbao = True
    
    # RGB sensor
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [720, 1280]
    rgb_sensor.position = [0.0, 1.6, 0.0]
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_sensor]
    
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


class MinimalHumanoidDemo:
    """Step 1: Get basic humanoid working with habitat-sim"""
    
    def __init__(self, scene_id="102344280"):
        self.scene_id = scene_id
        self.sim = None
        self.humanoid = None
        self.running = True
        
    def setup(self):
        """Initialize simulator and load humanoid"""
        print("\n🤖 Setting up Minimal Humanoid Demo...")
        
        # Create simulator
        print("  📦 Creating simulator...")
        cfg = create_minimal_sim_config(self.scene_id)
        self.sim = habitat_sim.Simulator(cfg)
        print("  ✅ Simulator created")
        
        # Initialize NavMesh
        if not self.sim.pathfinder.is_loaded:
            print("  🗺️  Computing NavMesh...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
        
        # Load humanoid URDF
        print("  👤 Loading humanoid...")
        base_path = os.path.dirname(__file__)
        urdf_path = os.path.join(base_path, "data/humanoids/male_1/male_1.urdf")
        
        ao_mgr = self.sim.get_articulated_object_manager()
        self.humanoid = ao_mgr.add_articulated_object_from_urdf(
            filepath=urdf_path,
            fixed_base=False
        )
        
        if self.humanoid:
            # Position humanoid
            if self.sim.pathfinder.is_loaded:
                spawn_point = self.sim.pathfinder.get_random_navigable_point()
            else:
                spawn_point = mn.Vector3(0, 0, 0)
            
            spawn_point.y += 0.9  # Lift to body center
            self.humanoid.translation = spawn_point
            self.humanoid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
            print(f"  ✅ Humanoid loaded at {spawn_point}")
            print(f"     Links: {self.humanoid.num_links}")
            print(f"     DOF: {len(self.humanoid.joint_positions)}")
        
        return True
    
    def run(self):
        """Main loop"""
        if not self.setup():
            return
        
        print("\n" + "="*60)
        print("🎮 Controls:")
        print("   W/S/A/D - Move humanoid")
        print("   ESC - Exit")
        print("="*60 + "\n")
        
        agent = self.sim.initialize_agent(0)
        
        while self.running:
            # Update camera to follow humanoid
            if self.humanoid:
                agent_state = agent.get_state()
                humanoid_pos = self.humanoid.translation
                
                # Third-person camera
                camera_offset = mn.Vector3(0, 1.5, 2.5)
                agent_state.position = np.array([
                    humanoid_pos.x + camera_offset.x,
                    humanoid_pos.y + camera_offset.y,
                    humanoid_pos.z + camera_offset.z
                ])
                
                # Look at humanoid
                look_at = humanoid_pos + mn.Vector3(0, 0.5, 0)
                forward = (look_at - mn.Vector3(*agent_state.position)).normalized()
                right = mn.math.cross(forward, mn.Vector3(0, 1, 0)).normalized()
                up = mn.math.cross(right, forward).normalized()
                
                rotation_matrix = mn.Matrix3x3(right, up, -forward)
                rotation_quat = mn.Quaternion.from_matrix(rotation_matrix)
                agent_state.rotation = np.quaternion(rotation_quat.scalar, *rotation_quat.vector)
                
                agent.set_state(agent_state)
            
            # Get observation
            obs = self.sim.get_sensor_observations()
            frame = obs["color_sensor"]
            
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # UI
            cv2.putText(frame, "Habitat-Lab Humanoid (Step 1: Basic Loading)", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(frame, f"Position: {self.humanoid.translation if self.humanoid else 'N/A'}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            cv2.imshow("Humanoid Demo", frame)
            
            # Input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                self.running = False
            elif key == ord('w') and self.humanoid:
                # Move forward
                forward = self.humanoid.rotation.transform_vector(mn.Vector3(0, 0, 1))
                new_pos = self.humanoid.translation + forward * 0.1
                
                if self.sim.pathfinder.is_loaded:
                    final_pos = self.sim.pathfinder.try_step(self.humanoid.translation, new_pos)
                    final_pos.y += 0.9
                    self.humanoid.translation = final_pos
                else:
                    self.humanoid.translation = new_pos
            elif key == ord('s') and self.humanoid:
                # Move backward
                backward = self.humanoid.rotation.transform_vector(mn.Vector3(0, 0, -1))
                new_pos = self.humanoid.translation + backward * 0.1
                
                if self.sim.pathfinder.is_loaded:
                    final_pos = self.sim.pathfinder.try_step(self.humanoid.translation, new_pos)
                    final_pos.y += 0.9
                    self.humanoid.translation = final_pos
                else:
                    self.humanoid.translation = new_pos
            elif key == ord('a') and self.humanoid:
                # Turn left
                self.humanoid.rotation = self.humanoid.rotation * mn.Quaternion.rotation(
                    mn.Rad(0.1), mn.Vector3(0, 1, 0))
            elif key == ord('d') and self.humanoid:
                # Turn right
                self.humanoid.rotation = self.humanoid.rotation * mn.Quaternion.rotation(
                    mn.Rad(-0.1), mn.Vector3(0, 1, 0))
            
            # Step physics
            self.sim.step_physics(1.0/60.0)
        
        cv2.destroyAllWindows()
        self.sim.close()
        print("\n✅ Demo completed!")


def main():
    print("🚀 Habitat-Lab Humanoid Demo - Progressive Development")
    print("This is Step 1: Basic humanoid loading and movement")
    print("Next steps will add KinematicHumanoid and proper habitat-lab integration\n")
    
    try:
        demo = MinimalHumanoidDemo(scene_id="102344280")
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
