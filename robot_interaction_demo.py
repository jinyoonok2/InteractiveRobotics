#!/usr/bin/env python3
"""
Robot Interaction Demo - Control Stretch Robot and Manipulate Objects
Focus: Direct robot control, object manipulation, and environment interaction

Controls:
  W/S  - Move forward/backward
  A/D  - Turn left/right
  Q/E  - Strafe left/right
  C    - Toggle camera (1st/3rd person)
  Z/X  - Rotate camera (3rd person only)
  J/K  - Lift arm down/up
  U/I  - Retract/extend arm
  G    - Open/close gripper
  P    - Pick/drop nearest object
  O    - Spawn object
  R    - Reset robot joints
  ESC  - Exit
"""

import habitat_sim
import numpy as np
import cv2
import magnum as mn

class RobotInteractionDemo:
    # Available scene options
    SCENES = {
        '1': {'id': '102344280', 'name': 'Modern Apartment'},
        '2': {'id': '102344250', 'name': 'Luxury House'},
        '3': {'id': '102344049', 'name': 'Office Space'},
        '4': {'id': '102344403', 'name': 'Large Villa'},
        '5': {'id': '102816036', 'name': 'Contemporary Home'},
        '6': {'id': '102815859', 'name': 'Traditional House'},
        '7': {'id': '102344115', 'name': 'Compact Studio'},
        '8': {'id': '102344469', 'name': 'Spacious Loft'},
    }
    
    def __init__(self, scene_id='102344280'):
        self.sim = None
        self.agent = None
        self.robot = None
        self.objects = []
        self.gripper_open = True
        self.held_object = None
        self.held_object_data = None
        self.gripper_link_id = None
        self.head_link_id = None
        self.camera_mode = 'third_person'
        self.camera_yaw = 0.0
        self.camera_distance = 2.5
        self.scene_id = scene_id
        
    def setup_simulator(self):
        """Setup Habitat-Sim with physics enabled"""
        print("🤖 Setting up Robot Interaction Demo...")
        
        import os
        base_path = os.path.dirname(__file__)
        
        sim_settings = {
            "scene_dataset": os.path.join(base_path, "data/scenes/hssd-hab/hssd-hab.scene_dataset_config.json"),
            "scene": self.scene_id,
            "width": 1280,
            "height": 720,
            "hfov": 90,
        }
        
        # Configure simulator with physics
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = sim_settings["scene_dataset"]
        sim_cfg.scene_id = sim_settings["scene"]
        sim_cfg.enable_physics = True
        sim_cfg.requires_textures = True
        sim_cfg.override_scene_light_defaults = True
        sim_cfg.enable_hbao = True
        
        # Setup sensors
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
        rgb_sensor.position = [0.0, 0.0, 0.0]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor]
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        
        # Initialize NavMesh for collision detection
        if not self.sim.pathfinder.is_loaded:
            print("  🗺️  Computing NavMesh...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = 1.5
            navmesh_settings.agent_radius = 0.3
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            
            if self.sim.pathfinder.is_loaded:
                print("  ✅ NavMesh computed successfully!")
            else:
                print("  ⚠️  NavMesh computation failed")
        
        # Initialize agent
        self.agent = self.sim.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.0, 0.0])
        self.agent.set_state(agent_state)
        
        # Load robot
        self.load_robot()
        
        print("✅ Demo ready!")
        print("🤖 Control the Stretch robot with WASD")
        print("📦 Press 'O' to spawn objects, 'P' to pick/drop")
        return sim_settings
    
    def load_robot(self):
        """Load Stretch robot"""
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        urdf_path = os.path.join(data_path, "robots/hab_stretch/urdf/hab_stretch.urdf")
        
        ao_mgr = self.sim.get_articulated_object_manager()
        self.robot = ao_mgr.add_articulated_object_from_urdf(
            filepath=urdf_path,
            fixed_base=True
        )
        
        if self.robot:
            pf = self.sim.pathfinder
            
            if pf.is_loaded:
                spawn_point = pf.get_random_navigable_point()
                print(f"  📍 Robot spawned at: {spawn_point}")
            else:
                spawn_point = mn.Vector3(0.0, 0.0, 0.0)
            
            self.robot.translation = spawn_point
            self.robot.rotation = mn.Quaternion.rotation(mn.Rad(0), mn.Vector3(0, 1, 0))
            self.robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
            # Find gripper and head links
            self.gripper_link_id = None
            self.head_link_id = None
            for link_id in range(self.robot.num_links):
                link_name = self.robot.get_link_name(link_id)
                if "gripper_finger_left" in link_name.lower():
                    self.gripper_link_id = link_id
                if "head" in link_name.lower() and "tilt" in link_name.lower():
                    self.head_link_id = link_id
            
            print(f"  ✅ Stretch robot loaded with {self.robot.num_links} links")
            if self.gripper_link_id:
                print(f"  🤏 Gripper link ID: {self.gripper_link_id}")
            if self.head_link_id:
                print(f"  📷 Head link ID: {self.head_link_id}")
    
    def update_camera(self):
        """Update camera position based on current mode"""
        if not self.robot:
            return
        
        agent_state = self.agent.get_state()
        
        if self.camera_mode == 'first_person':
            # First-person: camera from robot head
            if self.head_link_id is not None:
                head_node = self.robot.get_link_scene_node(self.head_link_id)
                head_pos = head_node.absolute_translation
                head_rot_matrix = head_node.absolute_transformation().rotation()
                head_quat = mn.Quaternion.from_matrix(head_rot_matrix)
                
                camera_offset = head_quat.transform_vector(mn.Vector3(0, 0.05, -0.35))
                camera_pos = head_pos + camera_offset
                
                agent_state.position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
                agent_state.rotation = np.quaternion(head_quat.scalar, *head_quat.vector)
            else:
                robot_pos = self.robot.translation
                agent_state.position = np.array([robot_pos.x, robot_pos.y + 1.2, robot_pos.z])
        
        else:  # third_person
            robot_pos = self.robot.translation
            
            camera_height = 1.5
            camera_offset_x = self.camera_distance * np.sin(self.camera_yaw)
            camera_offset_z = self.camera_distance * np.cos(self.camera_yaw)
            
            camera_pos = robot_pos + mn.Vector3(camera_offset_x, camera_height, camera_offset_z)
            look_at = robot_pos + mn.Vector3(0, 0.5, 0)
            
            forward = (look_at - camera_pos).normalized()
            right = mn.math.cross(forward, mn.Vector3(0, 1, 0)).normalized()
            up = mn.math.cross(right, forward).normalized()
            
            rotation_matrix = mn.Matrix3x3(right, up, -forward)
            rotation_quat = mn.Quaternion.from_matrix(rotation_matrix)
            
            agent_state.position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
            agent_state.rotation = np.quaternion(rotation_quat.scalar, *rotation_quat.vector)
        
        self.agent.set_state(agent_state)
    
    def toggle_camera_mode(self):
        """Switch between first-person and third-person camera"""
        if self.camera_mode == 'first_person':
            self.camera_mode = 'third_person'
            # Position camera behind robot
            robot_rot = self.robot.rotation
            forward = robot_rot.transform_vector(mn.Vector3(0, 0, -1))
            self.camera_yaw = np.arctan2(forward.x, forward.z) + np.pi
            print("📹 Camera: Third-person view (Z/X to rotate)")
        else:
            self.camera_mode = 'first_person'
            print("📹 Camera: First-person view")
    
    def spawn_object(self):
        """Spawn an object in front of robot"""
        if not self.robot:
            return
        
        robot_pos = self.robot.translation
        robot_rot = self.robot.rotation
        
        import random
        forward = robot_rot.transform_vector(mn.Vector3(0, 0, -1.0))
        lateral_offset = random.uniform(-0.2, 0.2)
        lateral = robot_rot.transform_vector(mn.Vector3(lateral_offset, 0, 0))
        spawn_pos = robot_pos + forward + lateral
        position = [spawn_pos.x, 0.5, spawn_pos.z]
        
        # Try spawning URDF block
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        block_paths = [
            "objects/red_block.urdf",
            "objects/blue_block.urdf",
            "objects/yellow_block.urdf",
            "objects/green_block.urdf"
        ]
        
        ao_mgr = self.sim.get_articulated_object_manager()
        block_path = block_paths[len(self.objects) % len(block_paths)]
        full_path = os.path.join(data_path, block_path)
        
        if os.path.exists(full_path):
            try:
                obj = ao_mgr.add_articulated_object_from_urdf(
                    filepath=full_path,
                    fixed_base=False
                )
                if obj:
                    obj.translation = mn.Vector3(position[0], position[1], position[2])
                    obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
                    obj.mass = 0.5
                    
                    self.objects.append({
                        'object': obj,
                        'type': 'block',
                        'position': position,
                        'id': len(self.objects)
                    })
                    
                    print(f"📦 Spawned {block_path.split('/')[-1]} at {position}")
                    return
            except Exception as e:
                print(f"⚠️ Failed to spawn block: {e}")
        
        print("⚠️ No objects available to spawn")
    
    def control_robot_joint(self, joint_id, delta):
        """Move a robot joint"""
        if self.robot is None:
            return
        
        current_pos = self.robot.joint_positions
        if joint_id < len(current_pos):
            new_pos = list(current_pos)
            old_val = new_pos[joint_id]
            new_pos[joint_id] += delta
            
            # Apply limits
            if joint_id <= 2:
                new_pos[joint_id] = max(0.0, min(0.5, new_pos[joint_id]))
            elif joint_id == 3:
                new_pos[joint_id] = max(0.0, min(0.3, new_pos[joint_id]))
            elif joint_id == 4:
                new_pos[joint_id] = max(0.0, min(0.15, new_pos[joint_id]))
            else:
                new_pos[joint_id] = max(-3.14, min(3.14, new_pos[joint_id]))
            
            self.robot.joint_positions = new_pos
            
            actual_val = self.robot.joint_positions[joint_id]
            print(f"🦾 Joint {joint_id}: {old_val:.3f} → {actual_val:.3f}")
    
    def toggle_gripper(self):
        """Open/close gripper"""
        if self.robot is None:
            return
        
        self.gripper_open = not self.gripper_open
        current_pos = list(self.robot.joint_positions)
        
        if len(current_pos) >= 2:
            current_pos[-2] = 0.04 if self.gripper_open else 0.0
            current_pos[-1] = 0.04 if self.gripper_open else 0.0
        
        self.robot.joint_positions = current_pos
        print(f"🤏 Gripper {'opened' if self.gripper_open else 'closed'}")
    
    def pick_nearest_object(self):
        """Pick or drop nearest object"""
        if self.robot is None:
            return
        
        # If holding, drop it
        if self.held_object is not None:
            self.drop_object()
            return
        
        if not self.objects:
            print("⚠️ No objects to pick up! Press 'O' to spawn objects.")
            return
        
        # Get gripper position
        if self.gripper_link_id is not None:
            gripper_pos = self.robot.get_link_scene_node(self.gripper_link_id).absolute_translation
        else:
            robot_pos = self.robot.translation
            gripper_reach = mn.Vector3(0.6, 0.3, 0.0)
            gripper_pos = robot_pos + gripper_reach
        
        # Find nearest object
        nearest = None
        min_dist = float('inf')
        
        for obj_data in self.objects:
            obj = obj_data['object']
            obj_pos = obj.translation
            dist = (gripper_pos - obj_pos).length()
            
            if dist < min_dist:
                min_dist = dist
                nearest = obj_data
        
        if nearest and min_dist < 0.8:
            obj = nearest['object']
            
            # Extend arm and close gripper
            current_pos = list(self.robot.joint_positions)
            if len(current_pos) > 0:
                for i in range(min(3, len(current_pos))):
                    current_pos[i] = 0.3
                
                if len(current_pos) >= 2:
                    current_pos[-2] = 0.0
                    current_pos[-1] = 0.0
                
                self.robot.joint_positions = current_pos
                self.gripper_open = False
            
            # Attach to gripper
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
            if self.gripper_link_id is not None:
                actual_gripper_pos = self.robot.get_link_scene_node(self.gripper_link_id).absolute_translation
                obj.translation = actual_gripper_pos + mn.Vector3(0.1, 0, 0)
            else:
                obj.translation = gripper_pos
            
            self.held_object = obj
            self.held_object_data = nearest
            
            print(f"✅ GRASPED object #{nearest['id']+1} (dist: {min_dist:.2f}m)")
        else:
            print(f"❌ Object too far: {min_dist:.2f}m (need < 0.8m)")
    
    def drop_object(self):
        """Release held object"""
        if self.held_object is None:
            return
        
        # Open gripper
        self.gripper_open = True
        current_pos = list(self.robot.joint_positions)
        if len(current_pos) >= 2:
            current_pos[-2] = 0.04
            current_pos[-1] = 0.04
            self.robot.joint_positions = current_pos
        
        # Position in front of robot
        robot_pos = self.robot.translation
        robot_rot = self.robot.rotation
        forward = robot_rot.transform_vector(mn.Vector3(0, 0, -0.5))
        drop_pos = robot_pos + forward
        drop_pos.y = 0.3
        self.held_object.translation = drop_pos
        
        # Make dynamic
        self.held_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        forward_vel = robot_rot.transform_vector(mn.Vector3(0, -0.3, -0.3))
        self.held_object.linear_velocity = forward_vel
        
        print(f"📤 DROPPED object #{self.held_object_data['id']+1}")
        
        self.held_object = None
        self.held_object_data = None
    
    def update_held_object(self):
        """Keep held object attached to gripper"""
        if self.held_object is not None and self.robot is not None:
            if self.gripper_link_id is not None:
                gripper_node = self.robot.get_link_scene_node(self.gripper_link_id)
                gripper_pos = gripper_node.absolute_translation
                robot_rot = self.robot.rotation
                offset = robot_rot.transform_vector(mn.Vector3(0, 0, -0.12))
                offset += mn.Vector3(0, 0.05, 0)
                
                self.held_object.translation = gripper_pos + offset
                self.held_object.rotation = robot_rot
    
    def render_ui(self, frame):
        """Draw UI overlay"""
        camera_mode_text = "1st Person" if self.camera_mode == 'first_person' else "3rd Person"
        
        instructions = [
            "🎮 ROBOT INTERACTION DEMO",
            "WS-Move | AD-Turn | QE-Strafe | C-Camera",
            "JK-Lift | UI-Arm | G-Gripper | O-Spawn | P-Pick",
            "",
            f"📹 Camera: {camera_mode_text}",
            f"🤖 Objects: {len(self.objects)}",
            f"🤏 Gripper: {'Open' if self.gripper_open else 'Closed'}",
            f"📦 Holding: {'Object #' + str(self.held_object_data['id']+1) if self.held_object else 'Nothing'}"
        ]
        
        y = 30
        for text in instructions:
            cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            y += 25
        
        return frame
    
    def run(self):
        """Main demo loop"""
        self.setup_simulator()
        
        print("\n" + "="*60)
        print("🎮 Robot Interaction Controls:")
        print("   W/S - Move forward/backward")
        print("   A/D - Turn left/right")
        print("   Q/E - Strafe left/right")
        print("   C - Toggle camera (1st/3rd person)")
        print("   Z/X - Rotate camera (3rd person)")
        print("   J/K - Lift arm DOWN/UP")
        print("   U/I - Retract/Extend arm")
        print("   1-5 - Fine joint control (Shift+1-5 reverse)")
        print("   G - Open/close gripper")
        print("   O - Spawn object")
        print("   P - Pick/drop object")
        print("   R - Reset robot joints")
        print("="*60 + "\n")
        
        running = True
        while running:
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC
                running = False
            elif key == ord('o') or key == ord('O'):
                self.spawn_object()
            elif key == ord('g') or key == ord('G'):
                self.toggle_gripper()
            elif key == ord('p') or key == ord('P'):
                self.pick_nearest_object()
            elif key == ord('r') or key == ord('R'):
                if self.robot:
                    self.robot.joint_positions = [0.0] * len(self.robot.joint_positions)
                    print("🔄 Robot reset")
            # Fine joint control with number keys
            elif key == ord('1'):
                self.control_robot_joint(0, 0.05)
            elif key == ord('!'):  # Shift+1
                self.control_robot_joint(0, -0.05)
            elif key == ord('2'):
                self.control_robot_joint(1, 0.05)
            elif key == ord('@'):  # Shift+2
                self.control_robot_joint(1, -0.05)
            elif key == ord('3'):
                self.control_robot_joint(2, 0.05)
            elif key == ord('#'):  # Shift+3
                self.control_robot_joint(2, -0.05)
            elif key == ord('4'):
                self.control_robot_joint(3, 0.05)
            elif key == ord('$'):  # Shift+4
                self.control_robot_joint(3, -0.05)
            elif key == ord('5'):
                self.control_robot_joint(4, 0.05)
            elif key == ord('%'):  # Shift+5
                self.control_robot_joint(4, -0.05)
            # Alternative controls: J/K for lift, U/I for arm extension
            elif key == ord('j') or key == ord('J'):
                self.control_robot_joint(0, -0.05)
            elif key == ord('k') or key == ord('K'):
                self.control_robot_joint(0, 0.05)
            elif key == ord('u') or key == ord('U'):
                self.control_robot_joint(1, -0.05)
            elif key == ord('i') or key == ord('I'):
                self.control_robot_joint(1, 0.05)
            elif key == ord('c') or key == ord('C'):
                self.toggle_camera_mode()
            elif key == ord('z') or key == ord('Z'):
                if self.camera_mode == 'third_person':
                    self.camera_yaw -= 0.1
            elif key == ord('x') or key == ord('X'):
                if self.camera_mode == 'third_person':
                    self.camera_yaw += 0.1
            # Movement controls
            elif key == ord('w'):
                if self.robot:
                    forward = self.robot.rotation.transform_vector(mn.Vector3(0, 0, 1))  # Fixed: +Z is forward
                    new_pos = self.robot.translation + forward * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(self.robot.translation, new_pos)
                        self.robot.translation = final_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('s'):
                if self.robot:
                    backward = self.robot.rotation.transform_vector(mn.Vector3(0, 0, -1))  # Fixed: -Z is backward
                    new_pos = self.robot.translation + backward * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(self.robot.translation, new_pos)
                        self.robot.translation = final_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('a'):
                if self.robot:
                    self.robot.rotation = self.robot.rotation * mn.Quaternion.rotation(
                        mn.Rad(0.1), mn.Vector3(0, 1, 0))
            elif key == ord('d'):
                if self.robot:
                    self.robot.rotation = self.robot.rotation * mn.Quaternion.rotation(
                        mn.Rad(-0.1), mn.Vector3(0, 1, 0))
            elif key == ord('q'):
                if self.robot:
                    left = self.robot.rotation.transform_vector(mn.Vector3(-1, 0, 0))
                    new_pos = self.robot.translation + left * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(self.robot.translation, new_pos)
                        self.robot.translation = final_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('e'):
                if self.robot:
                    right = self.robot.rotation.transform_vector(mn.Vector3(1, 0, 0))
                    new_pos = self.robot.translation + right * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(self.robot.translation, new_pos)
                        self.robot.translation = final_pos
                    else:
                        self.robot.translation = new_pos
            
            # Physics update
            self.sim.step_physics(1.0/60.0)
            
            # Update camera and held object
            self.update_camera()
            self.update_held_object()
            
            # Render
            obs = self.sim.get_sensor_observations()
            frame = obs["color_sensor"]
            
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            frame = self.render_ui(frame)
            cv2.imshow("Robot Interaction Demo", frame)
        
        cv2.destroyAllWindows()
        print("\n✅ Demo completed!")

def main():
    print("🚀 Starting Robot Interaction Demo...")
    print("🤖 Control the Stretch robot and manipulate objects!")
    
    # Select scene
    print("\n" + "="*60)
    print("🏠 SELECT ENVIRONMENT:")
    print("="*60)
    for key, scene in RobotInteractionDemo.SCENES.items():
        print(f"   {key}. {scene['name']} (ID: {scene['id']})")
    print("   0. Custom scene ID")
    print("="*60)
    
    choice = input("\nEnter your choice (1-8, or 0 for custom): ").strip()
    
    if choice == '0':
        scene_id = input("Enter scene ID: ").strip()
    elif choice in RobotInteractionDemo.SCENES:
        scene_id = RobotInteractionDemo.SCENES[choice]['id']
        print(f"\n✅ Selected: {RobotInteractionDemo.SCENES[choice]['name']}")
    else:
        print("\n⚠️ Invalid choice, using default (Modern Apartment)")
        scene_id = '102344280'
    
    try:
        demo = RobotInteractionDemo(scene_id=scene_id)
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
