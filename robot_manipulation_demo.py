#!/usr/bin/env python3
"""
Interactive Robot Manipulation Demo
Spawn objects and control the Stretch robot to pick them up!

Controls:
  W/S  - Move forward/backward
  A/D  - Turn left/right
  Q/E  - Strafe left/right
  Z/X  - Look up/down
  1-5  - Control robot arm joints
  G    - Open/close gripper
  P    - Pick up nearest object
  R    - Reset robot
  O    - Spawn random object
  ESC  - Exit
"""

import habitat_sim
import numpy as np
import cv2
import magnum as mn
import time

class RobotManipulationDemo:
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
        self.held_object = None  # Track grasped object
        self.held_object_data = None
        self.held_object_idx = None
        self.gripper_link_id = None
        self.head_link_id = None
        self.camera_pitch = 0.3  # Camera pitch angle (look down slightly)
        self.camera_distance = 2.5  # Distance behind robot
        self.scene_id = scene_id  # Selected scene
        self.held_object = None  # Track grasped object
        self.gripper_link_id = None  # Gripper attachment point
        
    def setup_simulator(self):
        """Setup Habitat-Sim with physics enabled"""
        print("🤖 Setting up Interactive Robot Manipulation Demo...")
        
        import os
        base_path = os.path.join(os.path.dirname(__file__), "home-robot")
        
        sim_settings = {
            "scene_dataset": os.path.join(base_path, "data/hssd-hab/hssd-hab.scene_dataset_config.json"),
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
        sim_cfg.enable_physics = True  # Required for object interaction
        sim_cfg.requires_textures = True
        
        # --- ENABLE HSSD HIGH-FIDELITY RENDERING ---
        # These flags improve material rendering and visual quality
        sim_cfg.override_scene_light_defaults = True  # Use improved default lighting
        sim_cfg.enable_hbao = True  # Ambient occlusion to prevent floating look
        
        # Setup sensors
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
        rgb_sensor.position = [0.0, 1.5, 0.0]
        
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.agent.ActionSpec(
                "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
            ),
            "turn_left": habitat_sim.agent.ActionSpec(
                "turn_left", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
            "turn_right": habitat_sim.agent.ActionSpec(
                "turn_right", habitat_sim.agent.ActuationSpec(amount=10.0)
            ),
        }
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        
        # Initialize and load NavMesh for collision detection
        if not self.sim.pathfinder.is_loaded:
            print("  🗺️  Computing NavMesh for collision detection...")
            navmesh_settings = habitat_sim.NavMeshSettings()
            navmesh_settings.set_defaults()
            navmesh_settings.agent_height = 1.5  # Stretch robot height
            navmesh_settings.agent_radius = 0.3  # Stretch robot radius
            self.sim.recompute_navmesh(self.sim.pathfinder, navmesh_settings)
            
            if self.sim.pathfinder.is_loaded:
                print("  ✅ NavMesh computed successfully!")
            else:
                print("  ⚠️  NavMesh computation failed - collision detection disabled")
        
        # Setup lighting
        self.setup_lighting()
        
        # Initialize agent
        self.agent = self.sim.initialize_agent(0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.0, 0.0])
        self.agent.set_state(agent_state)
        
        # Load robot
        self.load_robot()
        
        print("✅ Demo ready! Press 'O' to spawn objects!")
        return sim_settings
    
    def setup_lighting(self):
        """HSSD-optimized lighting with PBR support"""
        # HSSD scenes don't have predefined lighting, but we can still benefit from
        # PBR shading by using the configuration flags set in setup_simulator.
        # This setup provides better material rendering than the previous custom setup.
        
        # Note: With enable_hbao=True and override_scene_light_defaults=True,
        # the simulator will use improved default lighting that respects PBR materials.
        # We can optionally add subtle enhancement lights if needed.
        pass  # Rely on the simulator configuration flags for now
    
    def load_robot(self):
        """Load Stretch robot"""
        print("🤖 Loading Stretch robot...")
        import os
        base_path = os.path.join(os.path.dirname(__file__), "home-robot")
        urdf_path = os.path.join(base_path, "data/robots/hab_stretch/urdf/hab_stretch.urdf")
        
        ao_mgr = self.sim.get_articulated_object_manager()
        self.robot = ao_mgr.add_articulated_object_from_urdf(
            filepath=urdf_path,
            fixed_base=True,  # Keep base fixed for manipulation
            mass_scale=1.0
        )
        
        if self.robot:
            # Position robot at a valid navigable point instead of hardcoded coordinates
            pf = self.sim.pathfinder
            
            # Check if pathfinder is loaded
            if pf.is_loaded:
                spawn_point = pf.get_random_navigable_point()
                print(f"  📍 Robot spawned at navigable point: {spawn_point}")
            else:
                # Fallback to safe default position if pathfinder not available
                spawn_point = mn.Vector3(0.0, 0.0, 0.0)
                print(f"  ⚠️  PathFinder not loaded, using default spawn: {spawn_point}")
            
            # Ensure the robot is at floor level (y-coordinate from navmesh)
            self.robot.translation = spawn_point
            self.robot.rotation = mn.Quaternion.rotation(mn.Rad(0), mn.Vector3(0, 1, 0))
            
            # Set to kinematic for controllable movement
            self.robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
            print(f"  📷 Third-person camera will follow robot")
            
            # Find gripper link for attachment
            self.gripper_link_id = None
            self.head_link_id = None
            for link_id in range(self.robot.num_links):
                link_name = self.robot.get_link_name(link_id)
                if "gripper_finger_left" in link_name.lower():
                    self.gripper_link_id = link_id
                if "head" in link_name.lower() and "tilt" in link_name.lower():
                    self.head_link_id = link_id
            
            # Print joint info
            print(f"  ✅ Robot loaded with {self.robot.num_links} links")
            print(f"  📊 Joint positions: {self.robot.joint_positions}")
            if self.gripper_link_id:
                print(f"  🤏 Gripper link ID: {self.gripper_link_id}")
            if self.head_link_id:
                print(f"  📷 Head camera link ID: {self.head_link_id}")
            print(f"  🎮 Controllable joints:")
            
            # List joint names and IDs
            for link_id in range(self.robot.num_links):
                link_name = self.robot.get_link_name(link_id)
                if "joint" in link_name.lower() or "gripper" in link_name.lower():
                    print(f"     {link_id}: {link_name}")
    
    def spawn_object(self, position=None):
        """Spawn a manipulatable object"""
        if position is None:
            # Spawn in front of robot
            if self.robot:
                robot_pos = self.robot.translation
                robot_rot = self.robot.rotation
                # Spawn 1m in front of robot
                forward = robot_rot.transform_vector(mn.Vector3(0, 0, -1.0))
                spawn_pos = robot_pos + forward
                position = [spawn_pos.x, 0.5, spawn_pos.z]
            else:
                position = [2.5, 0.5, 1.5]
        
        # Use primitive shapes (sphere, cube, etc)
        rigid_obj_mgr = self.sim.get_rigid_object_manager()
        
        # Create a sphere primitive
        obj = rigid_obj_mgr.add_object_by_template_id(0)  # Default primitive
        
        if obj is not None:
            obj.translation = mn.Vector3(position[0], position[1], position[2])
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj.mass = 0.5  # Light objects for easy manipulation
            
            # Give it a random color
            colors = [
                [1.0, 0.0, 0.0],  # Red
                [0.0, 1.0, 0.0],  # Green
                [0.0, 0.0, 1.0],  # Blue
                [1.0, 1.0, 0.0],  # Yellow
            ]
            color = colors[len(self.objects) % len(colors)]
            
            self.objects.append({
                'object': obj,
                'color': color,
                'position': position,
                'id': len(self.objects)
            })
            
            print(f"📦 Spawned object #{len(self.objects)} at {position}")
            return obj
        else:
            print("❌ Failed to spawn object - trying alternative method...")
            # Try spawning a simple cube
            return self.spawn_simple_cube(position)
    
    def spawn_simple_cube(self, position):
        """Spawn using URDF assets"""
        import os
        base_path = os.path.join(os.path.dirname(__file__), "home-robot")
        
        # Try to use one of the block assets
        block_paths = [
            "assets/red_block.urdf",
            "assets/blue_block.urdf",
            "assets/yellow_block.urdf",
            "assets/green_block.urdf"
        ]
        
        ao_mgr = self.sim.get_articulated_object_manager()
        
        for block_path in block_paths:
            full_path = os.path.join(base_path, block_path)
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
                        return obj
                except Exception as e:
                    continue
        
        print("⚠️  No objects available to spawn")
        return None
    
    def control_robot_joint(self, joint_id, delta):
        """Move a robot joint"""
        if self.robot is None:
            return
        
        current_pos = self.robot.joint_positions
        if joint_id < len(current_pos):
            new_pos = list(current_pos)
            new_pos[joint_id] += delta
            self.robot.joint_positions = new_pos
            print(f"🦾 Joint {joint_id}: {new_pos[joint_id]:.3f}")
    
    def toggle_gripper(self):
        """Open/close gripper"""
        if self.robot is None:
            return
        
        self.gripper_open = not self.gripper_open
        
        # Find gripper joints (usually last few joints)
        current_pos = list(self.robot.joint_positions)
        gripper_joints = [-2, -1]  # Typically last two joints
        
        for joint_id in gripper_joints:
            if abs(joint_id) < len(current_pos):
                current_pos[joint_id] = 0.04 if self.gripper_open else 0.0
        
        self.robot.joint_positions = current_pos
        status = "opened" if self.gripper_open else "closed"
        print(f"🤏 Gripper {status}")
    
    def update_held_object(self):
        """Keep held object attached to gripper"""
        if self.held_object is not None and self.robot is not None:
            # Get actual gripper link position for accurate tracking
            if self.gripper_link_id is not None:
                gripper_node = self.robot.get_link_scene_node(self.gripper_link_id)
                gripper_pos = gripper_node.absolute_translation
                
                # Get robot rotation to apply offset in correct direction
                robot_rot = self.robot.rotation
                # Offset forward and slightly up from gripper
                offset = robot_rot.transform_vector(mn.Vector3(0, 0, -0.12))
                offset += mn.Vector3(0, 0.05, 0)  # Slight upward offset
                
                self.held_object.translation = gripper_pos + offset
                # Match object rotation to robot for natural look
                self.held_object.rotation = robot_rot
            else:
                # Fallback: estimate position from robot base
                robot_pos = self.robot.translation
                robot_rot = self.robot.rotation
                gripper_reach = robot_rot.transform_vector(mn.Vector3(0, 0, -0.4))
                self.held_object.translation = robot_pos + gripper_reach + mn.Vector3(0, 0.5, 0)
                self.held_object.rotation = robot_rot
    
    def drop_object(self):
        """Release held object in front of robot"""
        if self.held_object is None:
            return
        
        # Open gripper
        self.gripper_open = True
        current_pos = list(self.robot.joint_positions)
        if len(current_pos) >= 2:
            current_pos[-2] = 0.04  # Open left finger
            current_pos[-1] = 0.04  # Open right finger
            self.robot.joint_positions = current_pos
        
        # Position object in front of robot before dropping
        robot_pos = self.robot.translation
        robot_rot = self.robot.rotation
        # Drop 0.5m in front of robot at ground level
        forward = robot_rot.transform_vector(mn.Vector3(0, 0, -0.5))
        drop_pos = robot_pos + forward
        drop_pos.y = 0.3  # Slightly above ground
        self.held_object.translation = drop_pos
        
        # Make dynamic again so it falls
        self.held_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        
        # Give it a small forward velocity in robot's direction
        forward_vel = robot_rot.transform_vector(mn.Vector3(0, -0.3, -0.3))
        self.held_object.linear_velocity = forward_vel
        
        print(f"📤 DROPPED object #{self.held_object_data['id']+1} in front of robot")
        print(f"   🤏 Gripper opened")
        
        self.held_object = None
        self.held_object_data = None
    
    def pick_nearest_object(self):
        """Snap-grasp nearest object (like Habitat OVMM)"""
        if self.robot is None:
            print("⚠️  No robot loaded!")
            return
        
        # If already holding, drop it
        if self.held_object is not None:
            self.drop_object()
            return
        
        if not self.objects:
            print("⚠️  No objects to pick up! Press 'O' to spawn objects.")
            return
        
        # Get actual gripper position from robot link
        if self.gripper_link_id is not None:
            gripper_pos = self.robot.get_link_scene_node(self.gripper_link_id).absolute_translation
        else:
            # Fallback to estimated position
            robot_pos = self.robot.translation
            gripper_reach = mn.Vector3(0.6, 0.3, 0.0)
            gripper_pos = robot_pos + gripper_reach
        
        # Find nearest object
        nearest = None
        min_dist = float('inf')
        nearest_idx = -1
        
        for idx, obj_data in enumerate(self.objects):
            obj = obj_data['object']
            obj_pos = obj.translation
            dist = (gripper_pos - obj_pos).length()
            
            if dist < min_dist:
                min_dist = dist
                nearest = obj_data
                nearest_idx = idx
        
        if nearest and min_dist < 0.8:  # Within reach
            # SNAP TO GRIPPER (like Habitat OVMM)
            obj = nearest['object']
            
            # Extend arm and close gripper (visual feedback)
            current_pos = list(self.robot.joint_positions)
            if len(current_pos) > 0:
                # Extend arm lift joint (usually index 0 or 1)
                for i in range(min(3, len(current_pos))):
                    current_pos[i] = 0.3  # Extend arm
                
                # Close gripper fingers
                if len(current_pos) >= 2:
                    current_pos[-2] = 0.0  # Close left finger
                    current_pos[-1] = 0.0  # Close right finger
                
                self.robot.joint_positions = current_pos
                self.gripper_open = False
            
            # Make object kinematic and attach to gripper
            obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
            
            # Snap to gripper position with slight offset
            if self.gripper_link_id is not None:
                actual_gripper_pos = self.robot.get_link_scene_node(self.gripper_link_id).absolute_translation
                obj.translation = actual_gripper_pos + mn.Vector3(0.1, 0, 0)
            else:
                obj.translation = gripper_pos
            
            self.held_object = obj
            self.held_object_data = nearest
            self.held_object_idx = nearest_idx
            
            print(f"✅ GRASPED object #{nearest['id']+1} (snap-to-gripper)")
            print(f"   Distance: {min_dist:.2f}m")
            print(f"   🦾 Arm extended, gripper closed")
            print(f"   Press 'P' again to DROP")
        else:
            print(f"❌ Object too far: {min_dist:.2f}m (need < 0.8m)")
            print(f"   Move closer with WASD")
    
    def drop_object(self):
        """Release held object"""
        if self.held_object is None:
            return
        
        # Make dynamic again so it falls
        self.held_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        
        # Give it a small forward velocity
        self.held_object.linear_velocity = mn.Vector3(0.5, 0, 0)
        
        print(f"📤 DROPPED object #{self.held_object_data['id']+1}")
        
        self.held_object = None
        self.held_object_data = None
    
    def update_held_object(self):
        """Keep held object attached to gripper"""
        if self.held_object is not None and self.robot is not None:
            # Update object position to follow gripper
            robot_pos = self.robot.translation
            gripper_reach = mn.Vector3(0.6, 0.3, 0.0)
            self.held_object.translation = robot_pos + gripper_reach
    
    def render_ui(self, frame):
        """Draw UI overlay"""
        h, w = frame.shape[:2]
        
        # Instructions
        instructions = [
            "🎮 CONTROLS:",
            "WS - Move | AD - Turn | QE - Strafe | ZX - Look",
            "1-5 - Arm joints | G - Gripper | O - Spawn | P - Pick/Drop",
            "",
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
        print("🎮 Interactive Controls:")
        print("   W/S - Move forward/backward")
        print("   A/D - Turn left/right")
        print("   Q/E - Strafe left/right")
        print("   Z/X - Look up/down")
        print("   O - Spawn object near robot")
        print("   1-5 - Control arm joints")
        print("   G - Open/close gripper")
        print("   P - Pick/drop object (snap-to-gripper)")
        print("="*60 + "\n")
        
        running = True
        while running:
            # Step physics
            self.sim.step_physics(1.0/60.0)
                        # Update camera to follow robot (third-person view)
            if self.robot:
                # Position camera behind and above robot
                robot_pos = self.robot.translation
                robot_rot = self.robot.rotation
                
                # Camera offset: behind the robot
                back_offset = robot_rot.transform_vector(mn.Vector3(0, 0, self.camera_distance))
                up_offset = mn.Vector3(0, 0.8, 0)  # 0.8m above ground (lower camera)
                camera_pos = robot_pos + back_offset + up_offset
                
                # Create camera rotation with pitch
                # Start with robot's yaw rotation
                yaw_quat = robot_rot
                # Add pitch (tilt down to see robot)
                pitch_quat = mn.Quaternion.rotation(mn.Rad(self.camera_pitch), mn.Vector3(1, 0, 0))
                camera_quat = yaw_quat * pitch_quat
                
                agent_state = self.agent.get_state()
                agent_state.position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
                agent_state.rotation = np.array([camera_quat.vector.x, camera_quat.vector.y, camera_quat.vector.z, camera_quat.scalar])
                self.agent.set_state(agent_state)
                        # Update held object position
            self.update_held_object()
            
            # Get observation
            obs = self.sim.get_sensor_observations()
            frame = obs["color_sensor"]
            
            # Convert RGBA to BGR for OpenCV
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # Add UI
            frame = self.render_ui(frame)
            
            # Display
            cv2.imshow("Robot Manipulation Demo", frame)
            
            # Handle input
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
            elif key == ord('1'):
                self.control_robot_joint(0, 0.1)
            elif key == ord('2'):
                self.control_robot_joint(1, 0.1)
            elif key == ord('3'):
                self.control_robot_joint(2, 0.1)
            elif key == ord('4'):
                self.control_robot_joint(3, 0.1)
            elif key == ord('5'):
                self.control_robot_joint(4, 0.1)
            # Movement controls - move the ROBOT with collision detection
            elif key == ord('w'):
                # Move robot forward
                if self.robot:
                    # Use Magnum's quat rotation instead of habitat utils
                    forward = self.robot.rotation.transform_vector(mn.Vector3(0, 0, -1))
                    new_pos = self.robot.translation + forward * 0.1
                    
                    # Check if new position is navigable (prevents walking through walls)
                    pf = self.sim.pathfinder
                    if pf.is_loaded and pf.is_navigable(new_pos):
                        self.robot.translation = new_pos
                    elif pf.is_loaded:
                        # Try to snap to nearest valid position
                        snapped_pos = pf.snap_point(new_pos)
                        if snapped_pos != new_pos:  # Only update if snap found valid point
                            self.robot.translation = snapped_pos
                    else:
                        # Pathfinder not available, allow movement (no collision)
                        self.robot.translation = new_pos
            elif key == ord('s'):
                # Move robot backward
                if self.robot:
                    backward = self.robot.rotation.transform_vector(mn.Vector3(0, 0, 1))
                    new_pos = self.robot.translation + backward * 0.1
                    
                    # Check if new position is navigable
                    pf = self.sim.pathfinder
                    if pf.is_loaded and pf.is_navigable(new_pos):
                        self.robot.translation = new_pos
                    elif pf.is_loaded:
                        snapped_pos = pf.snap_point(new_pos)
                        if snapped_pos != new_pos:
                            self.robot.translation = snapped_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('a'):
                # Turn robot left
                if self.robot:
                    self.robot.rotation = self.robot.rotation * mn.Quaternion.rotation(
                        mn.Rad(0.1), mn.Vector3(0, 1, 0))
            elif key == ord('d'):
                # Turn robot right
                if self.robot:
                    self.robot.rotation = self.robot.rotation * mn.Quaternion.rotation(
                        mn.Rad(-0.1), mn.Vector3(0, 1, 0))
            elif key == ord('q'):
                # Strafe robot left
                if self.robot:
                    left = self.robot.rotation.transform_vector(mn.Vector3(-1, 0, 0))
                    new_pos = self.robot.translation + left * 0.1
                    
                    # Check if new position is navigable
                    pf = self.sim.pathfinder
                    if pf.is_loaded and pf.is_navigable(new_pos):
                        self.robot.translation = new_pos
                    elif pf.is_loaded:
                        snapped_pos = pf.snap_point(new_pos)
                        if snapped_pos != new_pos:
                            self.robot.translation = snapped_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('e'):
                # Strafe robot right
                if self.robot:
                    right = self.robot.rotation.transform_vector(mn.Vector3(1, 0, 0))
                    new_pos = self.robot.translation + right * 0.1
                    
                    # Check if new position is navigable
                    pf = self.sim.pathfinder
                    if pf.is_loaded and pf.is_navigable(new_pos):
                        self.robot.translation = new_pos
                    elif pf.is_loaded:
                        snapped_pos = pf.snap_point(new_pos)
                        if snapped_pos != new_pos:
                            self.robot.translation = snapped_pos
                    else:
                        self.robot.translation = new_pos
            elif key == ord('z'):
                # Camera pitch up (look more up)
                self.camera_pitch = max(self.camera_pitch - 0.1, -1.0)
                print(f"📷 Camera pitch: {self.camera_pitch:.2f}")
            elif key == ord('x'):
                # Camera pitch down (look more down)
                self.camera_pitch = min(self.camera_pitch + 0.1, 1.5)
                print(f"📷 Camera pitch: {self.camera_pitch:.2f}")
        
        cv2.destroyAllWindows()
        print("\n✅ Demo completed!")

def main():
    print("🚀 Starting Robot Manipulation Demo...")
    print("\n" + "="*60)
    print("🏠 SELECT ENVIRONMENT:")
    print("="*60)
    for key, scene in RobotManipulationDemo.SCENES.items():
        print(f"   {key}. {scene['name']} (ID: {scene['id']})")
    print("   0. Custom scene ID")
    print("="*60)
    
    choice = input("\nEnter your choice (1-8, or 0 for custom): ").strip()
    
    if choice == '0':
        scene_id = input("Enter scene ID: ").strip()
    elif choice in RobotManipulationDemo.SCENES:
        scene_id = RobotManipulationDemo.SCENES[choice]['id']
        print(f"\n✅ Selected: {RobotManipulationDemo.SCENES[choice]['name']}")
    else:
        print("\n⚠️  Invalid choice, using default (Modern Apartment)")
        scene_id = '102344280'
    
    try:
        demo = RobotManipulationDemo(scene_id=scene_id)
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
