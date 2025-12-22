#!/usr/bin/env python3
"""
Interactive Robot Manipulation Demo
Spawn objects and control the Stretch robot to pick them up!

Controls:
  W/S  - Move forward/backward
  A/D  - Turn left/right
  Q/E  - Strafe left/right
  C    - Toggle camera (1st/3rd person)
  Z/X  - Rotate camera (3rd person only)
  J/K  - Lift arm down/up
  U/I  - Retract/extend arm
  1-5  - Fine joint control (Shift+1-5 reverse)
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
    
    # Available robots
    ROBOTS = {
        '1': {'name': 'Stretch', 'path': 'robots/hab_stretch/urdf/hab_stretch.urdf'},
        '2': {'name': 'Spot', 'path': 'robots/hab_spot_arm/urdf/hab_spot_arm.urdf'},
    }
    
    def __init__(self, scene_id='102344280', robot_type='1'):
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
        self.camera_yaw = 0.0  # Independent camera rotation for third-person
        self.scene_id = scene_id  # Selected scene
        self.robot_type = robot_type  # Selected robot type
        self.held_object = None  # Track grasped object
        self.gripper_link_id = None  # Gripper attachment point
        self.camera_mode = 'third_person'  # 'first_person' or 'third_person'
        self.humanoids = []  # List of spawned humanoid avatars
        self.main_humanoid = None  # The humanoid we control
        self.humanoid_head_link_id = None  # Head link for camera
        self.walk_motion = None  # Walking animation data
        self.motion_frame = 0.0  # Current animation frame (float for smooth interpolation)
        self.is_walking = False  # Whether humanoid is currently walking
        self.num_walk_frames = 0  # Total number of animation frames
        self.motion_fps = 30.0  # FPS of the motion capture data
        self.prev_position = None  # Track previous position for distance calculation
        self.debug_frame_counter = 0  # Count frames for periodic debug output
        
    def setup_simulator(self):
        """Setup Habitat-Sim with physics enabled"""
        print("🤖 Setting up Interactive Robot Manipulation Demo...")
        
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
        sim_cfg.enable_physics = True  # Required for object interaction
        sim_cfg.requires_textures = True
        
        # --- ENABLE HSSD HIGH-FIDELITY RENDERING ---
        # These flags improve material rendering and visual quality
        sim_cfg.override_scene_light_defaults = True  # Use improved default lighting
        sim_cfg.enable_hbao = True  # Ambient occlusion to prevent floating look
        
        # Setup sensors - will attach to robot head after loading
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
        rgb_sensor.position = [0.0, 0.0, 0.0]  # Position relative to attachment point
        
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
        
        # Load robots as static scene objects
        self.load_scene_robots()
        
        # Spawn main controllable humanoid
        self.spawn_main_humanoid()
        
        print("✅ Demo ready! Press 'O' to spawn objects!")
        print("👤 You are controlling a humanoid avatar")
        print("🤖 Robots are placed in the scene - explore to find them!")
        print("📹 Camera: Third-person view (Press 'C' to toggle)")
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
    
    def update_camera(self):
        """Update camera position based on current mode"""
        # Follow humanoid if available, otherwise robot
        target = self.main_humanoid if self.main_humanoid else self.robot
        if not target:
            return
        
        agent_state = self.agent.get_state()
        
        if self.camera_mode == 'first_person':
            # First-person: camera from character head
            if self.main_humanoid and self.humanoid_head_link_id is not None:
                head_node = self.main_humanoid.get_link_scene_node(self.humanoid_head_link_id)
            elif self.robot and self.head_link_id is not None:
                head_node = self.robot.get_link_scene_node(self.head_link_id)
            else:
                # Fallback to body position
                target_pos = target.translation
                agent_state.position = np.array([target_pos.x, target_pos.y + 1.6, target_pos.z])
                self.agent.set_state(agent_state)
                return
            
            head_pos = head_node.absolute_translation
            head_rot_matrix = head_node.absolute_transformation().rotation()
            head_quat = mn.Quaternion.from_matrix(head_rot_matrix)
            
            # Position camera in front of head
            camera_offset = head_quat.transform_vector(mn.Vector3(0, 0.05, -0.35))
            camera_pos = head_pos + camera_offset
            
            agent_state.position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
            agent_state.rotation = np.quaternion(head_quat.scalar, *head_quat.vector)
        
        else:  # third_person
            # Third-person: camera orbits around character with independent rotation
            robot_pos = target.translation
            
            # Position camera using independent camera yaw (not tied to robot rotation)
            # This allows robot to rotate without spinning the camera
            camera_height = 1.5
            camera_offset_x = self.camera_distance * np.sin(self.camera_yaw)
            camera_offset_z = self.camera_distance * np.cos(self.camera_yaw)
            
            camera_pos = robot_pos + mn.Vector3(camera_offset_x, camera_height, camera_offset_z)
            
            # Look at robot center
            look_at = robot_pos + mn.Vector3(0, 0.5, 0)
            
            # Calculate camera orientation to look at robot
            forward = (look_at - camera_pos).normalized()
            right = mn.math.cross(forward, mn.Vector3(0, 1, 0)).normalized()
            up = mn.math.cross(right, forward).normalized()
            
            # Build rotation from axes
            rotation_matrix = mn.Matrix3x3(
                right,
                up,
                -forward  # Camera looks down -Z
            )
            rotation_quat = mn.Quaternion.from_matrix(rotation_matrix)
            
            agent_state.position = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
            agent_state.rotation = np.quaternion(rotation_quat.scalar, *rotation_quat.vector)
        
        self.agent.set_state(agent_state)
    
    def toggle_camera_mode(self):
        """Switch between first-person and third-person camera"""
        if self.camera_mode == 'first_person':
            self.camera_mode = 'third_person'
            # Reset camera yaw to be behind robot
            if self.robot:
                # Get robot's current facing direction and position camera behind it
                robot_rot = self.robot.rotation
                forward = robot_rot.transform_vector(mn.Vector3(0, 0, -1))
                self.camera_yaw = np.arctan2(forward.x, forward.z) + np.pi  # Behind robot
            print("📹 Camera: Third-person view (Z/X to rotate camera)")
        else:
            self.camera_mode = 'first_person'
            print("📹 Camera: First-person view")
    
    def load_scene_robots(self):
        """Load both robots as static objects in the scene"""
        print("🤖 Placing robots in the environment...")
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        
        pf = self.sim.pathfinder
        ao_mgr = self.sim.get_articulated_object_manager()
        
        # Load both Stretch and Spot robots at random navigable locations
        for robot_key, robot_info in self.ROBOTS.items():
            urdf_path = os.path.join(data_path, robot_info['path'])
            
            try:
                robot = ao_mgr.add_articulated_object_from_urdf(
                    filepath=urdf_path,
                    fixed_base=True
                )
                
                if robot:
                    # Position at random navigable point
                    if pf.is_loaded:
                        spawn_point = pf.get_random_navigable_point()
                    else:
                        spawn_point = mn.Vector3(float(robot_key) * 3.0, 0.0, 0.0)
                    
                    robot.translation = spawn_point
                    robot.rotation = mn.Quaternion.rotation(
                        mn.Rad(np.random.uniform(0, 6.28)), mn.Vector3(0, 1, 0)
                    )
                    robot.motion_type = habitat_sim.physics.MotionType.STATIC
                    
                    print(f"  ✅ {robot_info['name']} placed at {spawn_point}")
            except Exception as e:
                print(f"  ⚠️ Failed to load {robot_info['name']}: {e}")
    
    def load_robot(self):
        """Legacy method - now loads scene robots instead"""
        self.load_scene_robots()
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        urdf_path = os.path.join(data_path, self.ROBOTS['1']['path'])
        
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
        """Spawn a manipulatable object in front of robot"""
        if position is None:
            # Spawn in front of robot
            if self.robot:
                robot_pos = self.robot.translation
                robot_rot = self.robot.rotation
                # Spawn 1m in front of robot with slight random offset
                import random
                forward = robot_rot.transform_vector(mn.Vector3(0, 0, -1.0))
                lateral_offset = random.uniform(-0.2, 0.2)  # ±0.2m left/right
                lateral = robot_rot.transform_vector(mn.Vector3(lateral_offset, 0, 0))
                spawn_pos = robot_pos + forward + lateral
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
        data_path = os.path.join(os.path.dirname(__file__), "data")
        
        # Try to use one of the block assets
        block_paths = [
            "objects/red_block.urdf",
            "objects/blue_block.urdf",
            "objects/yellow_block.urdf",
            "objects/green_block.urdf"
        ]
        
        ao_mgr = self.sim.get_articulated_object_manager()
        
        for block_path in block_paths:
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
                        return obj
                except Exception as e:
                    continue
        
        print("⚠️  No objects available to spawn")
        return None
    
    def spawn_humanoid(self, position=None):
        """Spawn a humanoid avatar in the scene"""
        if position is None:
            # Spawn near robot
            if self.robot:
                robot_pos = self.robot.translation
                robot_rot = self.robot.rotation
                # Spawn 2m to the side of robot
                import random
                angle = random.uniform(0, 6.28)  # Random angle
                offset = mn.Vector3(2.0 * np.cos(angle), 0, 2.0 * np.sin(angle))
                spawn_pos = robot_pos + offset
                position = [spawn_pos.x, 0.0, spawn_pos.z]
            else:
                position = [3.0, 0.0, 0.0]
        
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        
        # Use female_0 humanoid (you can randomize this)
        humanoid_types = ['female_0', 'female_1', 'male_0', 'male_1', 'neutral_0']
        humanoid_type = humanoid_types[len(self.humanoids) % len(humanoid_types)]
        urdf_path = os.path.join(data_path, f"humanoids/{humanoid_type}/{humanoid_type}.urdf")
        
        if not os.path.exists(urdf_path):
            print(f"⚠️  Humanoid URDF not found: {urdf_path}")
            return None
        
        ao_mgr = self.sim.get_articulated_object_manager()
        try:
            humanoid = ao_mgr.add_articulated_object_from_urdf(
                filepath=urdf_path,
                fixed_base=False
            )
            
            if humanoid:
                humanoid.translation = mn.Vector3(position[0], position[1], position[2])
                humanoid.rotation = mn.Quaternion.rotation(mn.Rad(0), mn.Vector3(0, 1, 0))
                humanoid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                
                self.humanoids.append({
                    'object': humanoid,
                    'type': humanoid_type,
                    'position': position,
                    'id': len(self.humanoids)
                })
                
                print(f"👤 Spawned {humanoid_type} humanoid #{len(self.humanoids)} at {position}")
                return humanoid
        except Exception as e:
            print(f"❌ Failed to spawn humanoid: {e}")
        
        return None
    
    def spawn_main_humanoid(self):
        """Spawn the main controllable humanoid avatar"""
        print("👤 Spawning controllable humanoid avatar...")
        
        import os
        data_path = os.path.join(os.path.dirname(__file__), "data")
        
        # Use neutral_0 as the main character
        humanoid_type = 'male_1'
        urdf_path = os.path.join(data_path, f"humanoids/{humanoid_type}/{humanoid_type}.urdf")
        
        if not os.path.exists(urdf_path):
            print(f"⚠️ Humanoid URDF not found: {urdf_path}")
            print("   Falling back to robot camera mode")
            return
        
        ao_mgr = self.sim.get_articulated_object_manager()
        pf = self.sim.pathfinder
        
        try:
            self.main_humanoid = ao_mgr.add_articulated_object_from_urdf(
                filepath=urdf_path,
                fixed_base=False
            )
            
            if self.main_humanoid:
                # Spawn at random navigable point with proper height
                if pf.is_loaded:
                    spawn_point = pf.get_random_navigable_point()
                else:
                    spawn_point = mn.Vector3(0.0, 0.0, 0.0)
                
                # Humanoid URDF origin is at body center, not feet
                # For SMPL-X humanoids, center is approximately 0.9m above feet
                spawn_point.y = spawn_point.y + 0.9  # Lift center above navmesh
                
                self.main_humanoid.translation = spawn_point
                self.main_humanoid.rotation = mn.Quaternion.rotation(mn.Rad(0), mn.Vector3(0, 1, 0))
                # Keep as KINEMATIC to prevent physics issues
                self.main_humanoid.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                
                print(f"  📍 Humanoid position: {spawn_point}, y={spawn_point.y}")
                
                # Find head link for camera
                for link_id in range(self.main_humanoid.num_links):
                    link_name = self.main_humanoid.get_link_name(link_id)
                    if "head" in link_name.lower():
                        self.humanoid_head_link_id = link_id
                        break
                
                print(f"  ✅ {humanoid_type} spawned at {spawn_point}")
                if self.humanoid_head_link_id:
                    print(f"  📷 Head link ID: {self.humanoid_head_link_id}")
                else:
                    print(f"  ⚠️ No head link found, using body center for camera")
                
                # Load walking motion data
                self.load_walk_motion()
        except Exception as e:
            print(f"❌ Failed to spawn main humanoid: {e}")
            self.main_humanoid = None
    
    def load_walk_motion(self):
        """Load walking motion data from pickle file"""
        import os
        import pickle
        
        data_path = os.path.join(os.path.dirname(__file__), "data")
        motion_path = os.path.join(data_path, "humanoids/walking_motion_processed_smplx.pkl")
        
        if not os.path.exists(motion_path):
            print(f"  ⚠️ Walking motion file not found: {motion_path}")
            return
        
        try:
            with open(motion_path, 'rb') as f:
                self.walk_motion = pickle.load(f)
            
            # Debug: print what keys are in the motion data
            if isinstance(self.walk_motion, dict):
                print(f"  📊 Motion data keys: {list(self.walk_motion.keys())}")
                
                # Extract number of frames
                if 'walk_motion' in self.walk_motion:
                    motion_dict = self.walk_motion['walk_motion']
                    if 'joints_array' in motion_dict:
                        self.num_walk_frames = len(motion_dict['joints_array'])
                        print(f"  ✅ Walking motion loaded ({self.num_walk_frames} frames)")
                        print(f"  📊 Pose data shape: {motion_dict['joints_array'].shape}")
                        print(f"  📊 Humanoid DOF: {len(self.main_humanoid.joint_positions)}")
                        
                        # Apply first frame immediately to override URDF default pose
                        poses = motion_dict['joints_array']
                        initial_pose = poses[0].flatten()
                        new_positions = list(self.main_humanoid.joint_positions)
                        num_to_update = min(len(initial_pose), len(new_positions))
                        for i in range(num_to_update):
                            new_positions[i] = float(initial_pose[i])
                        self.main_humanoid.joint_positions = new_positions
                        print(f"  ✅ Applied initial pose from first animation frame")
            else:
                print(f"  ⚠️ Motion data is not a dict: {type(self.walk_motion)}")
        except Exception as e:
            print(f"  ⚠️ Failed to load motion data: {e}")
            self.walk_motion = None
    
    def update_humanoid_animation(self):
        """Update humanoid animation with smooth interpolation"""
        if not self.main_humanoid or not self.walk_motion:
            return
        
        # Calculate actual distance moved this frame
        current_pos = self.main_humanoid.translation
        if self.prev_position is None:
            self.prev_position = current_pos
        
        distance_moved = (current_pos - self.prev_position).length()
        self.prev_position = current_pos
        
        # Only advance animation when actually walking
        if self.is_walking and self.num_walk_frames > 0 and distance_moved > 0.001:
            # Advance proportionally to distance moved
            frames_to_advance = distance_moved * 30.0
            self.motion_frame += frames_to_advance
            if self.motion_frame >= self.num_walk_frames:
                self.motion_frame = self.motion_frame % self.num_walk_frames
            
            # Update pose only when walking
            self._manual_update_pose()
        # When idle, just freeze at current frame (don't update pose)
    
    def _manual_update_pose(self):
        """Manual pose update with smooth LERP interpolation"""
        if not self.walk_motion:
            return
        
        try:
            # Extract motion data
            if 'walk_motion' in self.walk_motion:
                motion_dict = self.walk_motion['walk_motion']
                if 'joints_array' in motion_dict:
                    poses = motion_dict['joints_array']  # Shape: (num_frames, 54, 4)
                else:
                    return
            else:
                return
            
            # Use LERP between frames
            frame_a = int(np.floor(self.motion_frame))
            frame_b = int(np.ceil(self.motion_frame)) % self.num_walk_frames
            alpha = self.motion_frame - frame_a  # Fractional part (0.0 to 1.0)
            
            # Get poses for interpolation
            pose_a = poses[frame_a].flatten()
            pose_b = poses[frame_b].flatten()
            
            # Debug: Check which joints are actually changing
            if hasattr(self, '_debug_counter'):
                self._debug_counter += 1
            else:
                self._debug_counter = 0
                self._prev_pose = None
            
            if self._debug_counter % 300 == 0:  # Every 5 seconds at 60 FPS
                print(f"  🔍 Frame {self.motion_frame:.2f}: First 12 pose values: {pose_a[:12]}")
                
                # Check which joints are NOT changing
                if self._prev_pose is not None:
                    diff = np.abs(pose_a - self._prev_pose)
                    frozen_joints = []
                    for i in range(0, min(len(diff), 216), 4):  # Check each joint (4 values per joint)
                        joint_diff = np.max(diff[i:i+4])
                        if joint_diff < 0.001:  # Joint hasn't moved
                            frozen_joints.append(i // 4)
                    if frozen_joints:
                        print(f"  ⚠️  Frozen joints (not changing): {frozen_joints[:10]}...")  # Show first 10
                
                self._prev_pose = pose_a.copy()
            
            # Linear interpolation (LERP) between the two poses
            blended_pose = (1.0 - alpha) * pose_a + alpha * pose_b
            
            # Apply the blended pose to the humanoid
            new_positions = list(self.main_humanoid.joint_positions)
            num_dof = len(new_positions)
            num_to_update = min(len(blended_pose), num_dof)
            
            for i in range(num_to_update):
                new_positions[i] = float(blended_pose[i])
            
            # DEBUG: Print shoulder joint values from mocap data
            if self._debug_counter % 300 == 0 and self.is_walking:
                print(f"  🔍 SHOULDER DEBUG:")
                print(f"     Left shoulder (DOF 64-67): {blended_pose[64:68]}")
                print(f"     Right shoulder (DOF 68-71): {blended_pose[68:72]}")
                # Try other potential shoulder locations
                print(f"     Alt location 1 (52-55): {blended_pose[52:56]}")
                print(f"     Alt location 2 (56-59): {blended_pose[56:60]}")
            
            # OVERRIDE: Manually animate ONLY left arm (right arm seems fine in mocap)
            if self.is_walking:
                # Create arm swing based on walk cycle
                swing_amplitude = 0.5  # Radians (~29 degrees)
                swing = np.sin(self.motion_frame * 0.2) * swing_amplitude
                
                # Left shoulder (DOF 64-67) - forward/back swing
                # Proper quaternion for rotation around X-axis: (sin(θ/2), 0, 0, cos(θ/2))
                left_shoulder_start = 64
                if left_shoulder_start + 3 < num_dof:
                    angle = swing
                    new_positions[left_shoulder_start] = np.sin(angle / 2)  # x component
                    new_positions[left_shoulder_start + 1] = 0.0  # y component
                    new_positions[left_shoulder_start + 2] = 0.0  # z component
                    new_positions[left_shoulder_start + 3] = np.cos(angle / 2)  # w component
                
                # DON'T override right shoulder - let mocap handle it
            
            self.main_humanoid.joint_positions = new_positions
            
        except Exception as e:
            print(f"❌ Animation error: {e}")
            import traceback
            traceback.print_exc()
    
    def control_robot_joint(self, joint_id, delta):
        """Move a robot joint"""
        if self.robot is None:
            return
        
        current_pos = self.robot.joint_positions
        if joint_id < len(current_pos):
            new_pos = list(current_pos)
            old_val = new_pos[joint_id]
            new_pos[joint_id] += delta
            
            # Apply limits (Stretch robot typical ranges)
            # Joint 0: Not actuated - ignored
            # Joints 1-2 (arm extension): 0.0 to 0.5m each (telescoping segments)
            # Joint 3 (LIFT): 0.0 to 0.3m (moves arm up/down - restricted)
            # Joint 4 (EXTENSION): 0.0 to 0.15m (stretches arm forward - restricted)
            if joint_id == 0:
                new_pos[joint_id] = max(0.0, min(0.01, new_pos[joint_id]))  # Essentially disabled
            elif joint_id <= 2:
                new_pos[joint_id] = max(0.0, min(0.5, new_pos[joint_id]))
            elif joint_id == 3:
                new_pos[joint_id] = max(0.0, min(0.3, new_pos[joint_id]))  # LIFT - reduced
            elif joint_id == 4:
                new_pos[joint_id] = max(0.0, min(0.15, new_pos[joint_id]))  # EXTENSION - reduced
            else:
                # Other joints (wrist, gripper): general safe limit
                new_pos[joint_id] = max(-3.14, min(3.14, new_pos[joint_id]))
            
            self.robot.joint_positions = new_pos
            
            # Debug output with actual achieved value
            actual_val = self.robot.joint_positions[joint_id]
            if abs(actual_val - old_val) < 0.001:
                print(f"⚠️  Joint {joint_id}: NO MOVEMENT (stuck at {old_val:.3f}, tried {delta:+.2f})")
            else:
                print(f"🦾 Joint {joint_id}: {old_val:.3f} → {actual_val:.3f} ({delta:+.2f})")
    
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
        camera_mode_text = "1st Person" if self.camera_mode == 'first_person' else "3rd Person"
        controls_line2 = "JK-Lift | UI-Arm | G-Grip | O-Spawn | P-Pick"
        if self.camera_mode == 'third_person':
            controls_line2 = "ZX-Cam | JK-Lift | UI-Arm | G-Grip | O/P"
        
        instructions = [
            "🎮 CONTROLS:",
            "WS - Move | AD - Turn | QE - Strafe | C - Camera",
            controls_line2,
            "",
            f"📹 Camera: {camera_mode_text}",
            f"🚶 Animation: {'Walking' if self.is_walking else 'Idle'} (Frame: {self.motion_frame})",
            f"🤖 Objects: {len(self.objects)}",
            f"👤 Humanoids: {len(self.humanoids)}",
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
        print("   W/S - Move forward/backward (with wall sliding)")
        print("   A/D - Turn left/right")
        print("   Q/E - Strafe left/right")
        print("   C - Toggle camera (1st/3rd person)")
        print("   Z/X - Rotate camera around robot (3rd person)")
        print("   J/K - Lift arm DOWN/UP")
        print("   U/I - Retract/Extend arm")
        print("   1-5 - Fine control joints (Shift+1-5 reverse)")
        print("   O - Spawn object in front of robot")
        print("   H - Spawn humanoid avatar")
        print("   G - Open/close gripper")
        print("   P - Pick/drop object")
        print("   R - Reset all joints")
        print("="*60 + "\n")
        
        running = True
        while running:
            # Handle input FIRST (before resetting is_walking)
            key = cv2.waitKey(1) & 0xFF
            
            # Reset walking state each frame (will be set True by movement keys below)
            self.is_walking = False
            
            if key == 27:  # ESC
                running = False
            elif key == ord('o') or key == ord('O'):
                self.spawn_object()
            elif key == ord('h') or key == ord('H'):
                self.spawn_humanoid()
            elif key == ord('g') or key == ord('G'):
                self.toggle_gripper()
            elif key == ord('p') or key == ord('P'):
                self.pick_nearest_object()
            elif key == ord('r') or key == ord('R'):
                if self.robot:
                    self.robot.joint_positions = [0.0] * len(self.robot.joint_positions)
                    print("🔄 Robot reset")
            elif key == ord('1'):
                self.control_robot_joint(0, 0.05)  # Lift up (smaller increment)
            elif key == ord('!'):
                self.control_robot_joint(0, -0.05)  # Lift down (Shift+1)
            elif key == ord('2'):
                self.control_robot_joint(1, 0.05)  # Arm extend segment 1
            elif key == ord('@'):
                self.control_robot_joint(1, -0.05)  # Arm retract (Shift+2)
            elif key == ord('3'):
                self.control_robot_joint(2, 0.05)  # Arm extend segment 2
            elif key == ord('#'):
                self.control_robot_joint(2, -0.05)  # Shift+3
            elif key == ord('4'):
                self.control_robot_joint(3, 0.05)  # Arm extend segment 3
            elif key == ord('$'):
                self.control_robot_joint(3, -0.05)  # Shift+4
            elif key == ord('5'):
                self.control_robot_joint(4, 0.05)  # Wrist/head joints
            elif key == ord('%'):
                self.control_robot_joint(4, -0.05)  # Shift+5
            # Alternative controls: J/K for lift, U/I for arm extension
            elif key == ord('j') or key == ord('J'):
                self.control_robot_joint(0, -0.05)  # Lift DOWN
            elif key == ord('k') or key == ord('K'):
                self.control_robot_joint(0, 0.05)  # Lift UP
            elif key == ord('u') or key == ord('U'):
                self.control_robot_joint(1, -0.05)  # Arm RETRACT
            elif key == ord('i') or key == ord('I'):
                self.control_robot_joint(1, 0.05)  # Arm EXTEND
            elif key == ord('c') or key == ord('C'):
                self.toggle_camera_mode()
            elif key == ord('z') or key == ord('Z'):
                # Rotate camera left (third-person only)
                if self.camera_mode == 'third_person':
                    self.camera_yaw -= 0.1
            elif key == ord('x') or key == ord('X'):
                # Rotate camera right (third-person only)
                if self.camera_mode == 'third_person':
                    self.camera_yaw += 0.1
            # Movement controls - move the HUMANOID with sliding collision
            elif key == ord('w'):
                # Move humanoid forward with sliding along walls
                self.is_walking = True  # Start walking animation
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    forward = target.rotation.transform_vector(mn.Vector3(0, 0, 1))  # Fixed: +Z is forward
                    new_pos = target.translation + forward * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        # try_step allows sliding along walls instead of stopping
                        final_pos = pf.try_step(target.translation, new_pos)
                        # Keep humanoid at correct height above navmesh
                        if self.main_humanoid:
                            final_pos.y = final_pos.y + 0.9  # Body center offset
                        target.translation = final_pos
                    else:
                        target.translation = new_pos
            elif key == ord('s'):
                # Move backward with sliding
                self.is_walking = True  # Start walking animation
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    backward = target.rotation.transform_vector(mn.Vector3(0, 0, -1))  # Fixed: -Z is backward
                    new_pos = target.translation + backward * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(target.translation, new_pos)
                        # Keep humanoid at correct height above navmesh
                        if self.main_humanoid:
                            final_pos.y = final_pos.y + 0.9  # Body center offset
                        target.translation = final_pos
                    else:
                        target.translation = new_pos
            elif key == ord('a'):
                # Turn left
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    target.rotation = target.rotation * mn.Quaternion.rotation(
                        mn.Rad(0.1), mn.Vector3(0, 1, 0))
            elif key == ord('d'):
                # Turn right
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    target.rotation = target.rotation * mn.Quaternion.rotation(
                        mn.Rad(-0.1), mn.Vector3(0, 1, 0))
            elif key == ord('q'):
                # Strafe left with sliding
                self.is_walking = True  # Start walking animation
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    left = target.rotation.transform_vector(mn.Vector3(-1, 0, 0))
                    new_pos = target.translation + left * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(target.translation, new_pos)
                        # Keep humanoid at correct height above navmesh
                        if self.main_humanoid:
                            final_pos.y = final_pos.y + 0.9  # Body center offset
                        target.translation = final_pos
                    else:
                        target.translation = new_pos
            elif key == ord('e'):
                # Strafe right with sliding
                self.is_walking = True  # Start walking animation
                target = self.main_humanoid if self.main_humanoid else self.robot
                if target:
                    right = target.rotation.transform_vector(mn.Vector3(1, 0, 0))
                    new_pos = target.translation + right * 0.1
                    
                    pf = self.sim.pathfinder
                    if pf.is_loaded:
                        final_pos = pf.try_step(target.translation, new_pos)
                        # Keep humanoid at correct height above navmesh
                        if self.main_humanoid:
                            final_pos.y = final_pos.y + 0.9  # Body center offset
                        target.translation = final_pos
                    else:
                        target.translation = new_pos
        
            # NOW do physics and animation updates (after is_walking is set)
            # Step physics
            self.sim.step_physics(1.0/60.0)
            
            # Update humanoid animation
            self.update_humanoid_animation()
            
            # Update camera position based on mode
            self.update_camera()
            
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
        
        cv2.destroyAllWindows()
        print("\n✅ Demo completed!")

def main():
    print("🚀 Starting Humanoid Exploration Demo...")
    print("👤 Control a humanoid avatar and explore environments!")
    print("🤖 Both Stretch and Spot robots are placed in the scene")
    
    # Select scene
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
        demo = RobotManipulationDemo(scene_id=scene_id, robot_type='1')
        demo.run()
    except KeyboardInterrupt:
        print("\n⏹️  Demo interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
