#!/usr/bin/env python3
"""
Furnished House Robot Demo
Interactive exploration of HSSD furnished house with one Stretch robot
Features: W/A/S/D navigation, robot discovery, UI overlays, and rich environment
"""

import habitat_sim
import numpy as np
import cv2
import time
import math
import signal
import sys
import magnum as mn

class InteractiveRoboticsDemo:
    def __init__(self):
        self.sim = None
        self.agent = None
        self.robots = self._define_robots()
        self.stats = {
            'steps': 0,
            'discovered_robots': set(),
            'start_time': time.time()
        }
        self.robot_objects = []  # Track spawned robot objects in scene
        self.ui_settings = {
            'show_robot_info': True,
            'show_minimap': False,
            'show_compass': True,
            'show_detailed_info': False
        }
        
    def _define_robots(self):
        """Define robot profiles with positions, capabilities, and metadata"""
        return [
            {
                "id": 0,
                "name": "Stretch Home Assistant", 
                "position": [2.0, 0.0, 1.5], 
                "color": [1.0, 0.5, 0.0],  # Orange
                "type": "Mobile Manipulator",
                "manufacturer": "Hello Robot",
                "capabilities": ["Home navigation", "Object retrieval", "Telepresence", "Assistive tasks"],
                "joints": 13,
                "description": "Hello Robot Stretch - specifically designed for home environments",
                "size": "Compact (1.2m tall, 33cm wide)",
                "workspace": "Homes, assisted living, personal assistance"
            }
        ]
    
    def setup_simulator(self):
        """Initialize Habitat-Sim with optimal settings"""
        print("🏠 Setting up fully furnished HSSD house with objects and furniture...")
        
        sim_settings = {
            "scene_dataset": "/home/jinyoonok/Projects/InteractiveRobotics/home-robot/data/hssd-hab/hssd-hab.scene_dataset_config.json",
            "scene": "102344280",  # Use scene ID instead of full path
            "default_agent": 0,
            "sensor_height": 1.5,
            "width": 1200,
            "height": 800,
            "hfov": 90,
        }
        
        # Create configuration (no physics to avoid compatibility issues)
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.gpu_device_id = 0
        sim_cfg.scene_dataset_config_file = sim_settings["scene_dataset"]
        sim_cfg.scene_id = sim_settings["scene"]
        sim_cfg.enable_physics = True  # Must be True to use Articulated Object Manager/URDF
        
        # Add asset root path for URDF and mesh loading
        STRETCH_ASSET_ROOT = "/home/jinyoonok/Projects/InteractiveRobotics/home-robot/data/robots"
        # This allows Habitat-Sim to resolve package:// paths in the URDF
        
        # Set up sensors
        rgb_sensor = habitat_sim.CameraSensorSpec()
        rgb_sensor.uuid = "color_sensor"
        rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
        rgb_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
        rgb_sensor.position = [0.0, sim_settings["sensor_height"], 0.0]
        
        depth_sensor = habitat_sim.CameraSensorSpec()
        depth_sensor.uuid = "depth_sensor" 
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
        depth_sensor.position = [0.0, sim_settings["sensor_height"], 0.0]
        
        # Agent configuration
        agent_cfg = habitat_sim.agent.AgentConfiguration()
        agent_cfg.sensor_specifications = [rgb_sensor, depth_sensor]
        
        # Create simulator
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        self.sim = habitat_sim.Simulator(cfg)
        
        # Initialize agent
        self.agent = self.sim.initialize_agent(sim_settings["default_agent"])
        
        # Set starting position
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array([0.0, 0.0, 0.0])
        agent_state.rotation = np.quaternion(1, 0, 0, 0)
        self.agent.set_state(agent_state)
        
        # Spawn robots in the scene
        self.spawn_robots()
        
        print(f"✅ HSSD furnished house loaded with 1 Stretch robot!")
        print("🏠 Exploring a complete furnished house - find the robot!")
        return sim_settings
    
    def spawn_robots(self):
        """Spawn Stretch robot using proper articulated object manager"""
        print("🤖 Loading Stretch robot using articulated object manager...")
        
        robot = self.robots[0]  # Our single Stretch robot
        stretch_urdf_path = "/home/jinyoonok/Projects/InteractiveRobotics/home-robot/data/robots/hab_stretch/urdf/hab_stretch.urdf"
        
        try:
            # Get articulated object manager
            ao_mgr = self.sim.get_articulated_object_manager()
            
            print(f"  📄 Loading Stretch URDF: {stretch_urdf_path}")
            print(f"  🔧 Bullet Physics enabled: {habitat_sim.built_with_bullet}")
            
            # Load the Stretch robot from URDF (now with Bullet physics enabled)
            stretch_robot = ao_mgr.add_articulated_object_from_urdf(
                filepath=stretch_urdf_path,
                fixed_base=False,  # Allow the robot to move
                maintain_link_order=False,
                mass_scale=1.0,
                force_reload=True  # Force reload to ensure fresh loading
            )
            
            if stretch_robot:
                # Set robot position
                stretch_robot.translation = mn.Vector3(
                    robot["position"][0],
                    robot["position"][1],  # Ground level
                    robot["position"][2]
                )
                
                # Set robot orientation (facing forward)
                stretch_robot.rotation = mn.Quaternion.rotation(mn.Rad(0), mn.Vector3(0, 1, 0))
                
                # Make robot kinematic (non-physics) for stability
                stretch_robot.motion_type = habitat_sim.physics.MotionType.KINEMATIC
                
                # Store robot reference
                self.robot_objects.append({
                    'object': stretch_robot,
                    'robot_data': robot,
                    'id': robot['id'],
                    'type': 'articulated_robot'
                })
                
                print(f"  ✅ Successfully spawned Stretch robot!")
                print(f"  🤖 Articulated robot object: {type(stretch_robot)}")
                print(f"  📍 Robot positioned at {robot['position']} and visible in scene!")
                print(f"  🎯 Robot should now be VISIBLE in the simulation!")
                return
                
        except FileNotFoundError:
            print(f"  ❌ URDF file not found: {stretch_urdf_path}")
        except Exception as e:
            print(f"  ❌ Articulated robot loading failed: {e}")
            print(f"      This might indicate URDF parsing or asset issues")
        
        # Fallback to virtual tracking
        print(f"  🔄 Using virtual robot tracking only")
        self.robot_objects.append({
            'object': None,
            'robot_data': robot,
            'id': robot['id'],
            'type': 'virtual'
        })
        
        print(f"  📊 Virtual robot tracking active for discovery system")
    
    def get_current_position(self):
        """Get agent's current position"""
        return self.agent.get_state().position
    
    def get_robot_distances(self):
        """Calculate distances to all robots, sorted by proximity"""
        current_pos = self.get_current_position()
        distances = []
        
        for robot in self.robots:
            distance = np.linalg.norm(current_pos - np.array(robot["position"]))
            distances.append((distance, robot))
        
        return sorted(distances)  # Sort by distance
    
    def update_robot_discovery(self):
        """Check for newly discovered robots"""
        discoveries = []
        robot_distances = self.get_robot_distances()
        
        for distance, robot in robot_distances:
            if distance < 2.5 and robot["id"] not in self.stats['discovered_robots']:
                self.stats['discovered_robots'].add(robot["id"])
                discoveries.append(robot)
        
        return discoveries
    
    def draw_ui_overlay(self, rgb_bgr):
        """Draw all UI elements on the frame"""
        current_pos = self.get_current_position()
        
        # Basic info
        cv2.putText(rgb_bgr, f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})", 
                   (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        discovery_rate = len(self.stats['discovered_robots']) / len(self.robots) * 100
        cv2.putText(rgb_bgr, f"Steps: {self.stats['steps']} | Discovered: {len(self.stats['discovered_robots'])}/{len(self.robots)} ({discovery_rate:.0f}%)", 
                   (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Robot radar
        if self.ui_settings['show_robot_info']:
            self._draw_robot_radar(rgb_bgr)
        
        # Detailed robot info
        if self.ui_settings['show_detailed_info']:
            self._draw_detailed_info(rgb_bgr)
        
        # Mini-map
        if self.ui_settings['show_minimap']:
            self._draw_minimap(rgb_bgr)
        
        # Compass
        if self.ui_settings['show_compass']:
            self._draw_compass(rgb_bgr)
        
        # Controls help
        self._draw_controls(rgb_bgr)
    
    def _draw_robot_radar(self, rgb_bgr):
        """Draw robot proximity radar"""
        robot_distances = self.get_robot_distances()
        
        y_offset = 110
        cv2.putText(rgb_bgr, f"🤖 Robot Radar:", 
                   (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 100), 2)
        
        # Show top 4 nearest robots
        for i, (distance, robot) in enumerate(robot_distances[:4]):
            y_offset += 30
            
            # Status and color based on distance
            if distance < 2.5:
                status = "🟢 DISCOVERED"
                color = (0, 255, 0)
            elif distance < 5.0:
                status = "🟡 NEARBY"  
                color = (0, 255, 255)
            else:
                status = "🔴 DISTANT"
                color = (100, 100, 255)
            
            # Robot type icon
            type_icon = "🦾" if robot["type"] == "Manipulator Arm" else "🤖" if "Humanoid" in robot["type"] else "🐕" if "Quadruped" in robot["type"] else "🚀"
            
            text = f"{type_icon} {robot['name']}: {distance:.1f}m {status}"
            cv2.putText(rgb_bgr, text, (15, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
    
    def _draw_detailed_info(self, rgb_bgr):
        """Draw detailed information about the nearest robot"""
        robot_distances = self.get_robot_distances()
        if not robot_distances:
            return
        
        nearest_distance, nearest_robot = robot_distances[0]
        
        # Info panel background
        panel_x, panel_y = 15, 320
        panel_w, panel_h = 400, 200
        
        # Semi-transparent background
        overlay = rgb_bgr.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, rgb_bgr, 0.2, 0, rgb_bgr)
        cv2.rectangle(rgb_bgr, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), (200, 200, 200), 2)
        
        # Content
        y = panel_y + 25
        cv2.putText(rgb_bgr, f"📋 NEAREST ROBOT DETAILS", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        y += 30
        cv2.putText(rgb_bgr, f"Name: {nearest_robot['name']}", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 25
        cv2.putText(rgb_bgr, f"Type: {nearest_robot['type']} | Manufacturer: {nearest_robot['manufacturer']}", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 20
        cv2.putText(rgb_bgr, f"Distance: {nearest_distance:.2f}m | Joints: {nearest_robot['joints']} | Size: {nearest_robot['size']}", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 25
        cv2.putText(rgb_bgr, f"Primary Use: {nearest_robot['workspace']}", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 255, 150), 1)
        
        # Capabilities (split into lines)
        y += 25
        cv2.putText(rgb_bgr, "Key Capabilities:", (panel_x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)
        caps = nearest_robot['capabilities']
        for i in range(0, len(caps), 2):  # Show 2 capabilities per line
            y += 20
            caps_text = f"• {', '.join(caps[i:i+2])}"
            cv2.putText(rgb_bgr, caps_text, (panel_x + 15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 255), 1)
    
    def _draw_minimap(self, rgb_bgr):
        """Draw mini-map with robot positions"""
        map_size = 220
        map_x = rgb_bgr.shape[1] - map_size - 15
        map_y = 15
        
        # Map background
        cv2.rectangle(rgb_bgr, (map_x, map_y), (map_x + map_size, map_y + map_size), (30, 30, 30), -1)
        cv2.rectangle(rgb_bgr, (map_x, map_y), (map_x + map_size, map_y + map_size), (255, 255, 255), 2)
        
        # Map scale and center
        map_scale = map_size / 12.0  # Apartment spans roughly -6 to +6
        map_center_x = map_x + map_size // 2
        map_center_y = map_y + map_size // 2
        
        # Draw coordinate grid
        for i in range(-2, 3):
            grid_x = int(map_center_x + i * map_scale * 2)
            grid_y = int(map_center_y + i * map_scale * 2) 
            cv2.line(rgb_bgr, (grid_x, map_y + 5), (grid_x, map_y + map_size - 5), (80, 80, 80), 1)
            cv2.line(rgb_bgr, (map_x + 5, grid_y), (map_x + map_size - 5, grid_y), (80, 80, 80), 1)
        
        # Draw player position
        current_pos = self.get_current_position()
        player_x = int(map_center_x + current_pos[0] * map_scale)
        player_y = int(map_center_y + current_pos[2] * map_scale)
        cv2.circle(rgb_bgr, (player_x, player_y), 6, (0, 255, 0), -1)
        cv2.circle(rgb_bgr, (player_x, player_y), 8, (255, 255, 255), 2)
        
        # Draw robots
        for robot in self.robots:
            robot_x = int(map_center_x + robot["position"][0] * map_scale)
            robot_y = int(map_center_y + robot["position"][2] * map_scale)
            
            # Color based on discovery
            if robot["id"] in self.stats['discovered_robots']:
                color = (0, 255, 0)  # Green if discovered
                radius = 5
            else:
                color = (int(robot["color"][2]*255), int(robot["color"][1]*255), int(robot["color"][0]*255))
                radius = 4
            
            cv2.circle(rgb_bgr, (robot_x, robot_y), radius, color, -1)
        
        # Map legend
        cv2.putText(rgb_bgr, "Map View", (map_x + 5, map_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _draw_compass(self, rgb_bgr):
        """Draw navigation compass"""
        compass_x = rgb_bgr.shape[1] - 90
        compass_y = 280
        
        # Get agent rotation
        agent_rotation = self.agent.get_state().rotation
        yaw = math.atan2(2.0 * (agent_rotation.w * agent_rotation.y + agent_rotation.x * agent_rotation.z),
                        1.0 - 2.0 * (agent_rotation.y * agent_rotation.y + agent_rotation.z * agent_rotation.z))
        
        # Compass circle
        cv2.circle(rgb_bgr, (compass_x, compass_y), 35, (100, 100, 100), 3)
        cv2.circle(rgb_bgr, (compass_x, compass_y), 30, (50, 50, 50), -1)
        
        # North indicator
        north_x = int(compass_x + 25 * math.cos(-math.pi/2))
        north_y = int(compass_y + 25 * math.sin(-math.pi/2))
        cv2.arrowedLine(rgb_bgr, (compass_x, compass_y), (north_x, north_y), (0, 100, 255), 3)
        cv2.putText(rgb_bgr, "N", (compass_x - 6, compass_y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 2)
        
        # Agent direction
        dir_x = int(compass_x + 25 * math.cos(yaw - math.pi/2))
        dir_y = int(compass_y + 25 * math.sin(yaw - math.pi/2))
        cv2.arrowedLine(rgb_bgr, (compass_x, compass_y), (dir_x, dir_y), (0, 255, 0), 4)
    
    def _draw_controls(self, rgb_bgr):
        """Draw control instructions"""
        controls = [
            "🎮 WASD: Move & Turn | QE: Strafe | ZX: Look Up/Down | R: Reset",
            "🔧 T: Radar | I: Details | M: Map | C: Compass | ESC: Exit"
        ]
        
        for i, text in enumerate(controls):
            y = rgb_bgr.shape[0] - 40 + i * 20
            cv2.putText(rgb_bgr, text, (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def handle_movement(self, key):
        """Process movement input and return if moved"""
        moved = False
        move_amount = 0.25
        
        if key == ord('w'):
            self.agent.act("move_forward")
            moved = True
            
        elif key == ord('s'):
            # Use manual position update for backward movement
            current_state = self.agent.get_state()
            forward_quat = current_state.rotation * habitat_sim.utils.quat_from_angle_axis(np.pi, np.array([0, 1, 0]))
            backward_direction = habitat_sim.utils.quat_rotate_vector(forward_quat, np.array([0, 0, -1]))
            
            agent_state = habitat_sim.AgentState()
            agent_state.position = current_state.position + backward_direction * move_amount
            agent_state.rotation = current_state.rotation
            self.agent.set_state(agent_state)
            moved = True
            
        elif key == ord('a'):
            self.agent.act("turn_left")
            moved = True
            
        elif key == ord('d'):
            self.agent.act("turn_right")
            moved = True
            
        elif key == ord('z'):  # Look up
            current_state = self.agent.get_state()
            # Create pitch up rotation (around X axis)
            pitch_up = habitat_sim.utils.quat_from_angle_axis(-0.1, np.array([1, 0, 0]))
            new_rotation = current_state.rotation * pitch_up
            
            agent_state = habitat_sim.AgentState()
            agent_state.position = current_state.position
            agent_state.rotation = new_rotation
            self.agent.set_state(agent_state)
            moved = True
            
        elif key == ord('x'):  # Look down
            current_state = self.agent.get_state()
            # Create pitch down rotation (around X axis)
            pitch_down = habitat_sim.utils.quat_from_angle_axis(0.1, np.array([1, 0, 0]))
            new_rotation = current_state.rotation * pitch_down
            
            agent_state = habitat_sim.AgentState()
            agent_state.position = current_state.position
            agent_state.rotation = new_rotation
            self.agent.set_state(agent_state)
            moved = True
            
        elif key == ord('q'):  # Strafe left
            current_state = self.agent.get_state()
            left_quat = current_state.rotation * habitat_sim.utils.quat_from_angle_axis(np.pi/2, np.array([0, 1, 0]))
            left_direction = habitat_sim.utils.quat_rotate_vector(left_quat, np.array([0, 0, -1]))
            
            agent_state = habitat_sim.AgentState()
            agent_state.position = current_state.position + left_direction * move_amount
            agent_state.rotation = current_state.rotation  
            self.agent.set_state(agent_state)
            moved = True
            
        elif key == ord('e'):  # Strafe right
            current_state = self.agent.get_state()
            right_quat = current_state.rotation * habitat_sim.utils.quat_from_angle_axis(-np.pi/2, np.array([0, 1, 0]))
            right_direction = habitat_sim.utils.quat_rotate_vector(right_quat, np.array([0, 0, -1]))
            
            agent_state = habitat_sim.AgentState()
            agent_state.position = current_state.position + right_direction * move_amount
            agent_state.rotation = current_state.rotation
            self.agent.set_state(agent_state)
            moved = True
            
        elif key == ord('r'):  # Reset
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([0.0, 0.0, 0.0])
            agent_state.rotation = np.quaternion(1, 0, 0, 0)
            self.agent.set_state(agent_state)
            self.stats['steps'] = 0
            self.stats['discovered_robots'].clear()
            moved = True
            print("🔄 Reset to starting position")
        
        return moved
    
    def handle_ui_toggles(self, key):
        """Handle UI toggle commands"""
        if key == ord('t'):
            self.ui_settings['show_robot_info'] = not self.ui_settings['show_robot_info']
            print(f"🤖 Robot radar: {'ON' if self.ui_settings['show_robot_info'] else 'OFF'}")
            
        elif key == ord('i'):
            self.ui_settings['show_detailed_info'] = not self.ui_settings['show_detailed_info']
            print(f"📋 Detailed info: {'ON' if self.ui_settings['show_detailed_info'] else 'OFF'}")
            
        elif key == ord('m'):
            self.ui_settings['show_minimap'] = not self.ui_settings['show_minimap']
            print(f"🗺️ Mini-map: {'ON' if self.ui_settings['show_minimap'] else 'OFF'}")
            
        elif key == ord('c'):
            self.ui_settings['show_compass'] = not self.ui_settings['show_compass']
            print(f"🧭 Compass: {'ON' if self.ui_settings['show_compass'] else 'OFF'}")
    
    def run(self):
        """Main demo loop"""
        print("🚀 Starting Furnished House Robot Demo...")
        
        try:
            sim_settings = self.setup_simulator()
            
            print("\n🎮 Interactive Controls:")
            print("   Movement: W/A/S/D (forward/left/back/right), Q/E (strafe)")
            print("   Look: Z (look up), X (look down)")
            print("   UI Toggles: T (radar), I (details), M (map), C (compass)")
            print("   Other: R (reset), ESC (exit)")
            print("🎯 Goal: Navigate and find the Stretch robot in the house!\n")
            
            # Create window
            cv2.namedWindow('Interactive Robotics Environment', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Interactive Robotics Environment', sim_settings["width"], sim_settings["height"])
            
            last_position = self.get_current_position().copy()
            
            while True:
                # Get observations
                observations = self.sim.get_sensor_observations()
                rgb = observations["color_sensor"]
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                
                # Draw UI overlay
                self.draw_ui_overlay(rgb_bgr)
                
                # Display frame
                cv2.imshow('Interactive Robotics Environment', rgb_bgr)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                
                # Movement
                if self.handle_movement(key):
                    self.stats['steps'] += 1
                    
                    # Check for discoveries
                    new_discoveries = self.update_robot_discovery()
                    for robot in new_discoveries:
                        print(f"🎉 DISCOVERED: {robot['name']}!")
                        print(f"   🏭 {robot['manufacturer']} {robot['type']}")
                        print(f"   📝 {robot['description']}")
                    
                    # Position logging
                    current_pos = self.get_current_position()
                    if np.linalg.norm(current_pos - last_position) > 0.8:
                        print(f"📍 Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})")
                        last_position = current_pos.copy()
                        
                        # Achievement check
                        if len(self.stats['discovered_robots']) == len(self.robots):
                            print("🏆 ACHIEVEMENT UNLOCKED: Master Robot Explorer!")
                
                # UI toggles
                self.handle_ui_toggles(key)
                
                # Exit
                if key == 27:  # ESC
                    break
                
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏹️ Demo interrupted by user")
        
        finally:
            cv2.destroyAllWindows()
            if self.sim:
                self.sim.close()
        
        # Final stats
        runtime = time.time() - self.stats['start_time']
        discovery_rate = len(self.stats['discovered_robots']) / len(self.robots) * 100
        
        print(f"\n🏁 Demo Complete!")
        print(f"📊 Final Statistics:")
        print(f"   • Runtime: {runtime:.1f} seconds")
        print(f"   • Steps taken: {self.stats['steps']}")
        print(f"   • Robots discovered: {len(self.stats['discovered_robots'])}/{len(self.robots)} ({discovery_rate:.1f}%)")
        
        if discovery_rate == 100:
            print("🏆 Perfect Score - All robots discovered!")
        elif discovery_rate >= 80:
            print("🥈 Excellent exploration!")
        elif discovery_rate >= 50:
            print("🥉 Good job exploring!")
        else:
            print("🎯 Keep exploring to find more robots!")
        
        return True

def main():
    """Run the furnished house robot demo"""
    demo = InteractiveRoboticsDemo()
    try:
        success = demo.run()
        if success:
            print("✅ Furnished house robot demo completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure you're in the home-robot conda environment")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()