#!/usr/bin/env python3
"""
Interactive Habitat-Sim Demo
Navigate through photorealistic 3D environments with keyboard controls
"""

import habitat_sim
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
from collections import defaultdict

def interactive_habitat_demo():
    """
    Interactive navigation in photorealistic Habitat-Sim environment
    """
    print("🚀 Starting Interactive Habitat-Sim Demo...")
    print("🏠 Loading photorealistic apartment scene...")
    
    # Configure simulator settings
    sim_settings = {
        "scene": "/home/jinyoonok/Projects/InteractiveRobotics/home-robot/data/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        "default_agent": 0,
        "sensor_height": 1.5,
        "width": 800,
        "height": 600,
        "hfov": 90,
    }
    
    # Create configuration
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = sim_settings["scene"]
    sim_cfg.enable_physics = False
    # sim_cfg.allow_sliding = True  # Not needed without physics
    
    # Set up RGB sensor (camera)
    rgb_sensor = habitat_sim.CameraSensorSpec()
    rgb_sensor.uuid = "color_sensor"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [sim_settings["height"], sim_settings["width"]]
    rgb_sensor.position = [0.0, sim_settings["sensor_height"], 0.0]
    
    # Set up depth sensor
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
    sim = habitat_sim.Simulator(cfg)
    
    # Initialize agent
    agent = sim.initialize_agent(sim_settings["default_agent"])
    
    # Set starting position
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array([0.0, 0.0, 0.0])
    agent_state.rotation = np.quaternion(1, 0, 0, 0)
    agent.set_state(agent_state)
    
    print("✅ Photorealistic apartment loaded!")
    print("🎮 Interactive Controls:")
    print("   - W/S: Move forward/backward")
    print("   - A/D: Turn left/right")  
    print("   - Q/E: Strafe left/right")
    print("   - R: Reset to start position")
    print("   - ESC: Exit")
    print("📱 Click on the window and use keys to navigate!")
    
    # Movement parameters
    move_amount = 0.25
    turn_amount = np.pi / 12  # 15 degrees
    
    # Create display window
    cv2.namedWindow('Habitat-Sim Interactive', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Habitat-Sim Interactive', sim_settings["width"], sim_settings["height"])
    
    # Stats tracking
    steps = 0
    last_position = agent_state.position.copy()
    
    try:
        while True:
            # Get current observations
            observations = sim.get_sensor_observations()
            rgb = observations["color_sensor"]
            depth = observations["depth_sensor"]
            
            # Convert RGB from RGB to BGR for OpenCV
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            # Add UI overlay
            current_pos = agent.get_state().position
            
            # Draw position info
            cv2.putText(rgb_bgr, f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rgb_bgr, f"Steps: {steps}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(rgb_bgr, "W/S: Move | A/D: Turn | Q/E: Strafe | R: Reset | ESC: Exit", 
                       (10, rgb_bgr.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display the frame
            cv2.imshow('Habitat-Sim Interactive', rgb_bgr)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            moved = False
            
            if key == ord('w'):  # Move forward
                agent.act("move_forward")
                moved = True
                print("⬆️ Moving forward")
                
            elif key == ord('s'):  # Move backward  
                agent.act("move_backward")
                moved = True
                print("⬇️ Moving backward")
                
            elif key == ord('a'):  # Turn left
                agent.act("turn_left")
                moved = True  
                print("⬅️ Turning left")
                
            elif key == ord('d'):  # Turn right
                agent.act("turn_right")
                moved = True
                print("➡️ Turning right")
                
            elif key == ord('q'):  # Strafe left (custom movement)
                current_state = agent.get_state()
                forward = habitat_sim.utils.quat_to_coeffs(current_state.rotation.inverse())
                # Calculate left direction (90 degrees from forward)
                left_quat = current_state.rotation * habitat_sim.utils.quat_from_angle_axis(np.pi/2, np.array([0, 1, 0]))
                left_direction = habitat_sim.utils.quat_rotate_vector(left_quat, np.array([0, 0, -1]))
                
                new_position = current_state.position + left_direction * move_amount
                agent_state.position = new_position
                agent_state.rotation = current_state.rotation  
                agent.set_state(agent_state)
                moved = True
                print("⬅️ Strafing left")
                
            elif key == ord('e'):  # Strafe right (custom movement)
                current_state = agent.get_state()
                # Calculate right direction (90 degrees from forward)
                right_quat = current_state.rotation * habitat_sim.utils.quat_from_angle_axis(-np.pi/2, np.array([0, 1, 0]))
                right_direction = habitat_sim.utils.quat_rotate_vector(right_quat, np.array([0, 0, -1]))
                
                new_position = current_state.position + right_direction * move_amount  
                agent_state.position = new_position
                agent_state.rotation = current_state.rotation
                agent.set_state(agent_state)
                moved = True
                print("➡️ Strafing right")
                
            elif key == ord('r'):  # Reset position
                agent_state.position = np.array([0.0, 0.0, 0.0])
                agent_state.rotation = np.quaternion(1, 0, 0, 0)
                agent.set_state(agent_state)
                moved = True
                steps = 0
                print("🔄 Reset to start position")
                
            elif key == 27:  # ESC key
                print("👋 Exiting...")
                break
                
            # Update step counter if moved
            if moved:
                steps += 1
                new_position = agent.get_state().position
                distance = np.linalg.norm(new_position - last_position)
                if distance > 0.1:  # Only print if significant movement
                    print(f"📍 New position: ({new_position[0]:.2f}, {new_position[1]:.2f}, {new_position[2]:.2f})")
                    last_position = new_position.copy()
            
            # Small delay to prevent high CPU usage
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("\n⏹️ Demo stopped by user")
    
    finally:
        cv2.destroyAllWindows()
        sim.close()
    
    print("🏁 Interactive Habitat-Sim demo completed!")
    print(f"📊 Total steps taken: {steps}")
    return True

if __name__ == "__main__":
    try:
        success = interactive_habitat_demo()
        if success:
            print("🎉 Interactive photorealistic simulation successful!")
    except Exception as e:
        print(f"❌ Error: {e}")
        print("💡 Make sure habitat-sim and opencv-python are installed")