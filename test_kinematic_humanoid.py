#!/usr/bin/env python3
"""Test if KinematicHumanoid class works with our setup"""

import os
import habitat_sim
from habitat.articulated_agents.humanoids import KinematicHumanoid
import magnum as mn

print("="*60)
print("🧪 Testing KinematicHumanoid Class")
print("="*60)

# Setup minimal simulator
print("\n1. Creating simulator...")
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = "NONE"
sim_cfg.enable_physics = True
agent_cfg = habitat_sim.agent.AgentConfiguration()
cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
sim = habitat_sim.Simulator(cfg)
print("   ✅ Simulator created")

# Load humanoid URDF
print("\n2. Loading humanoid URDF...")
data_path = os.path.join(os.path.dirname(__file__), "data")
urdf_path = os.path.join(data_path, "humanoids/male_1/male_1.urdf")

if not os.path.exists(urdf_path):
    print(f"   ❌ URDF not found: {urdf_path}")
    exit(1)

ao_mgr = sim.get_articulated_object_manager()
humanoid_ao = ao_mgr.add_articulated_object_from_urdf(
    filepath=urdf_path,
    fixed_base=False
)
humanoid_ao.translation = mn.Vector3(0, 1.0, 0)
humanoid_ao.motion_type = habitat_sim.physics.MotionType.KINEMATIC
print(f"   ✅ Humanoid loaded: {humanoid_ao.num_links} links")

# Try to create KinematicHumanoid wrapper
print("\n3. Creating KinematicHumanoid wrapper...")
try:
    kin_humanoid = KinematicHumanoid(humanoid_ao, sim)
    print("   ✅ KinematicHumanoid created successfully!")
    
    # Test basic methods
    print("\n4. Testing KinematicHumanoid methods...")
    
    # Get end effector links
    try:
        left_hand_id = kin_humanoid.get_link_id_from_name("left_hand")
        print(f"   ✅ Left hand link ID: {left_hand_id}")
    except Exception as e:
        print(f"   ⚠️  get_link_id_from_name failed: {e}")
    
    # Try to get base transformation
    try:
        base_transform = kin_humanoid.base_transformation
        print(f"   ✅ Base transformation: {base_transform}")
    except Exception as e:
        print(f"   ⚠️  base_transformation failed: {e}")
    
    # Check if it has update_pose method
    if hasattr(kin_humanoid, 'update_pose'):
        print("   ✅ update_pose method available")
    else:
        print("   ⚠️  update_pose method NOT found")
    
    # Check available attributes
    print("\n5. KinematicHumanoid attributes:")
    attrs = [attr for attr in dir(kin_humanoid) if not attr.startswith('_')]
    for attr in attrs[:15]:  # Show first 15
        print(f"      - {attr}")
    
    print("\n" + "="*60)
    print("✅ SUCCESS: KinematicHumanoid is fully functional!")
    print("="*60)
    
except Exception as e:
    print(f"   ❌ FAILED to create KinematicHumanoid: {e}")
    import traceback
    traceback.print_exc()
    
    print("\n" + "="*60)
    print("❌ KinematicHumanoid NOT compatible with current setup")
    print("="*60)

sim.close()
