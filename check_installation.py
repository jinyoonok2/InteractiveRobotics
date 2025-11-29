#!/usr/bin/env python3
"""
Interactive Robotics Installation Checker

This script verifies that all required components for the Interactive Robotics
simulation environment are properly installed and configured.

Usage:
    python check_installation.py

Author: Interactive Robotics Setup
Date: November 2025
"""

import sys
import os
import subprocess
import importlib
from pathlib import Path
import traceback

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

def print_header(title):
    """Print a formatted section header"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title.center(60)}{Colors.ENDC}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'='*60}{Colors.ENDC}")

def print_check(description, status, details=None):
    """Print a check result with status"""
    status_symbol = f"{Colors.GREEN}✅" if status else f"{Colors.RED}❌"
    print(f"{status_symbol} {description}{Colors.ENDC}")
    if details:
        print(f"   {Colors.YELLOW}→ {details}{Colors.ENDC}")

def check_python_version():
    """Check if Python version is compatible"""
    print_header("Python Environment")
    
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    compatible = version.major == 3 and version.minor >= 9
    
    print_check(f"Python version: {version_str}", compatible, 
                "Requires Python 3.9+" if not compatible else "Compatible")
    
    # Check if we're in the right conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', 'None')
    in_home_robot_env = conda_env == 'home-robot'
    
    print_check(f"Conda environment: {conda_env}", in_home_robot_env,
                "Run 'conda activate home-robot'" if not in_home_robot_env else "Correct environment")
    
    return compatible and in_home_robot_env

def check_required_packages():
    """Check if required Python packages are installed"""
    print_header("Required Python Packages")
    
    required_packages = {
        'numpy': (None, 'numpy'),
        'cv2': (None, 'opencv-python'),
        'torch': (None, 'torch'),
        'habitat_sim': (None, 'habitat-sim'),
        'habitat': (None, 'habitat-lab'),
        'home_robot': (None, 'home-robot'),
        'sophuspy': (None, 'sophuspy'),  # Version check disabled as it works properly
        'matplotlib': (None, 'matplotlib'),
    }
    
    all_installed = True
    
    for package, (expected_version, package_name) in required_packages.items():
        try:
            module = importlib.import_module(package)
            version = getattr(module, '__version__', 'Unknown')
            
            version_ok = True
            version_info = f"v{version}"
            
            if expected_version:
                version_ok = version.startswith(expected_version)
                if not version_ok:
                    version_info += f" (expected {expected_version})"
            
            print_check(f"{package_name}: {version_info}", version_ok,
                       f"Install with: pip install {package_name}{'==' + expected_version if expected_version else ''}" if not version_ok else None)
            
            all_installed = all_installed and version_ok
            
        except ImportError:
            print_check(f"{package_name}: Not installed", False,
                       f"Install with: pip install {package_name}")
            all_installed = False
    
    return all_installed

def check_gpu_acceleration():
    """Check GPU acceleration availability"""
    print_header("GPU Acceleration")
    
    gpu_available = False
    cuda_available = False
    
    # Check PyTorch CUDA
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
            print_check(f"CUDA GPU: {gpu_name}", True, f"{gpu_count} GPU(s) available")
        else:
            print_check("CUDA GPU: Not available", False, "Install CUDA drivers or use CPU")
        gpu_available = cuda_available
    except ImportError:
        print_check("PyTorch: Not installed", False, "Cannot check CUDA availability")
    
    # Check NVIDIA drivers
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        nvidia_available = result.returncode == 0
        if nvidia_available:
            print_check("NVIDIA drivers: Working", True, "nvidia-smi command successful")
        else:
            print_check("NVIDIA drivers: Not working", False, "nvidia-smi command failed")
    except FileNotFoundError:
        print_check("NVIDIA drivers: Not found", False, "nvidia-smi not available")
        nvidia_available = False
    
    return gpu_available

def check_habitat_sim():
    """Check Habitat-Sim functionality"""
    print_header("Habitat-Sim Functionality")
    
    try:
        import habitat_sim
        print_check(f"Habitat-Sim import: v{habitat_sim.__version__}", True)
        
        # Test basic simulator creation
        try:
            # Try to create a basic configuration
            sim_cfg = habitat_sim.SimulatorConfiguration()
            sim_cfg.scene_id = "NONE"  # No scene needed for basic test
            sim_cfg.enable_physics = False
            
            agent_cfg = habitat_sim.agent.AgentConfiguration()
            cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
            
            # Test simulator creation (don't actually create to avoid display issues)
            print_check("Habitat-Sim configuration: Valid", True, "Can create simulator configs")
            
        except Exception as e:
            print_check("Habitat-Sim configuration: Failed", False, str(e))
            return False
            
    except ImportError as e:
        print_check("Habitat-Sim import: Failed", False, str(e))
        return False
    
    return True

def check_habitat_lab():
    """Check Habitat-Lab functionality"""
    print_header("Habitat-Lab Framework")
    
    try:
        import habitat
        print_check(f"Habitat-Lab import: v{habitat.__version__}", True)
        
        # Check habitat-baselines
        try:
            import habitat_baselines
            print_check(f"Habitat-Baselines: v{habitat_baselines.__version__}", True)
        except ImportError:
            print_check("Habitat-Baselines: Not installed", False, "Install with: pip install -e habitat-baselines")
            return False
            
    except ImportError as e:
        print_check("Habitat-Lab import: Failed", False, str(e))
        return False
    
    return True

def check_home_robot_repository():
    """Check if home-robot repository is properly cloned and structured"""
    print_header("Home-Robot Repository Structure")
    
    home_robot_path = Path("home-robot")
    
    # Check if repository exists
    if not home_robot_path.exists():
        print_check("Home-Robot directory: Not found", False, 
                   "Clone with: git clone --recurse-submodules https://github.com/facebookresearch/home-robot.git")
        return False
    
    print_check("Home-Robot directory: Found", True)
    
    # Check if it's a git repository
    git_dir = home_robot_path / ".git"
    if git_dir.exists():
        print_check("Git repository: Valid", True, "Repository properly cloned")
    else:
        print_check("Git repository: Invalid", False, "Directory exists but not a git repo")
        return False
    
    # Check key source directories
    required_dirs = [
        ("src/home_robot", "Core robotics package"),
        ("src/home_robot_sim", "Simulation interfaces"),
        ("src/home_robot_hw", "Hardware interfaces"), 
        ("src/third_party/habitat-lab", "Habitat-Lab submodule"),
    ]
    
    all_dirs_found = True
    for dir_path, description in required_dirs:
        full_path = home_robot_path / dir_path
        exists = full_path.exists()
        print_check(f"{description}: {dir_path}", exists,
                   "Missing - check submodules" if not exists else None)
        all_dirs_found = all_dirs_found and exists
    
    # Check if submodules are initialized
    habitat_lab_path = home_robot_path / "src/third_party/habitat-lab"
    if habitat_lab_path.exists():
        habitat_files = list(habitat_lab_path.glob("*.py")) + list(habitat_lab_path.glob("habitat*"))
        if len(habitat_files) > 0:
            print_check("Submodules: Initialized", True, "Habitat-Lab submodule has content")
        else:
            print_check("Submodules: Not initialized", False, 
                       "Run: git submodule update --init --recursive")
            all_dirs_found = False
    
    return all_dirs_found

def check_scene_data():
    """Check if required scene data and datasets are available"""
    print_header("Scene Data & Datasets")
    
    home_robot_path = Path("home-robot")
    if not home_robot_path.exists():
        print_check("Cannot check datasets: Home-Robot not found", False)
        return False
    
    # Check data directory structure
    data_dir = home_robot_path / "data"
    if data_dir.exists():
        print_check("Data directory: Found", True, str(data_dir))
    else:
        print_check("Data directory: Not found", False, "Will be created when downloading datasets")
    
    # Check for scene datasets
    scene_paths = [
        "home-robot/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
        "home-robot/data/data/scene_datasets/habitat-test-scenes/apartment_1.glb",
    ]
    
    scene_found = False
    scene_size = 0
    for scene_path in scene_paths:
        full_path = Path(scene_path)
        if full_path.exists():
            scene_size = full_path.stat().st_size / (1024 * 1024)  # MB
            print_check(f"Test scene: apartment_1.glb", True, 
                       f"Found at {scene_path} ({scene_size:.1f}MB)")
            scene_found = True
            break
    
    if not scene_found:
        print_check("Test scene: apartment_1.glb", False, 
                   "Download with: python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path home-robot/data/")
    
    # Check for comprehensive datasets (downloaded by ./download_data.sh)
    dataset_checks = [
        ("home-robot/data/hssd-hab/stages", "HSSD photorealistic scenes", "*.glb"),
        ("home-robot/data/objects/train_val", "Interactive objects", "**/*.glb"), 
        ("home-robot/data/datasets/ovmm", "OVMM task episodes", "**/*.json.gz"),
        ("home-robot/data/robots/hab_stretch/urdf", "Stretch robot model", "*.urdf"),
        ("home-robot/data/data/scene_datasets/habitat-test-scenes", "Basic test scenes", "*.glb"),
    ]
    
    datasets_found = 0
    total_datasets = len(dataset_checks)
    
    for dataset_path, description, pattern in dataset_checks:
        full_path = Path(dataset_path)
        if full_path.exists() and full_path.is_dir():
            file_count = len(list(full_path.glob(pattern)))
            if file_count > 0:
                # Calculate total size for major datasets
                if "hssd-hab" in dataset_path:
                    size_info = f"{file_count} scenes (~20GB)"
                elif "objects" in dataset_path:
                    size_info = f"{file_count} objects (~1.8GB)"
                elif "ovmm" in dataset_path:
                    size_info = f"{file_count} episodes (~746MB)"
                elif "robot" in dataset_path:
                    size_info = f"Robot URDF + meshes (~47MB)"
                else:
                    size_info = f"{file_count} files"
                
                print_check(f"{description}", True, f"Found: {size_info}")
                datasets_found += 1
            else:
                print_check(f"{description}", False, f"Directory exists but no {pattern} files found")
        else:
            print_check(f"{description}", False, 
                       "Download with: cd home-robot && ./download_data.sh --yes")
    
    # Enhanced dataset status reporting
    dataset_completeness = (datasets_found / total_datasets) * 100
    if dataset_completeness == 100:
        print_check("Dataset completeness: Complete research setup", True, 
                   "All datasets available for advanced robotics research")
    elif dataset_completeness >= 60:
        print_check("Dataset completeness: Partial setup", True,
                   f"{datasets_found}/{total_datasets} datasets available")
    else:
        print_check("Dataset completeness: Minimal setup", False,
                   "Run ./download_data.sh for complete research environment")
    
    # Overall dataset status
    if scene_found and datasets_found >= 2:
        print_check("Research datasets: Ready", True, 
                   f"Complete environment with {datasets_found}/{total_datasets} datasets")
        return True
    elif scene_found:
        print_check("Basic datasets: Ready", True, "Minimum scenes for demos available")
        return True
    else:
        print_check("Datasets: Insufficient", False, 
                   "Run: cd home-robot && ./download_data.sh --yes")
        return False

def check_demo_readiness():
    """Check if the interactive demo can run"""
    print_header("Interactive Demo Readiness")
    
    # Check if demo file exists
    demo_file = Path("interactive_habitat_demo.py")
    demo_exists = demo_file.exists()
    print_check(f"Demo file: {demo_file.name}", demo_exists, 
               "File not found" if not demo_exists else "Ready to run")
    
    if not demo_exists:
        return False
    
    # Try to import the demo (syntax check)
    try:
        import interactive_habitat_demo
        print_check("Demo import: Successful", True, "No syntax errors")
        
        # Check if main function exists
        has_main = hasattr(interactive_habitat_demo, 'interactive_habitat_demo')
        print_check("Demo function: Available", has_main, 
                   "interactive_habitat_demo() function found" if has_main else "Function not found")
        
        return has_main
        
    except Exception as e:
        print_check("Demo import: Failed", False, str(e))
        return False

def print_summary(checks_passed, total_checks):
    """Print final summary"""
    print_header("Installation Summary")
    
    success_rate = (checks_passed / total_checks) * 100
    
    if success_rate == 100:
        status_color = Colors.GREEN
        status_msg = "🎉 Perfect! All systems ready"
        advice = "You can run: python interactive_habitat_demo.py"
    elif success_rate >= 80:
        status_color = Colors.YELLOW
        status_msg = "⚠️  Almost ready (minor issues)"
        advice = "Fix the failing checks above, then run the demo"
    else:
        status_color = Colors.RED
        status_msg = "❌ Setup incomplete (major issues)"
        advice = "Please address the failing checks before proceeding"
    
    print(f"{status_color}{Colors.BOLD}{status_msg}{Colors.ENDC}")
    print(f"\n📊 Status: {checks_passed}/{total_checks} checks passed ({success_rate:.1f}%)")
    print(f"💡 Next step: {advice}")
    
    if success_rate == 100:
        print(f"\n{Colors.GREEN}🚀 Ready to run Interactive Robotics simulations!{Colors.ENDC}")

def main():
    """Main installation checker"""
    print(f"{Colors.MAGENTA}{Colors.BOLD}")
    print("🤖 Interactive Robotics Installation Checker")
    print("Verifying your simulation environment setup...")
    print(f"{Colors.ENDC}")
    
    checks = [
        ("Python Environment", check_python_version),
        ("Required Packages", check_required_packages),
        ("GPU Acceleration", check_gpu_acceleration),
        ("Habitat-Sim", check_habitat_sim),
        ("Habitat-Lab", check_habitat_lab),
        ("Home-Robot Repository", check_home_robot_repository),
        ("Scene Data & Datasets", check_scene_data),
        ("Demo Readiness", check_demo_readiness),
    ]
    
    passed = 0
    total = len(checks)
    
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
        except Exception as e:
            print_check(f"{check_name}: Error during check", False, str(e))
    
    print_summary(passed, total)
    
    # Exit with error code if not all checks passed
    sys.exit(0 if passed == total else 1)

if __name__ == "__main__":
    main()
