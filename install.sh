#!/bin/bash
#
# Interactive Robotics - Automated Installation Script
# This script installs all required dependencies for the Interactive Robotics simulation environment
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Print formatted messages
print_header() {
    echo -e "\n${CYAN}${BOLD}============================================================${NC}"
    echo -e "${CYAN}${BOLD}  $1${NC}"
    echo -e "${CYAN}${BOLD}============================================================${NC}\n"
}

print_step() {
    echo -e "${BLUE}${BOLD}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

print_header "Interactive Robotics Installation"

# Check if conda is installed
print_step "Checking for conda installation..."
if ! command -v conda &> /dev/null; then
    print_error "Conda is not installed. Please install Miniconda or Anaconda first."
    echo "Visit: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi
print_success "Conda found: $(conda --version)"

# Set HOME_ROBOT_ROOT
export HOME_ROBOT_ROOT="$SCRIPT_DIR/home-robot"
echo -e "${YELLOW}HOME_ROBOT_ROOT set to: $HOME_ROBOT_ROOT${NC}"

# Check if home-robot submodule exists and is populated
print_step "Checking home-robot submodule..."
if [ ! -d "$HOME_ROBOT_ROOT" ] || [ -z "$(ls -A $HOME_ROBOT_ROOT)" ]; then
    print_warning "home-robot submodule not found or empty. Initializing..."
    git submodule update --init --recursive
    print_success "Submodules initialized"
else
    print_success "home-robot submodule found"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if environment exists
print_step "Checking for home-robot conda environment..."
if conda env list | grep -q "^home-robot "; then
    read -p "Environment 'home-robot' already exists. Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda env remove -n home-robot -y
        print_step "Creating fresh conda environment 'home-robot' with Python 3.9..."
        conda create -n home-robot python=3.9 cmake -c conda-forge -y
        print_success "Conda environment created"
    else
        print_warning "Continuing with existing environment..."
    fi
else
    print_step "Creating conda environment 'home-robot' with Python 3.9..."
    conda create -n home-robot python=3.9 cmake -c conda-forge -y
    print_success "Conda environment created"
fi

# Activate environment
print_step "Activating home-robot environment..."
conda activate home-robot
print_success "Environment activated"

# Install system dependencies
print_header "Installing System Dependencies"
print_step "Installing OpenGL and graphics libraries via conda..."
conda install -c conda-forge libglu libopengl libtiff -y
print_success "Graphics libraries installed"

# Fix libtiff compatibility (habitat-sim needs libtiff.so.5 but conda has 6)
print_step "Creating libtiff compatibility symlink..."
if [ ! -f "$CONDA_PREFIX/lib/libtiff.so.5" ]; then
    ln -sf "$CONDA_PREFIX/lib/libtiff.so.6" "$CONDA_PREFIX/lib/libtiff.so.5"
    print_success "libtiff.so.5 symlink created"
else
    print_success "libtiff.so.5 already exists"
fi

# Install Habitat-Sim with Bullet Physics (this will install numpy 1.26.4)
print_header "Installing Habitat-Sim with Bullet Physics"
print_step "This is critical for robot URDF loading and visualization..."
print_step "Note: This will install numpy 1.26.4 (required by habitat-sim)"
conda install habitat-sim withbullet -c aihabitat -c conda-forge -y
print_success "Habitat-Sim installed with Bullet Physics support"

# Verify Bullet Physics
print_step "Verifying Bullet Physics support..."
if python -c "import habitat_sim; print(f'Bullet Physics: {habitat_sim.built_with_bullet}'); assert habitat_sim.built_with_bullet, 'Bullet not enabled'"; then
    print_success "Bullet Physics is enabled"
else
    print_error "Bullet Physics verification failed"
    echo "Attempting to diagnose the issue..."
    python -c "import habitat_sim; print('habitat_sim version:', habitat_sim.__version__); print('Bullet enabled:', habitat_sim.built_with_bullet)" || echo "Failed to import habitat_sim"
    exit 1
fi

# Install Habitat-Lab
print_header "Installing Habitat-Lab"
print_step "Installing Habitat-Lab framework..."
if [ -d "$HOME_ROBOT_ROOT/src/third_party/habitat-lab/habitat-lab" ]; then
    pip install -e "$HOME_ROBOT_ROOT/src/third_party/habitat-lab/habitat-lab"
    print_success "Habitat-Lab installed"
else
    print_warning "Habitat-Lab submodule not found, attempting pip install..."
    pip install habitat-lab==0.2.5
fi

# Install Habitat-Baselines (optional but recommended)
if [ -d "$HOME_ROBOT_ROOT/src/third_party/habitat-lab/habitat-baselines" ]; then
    print_step "Installing Habitat-Baselines..."
    pip install -e "$HOME_ROBOT_ROOT/src/third_party/habitat-lab/habitat-baselines"
    print_success "Habitat-Baselines installed"
fi

# Apply compatibility fixes to home-robot setup.py BEFORE installing
print_header "Applying Compatibility Fixes to Home-Robot"
print_step "Updating setup.py dependencies..."
if [ -f "$HOME_ROBOT_ROOT/src/home_robot/setup.py" ]; then
    # Fix sophuspy version
    sed -i 's/"sophuspy==0\.0\.8"/"sophuspy==1.2.0"/g' "$HOME_ROBOT_ROOT/src/home_robot/setup.py"
    # Fix numpy version constraint to allow 1.26.4
    sed -i 's/"numpy<1\.24"/"numpy>=1.23,<2.0"/g' "$HOME_ROBOT_ROOT/src/home_robot/setup.py"
    # Fix pillow version to match habitat-sim
    sed -i 's/"pillow==10\.3\.0"/"pillow==10.4.0"/g' "$HOME_ROBOT_ROOT/src/home_robot/setup.py"
    print_success "setup.py dependencies updated"
fi

print_step "Fixing import statements in home_robot code..."
if [ -f "$HOME_ROBOT_ROOT/src/home_robot/home_robot/core/state.py" ]; then
    sed -i 's/import sophus as sp/import sophuspy as sp/g' "$HOME_ROBOT_ROOT/src/home_robot/home_robot/core/state.py"
    print_success "state.py import fixed"
fi

if [ -f "$HOME_ROBOT_ROOT/src/home_robot/home_robot/utils/geometry/_base.py" ]; then
    # Replace the try-except import block with direct import
    sed -i '/try:/,/import sophuspy as sp/c\import sophuspy as sp' "$HOME_ROBOT_ROOT/src/home_robot/home_robot/utils/geometry/_base.py"
    print_success "_base.py import fixed"
fi

# Install Home-Robot packages
print_header "Installing Home-Robot Packages"

if [ -d "$HOME_ROBOT_ROOT/src/home_robot" ]; then
    print_step "Installing home_robot core package..."
    pip install -e "$HOME_ROBOT_ROOT/src/home_robot"
    print_success "home_robot installed"
else
    print_warning "home_robot package not found at $HOME_ROBOT_ROOT/src/home_robot"
fi

if [ -d "$HOME_ROBOT_ROOT/src/home_robot_hw" ]; then
    print_step "Installing home_robot_hw package..."
    pip install -e "$HOME_ROBOT_ROOT/src/home_robot_hw"
    print_success "home_robot_hw installed"
fi

if [ -d "$HOME_ROBOT_ROOT/src/home_robot_sim" ]; then
    print_step "Installing home_robot_sim package..."
    pip install -e "$HOME_ROBOT_ROOT/src/home_robot_sim"
    print_success "home_robot_sim installed"
fi

# Install additional dependencies
print_header "Installing Additional Dependencies"
print_step "Installing opencv, numpy, matplotlib..."
pip install opencv-python numpy matplotlib
print_success "Core dependencies installed"

# Fix sophuspy installation issue - needs newer cmake
print_step "Upgrading cmake for sophuspy compilation..."
conda install cmake -c conda-forge -y
print_success "CMake upgraded"

print_step "Installing sophuspy (this may take a minute to compile)..."
pip install sophuspy==1.2.0
print_success "sophuspy installed"

# Ask about dataset download
print_header "Dataset Download"
echo -e "${YELLOW}${BOLD}Dataset Information:${NC}"
echo "  • Complete dataset: ~22GB (HSSD scenes, objects, robot models)"
echo "  • Minimal dataset: ~107MB (basic test scenes only)"
echo ""
read -p "Do you want to download the complete dataset now? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Downloading complete datasets (this may take a while)..."
    cd "$HOME_ROBOT_ROOT"
    if [ -f "download_data.sh" ]; then
        chmod +x download_data.sh
        ./download_data.sh --yes
        print_success "Datasets downloaded"
    else
        print_error "download_data.sh not found"
    fi
    cd "$SCRIPT_DIR"
else
    print_warning "Skipping dataset download. You can run it later:"
    echo "  cd home-robot && ./download_data.sh --yes"
    echo ""
    echo "Or download minimal test scenes:"
    echo "  python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path $HOME_ROBOT_ROOT/data/"
fi

# Run installation check
print_header "Running Installation Verification"
if [ -f "check_installation.py" ]; then
    python check_installation.py
else
    print_warning "check_installation.py not found, skipping verification"
fi

# Summary
print_header "Installation Complete!"
echo -e "${GREEN}${BOLD}✨ Your Interactive Robotics environment is ready!${NC}\n"
echo "To use the environment:"
echo "  1. Activate: ${CYAN}conda activate home-robot${NC}"
echo "  2. Set path: ${CYAN}export HOME_ROBOT_ROOT=$HOME_ROBOT_ROOT${NC}"
echo "  3. Run demo: ${CYAN}python furnished_house_robot_demo.py${NC}"
echo ""
echo "For future sessions, add this to your ~/.bashrc:"
echo "  ${CYAN}export HOME_ROBOT_ROOT=$HOME_ROBOT_ROOT${NC}"
