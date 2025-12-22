#!/bin/bash
#
# Interactive Robotics - Automated Installation Script
# This script installs Habitat-Sim and Habitat-Lab for interactive robot simulation
# Note: This project uses only data from home-robot, not the library itself
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

# Set DATA_PATH to project root data folder
export DATA_PATH="$SCRIPT_DIR/data"
echo -e "${YELLOW}DATA_PATH set to: $DATA_PATH${NC}"

# Check if data folder exists
print_step "Checking data folder..."
if [ ! -d "$DATA_PATH" ]; then
    print_warning "Data folder not found. Creating directory structure..."
    mkdir -p "$DATA_PATH"/{scenes,objects,robots,humanoids}
    print_success "Data directories created"
else
    print_success "Data folder found"
fi

# Initialize conda for bash
eval "$(conda shell.bash hook)"

# Check if environment exists
print_step "Checking for interactive-robotics conda environment..."
if conda env list | grep -q "^interactive-robotics "; then
    read -p "Environment 'interactive-robotics' already exists. Remove and recreate? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Removing existing environment..."
        conda env remove -n interactive-robotics -y
        print_step "Creating fresh conda environment 'interactive-robotics' with Python 3.9..."
        conda create -n interactive-robotics python=3.9 cmake -c conda-forge -y
        print_success "Conda environment created"
    else
        print_warning "Continuing with existing environment..."
    fi
else
    print_step "Creating conda environment 'interactive-robotics' with Python 3.9..."
    conda create -n interactive-robotics python=3.9 cmake -c conda-forge -y
    print_success "Conda environment created"
fi

# Activate environment
print_step "Activating interactive-robotics environment..."
conda activate interactive-robotics
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
print_step "Installing Habitat-Lab framework (latest version)..."
pip install habitat-lab
print_success "Habitat-Lab installed"

# Install additional required packages
print_header "Installing Additional Dependencies"
print_step "Installing opencv, numpy, matplotlib..."
pip install opencv-python numpy matplotlib
print_success "Core dependencies installed"

# Ask about dataset download
print_header "Dataset Information"
echo -e "${YELLOW}${BOLD}Project Data Structure:${NC}"
echo "  • Data folder location: $DATA_PATH"
echo "  • HSSD scenes: $DATA_PATH/scenes/hssd-hab/"
echo "  • Robot models: $DATA_PATH/robots/"
echo "  • Humanoid models: $DATA_PATH/humanoids/"
echo "  • Objects: $DATA_PATH/objects/"
echo ""
echo -e "${CYAN}${BOLD}Note:${NC} Datasets will be downloaded directly to $DATA_PATH"
echo ""

# Check if home-robot submodule exists for downloading
if [ -d "$SCRIPT_DIR/home-robot" ]; then
    read -p "Do you want to download datasets now? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_step "Downloading datasets using home-robot download script..."
        
        # Temporarily change to home-robot directory
        cd "$SCRIPT_DIR/home-robot"
        
        if [ -f "download_data.sh" ]; then
            # Make script executable
            chmod +x download_data.sh
            
            # Run download script - it will download to home-robot/data by default
            print_warning "This will download ~22GB of data. This may take a while..."
            ./download_data.sh --yes
            
            # Move downloaded data to project root data folder
            print_step "Moving downloaded data to project data folder..."
            
            if [ -d "data/scenes" ] && [ ! -d "$DATA_PATH/scenes" ]; then
                mv data/scenes "$DATA_PATH/"
                print_success "Scenes moved to $DATA_PATH/scenes/"
            fi
            
            if [ -d "data/robots" ] && [ ! -d "$DATA_PATH/robots" ]; then
                mv data/robots "$DATA_PATH/"
                print_success "Robot models moved to $DATA_PATH/robots/"
            fi
            
            if [ -d "data/objects" ] && [ ! -d "$DATA_PATH/objects" ]; then
                mv data/objects "$DATA_PATH/"
                print_success "Objects moved to $DATA_PATH/objects/"
            fi
            
            if [ -d "data/datasets" ] && [ ! -d "$DATA_PATH/datasets" ]; then
                mv data/datasets "$DATA_PATH/"
                print_success "Datasets moved to $DATA_PATH/datasets/"
            fi
            
            # Clean up empty data folder in home-robot
            if [ -d "data" ] && [ -z "$(ls -A data)" ]; then
                rmdir data
            fi
            
            print_success "All datasets moved to $DATA_PATH"
        else
            print_error "download_data.sh not found in home-robot directory"
        fi
        
        cd "$SCRIPT_DIR"
    else
        print_warning "Skipping dataset download."
    fi
else
    print_warning "home-robot submodule not found."
    echo "To download datasets manually:"
    echo "  1. git submodule update --init --recursive"
    echo "  2. cd home-robot && ./download_data.sh --yes"
    echo "  3. mv home-robot/data/* $DATA_PATH/"
fi

# Download humanoid models separately (if needed)
print_header "Humanoid Models"
if [ ! -d "$DATA_PATH/humanoids" ] || [ -z "$(ls -A $DATA_PATH/humanoids 2>/dev/null)" ]; then
    echo -e "${YELLOW}Humanoid models not found in $DATA_PATH/humanoids/${NC}"
    echo "You'll need to obtain SMPL-X humanoid models separately."
    echo ""
    echo "If you have humanoid models in home-robot/data/humanoids:"
    echo "  mv home-robot/data/humanoids $DATA_PATH/"
    echo ""
    echo "Or if they're elsewhere, copy them to:"
    echo "  $DATA_PATH/humanoids/"
else
    print_success "Humanoid models found in $DATA_PATH/humanoids/"
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
echo "  1. Activate: ${CYAN}conda activate interactive-robotics${NC}"
echo "  2. Run demos:"
echo "     ${CYAN}python humanoid_exploration_demo.py${NC}  - Humanoid avatar control"
echo "     ${CYAN}python robot_interaction_demo.py${NC}     - Robot manipulation"
echo ""
echo "Data location: ${CYAN}$DATA_PATH${NC}"
echo ""
echo "For future sessions, add this to your ~/.bashrc:"
echo "  ${CYAN}alias activate-robots='conda activate interactive-robotics'${NC}"
