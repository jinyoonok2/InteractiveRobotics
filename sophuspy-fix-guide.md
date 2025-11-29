# SophysPy Compatibility Fix for Home-Robot

## Problem Summary

The home-robot package was originally configured to use `sophuspy==0.0.8` which uses the import syntax `import sophus as sp`. However, newer versions of sophuspy (1.2.0+) changed the import to `import sophuspy as sp`, causing import errors.

## The Error
```
ModuleNotFoundError: No module named 'sophus'
Import "sophuspy" could not be resolved
```

## Root Causes
1. **Import naming incompatibility** between sophuspy versions:
   - `sophuspy==0.0.8`: Uses `import sophus as sp`
   - `sophuspy==1.2.0`: Uses `import sophuspy as sp`

2. **VS Code Python environment mismatch**: 
   - VS Code was using system Python (`/bin/python3`) 
   - But sophuspy was installed in conda environment (`/home/jinyoonok/miniconda3/envs/home-robot/bin/python`)

## Complete Solution

### Step 1: Update Python Environment in VS Code
```bash
# Switch VS Code to use the correct conda environment
# This was done via Pylance tools, but can also be done via:
# - Ctrl+Shift+P → "Python: Select Interpreter"
# - Choose: /home/jinyoonok/miniconda3/envs/home-robot/bin/python
```

### Step 2: Update setup.py Dependency
File: `src/home_robot/setup.py`
```python
# Change from:
"sophuspy==0.0.8",
# To:
"sophuspy==1.2.0",
```

### Step 3: Standardize Import Pattern
File: `src/home_robot/home_robot/utils/geometry/_base.py`
```python
# Change from compatibility layer:
try:
    import sophus as sp  # sophuspy 0.0.8 style
except ImportError:
    import sophuspy as sp  # sophuspy 1.2.0 style

# To direct import:
import sophuspy as sp
```

File: `src/home_robot/home_robot/core/state.py`
```python
# Change from:
import sophus as sp
# To:
import sophuspy as sp
```

### Step 4: Reinstall Package
```bash
cd /home/jinyoonok/Projects/InteractiveRobotics/home-robot/src/home_robot
pip install -e .
```

## Verification Commands

### Check sophuspy version:
```bash
pip show sophuspy
pip list | grep sophuspy
```

### Test imports work:
```bash
python -c "from home_robot.utils.geometry import xyt_global_to_base; import numpy as np; result = xyt_global_to_base(np.array([1,2,0.5]), np.array([0,0,0])); print('✅ Everything working! Result:', result)"
```

### Test in notebook:
```python
import sophuspy as sp
from home_robot.control.traj_following_controller import TrajFollower
print("✅ All imports successful!")
```

## Files Modified
1. `/src/home_robot/setup.py` - Updated dependency version
2. `/src/home_robot/home_robot/core/state.py` - Updated import
3. `/src/home_robot/home_robot/utils/geometry/_base.py` - Simplified import

## Key Lessons
- **Version consistency**: Keep sophuspy version consistent between setup.py and actual usage
- **Environment management**: Ensure VS Code uses the same Python environment where packages are installed  
- **Import standardization**: Use the newer `sophuspy` import pattern for forward compatibility
- **Function names**: Kept existing function names like `xyt2sophus()` for API compatibility

## Final Result
- ✅ sophuspy 1.2.0 working correctly
- ✅ All home-robot imports functional
- ✅ Trajectory follower example working
- ✅ VS Code import resolution fixed
- ✅ Setup.py installation working

## Date Fixed
November 28, 2025