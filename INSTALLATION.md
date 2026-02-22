# MRCpy Installation Guide

## Quick Start (Recommended)

### Option 1: Install with pip (from source)

```bash
cd main
pip install -e .
```

This installs the core package with all essential dependencies.

### Option 2: Install with PyTorch support

```bash
cd main
pip install -e ".[pytorch]"
```

### Option 3: Install with all optional features

```bash
cd main
pip install -e ".[all]"
```

## Installation Options

The package supports several installation extras:

- `pytorch`: PyTorch MGCE classifier support
- `lmrc`: LMRC classifier (requires pycddlib)
- `solvers`: Commercial solvers (Gurobi, Mosek)
- `all`: All optional dependencies

**Examples:**
```bash
# Just PyTorch
pip install -e ".[pytorch]"

# PyTorch + LMRC
pip install -e ".[pytorch,lmrc]"

# Everything
pip install -e ".[all]"
```

## Alternative: Manual Installation

If you prefer to install dependencies manually:

### Minimal Installation

For core MRCpy functionality:

```bash
pip install -r requirements-minimal.txt
```

### PyTorch MGCE Classifier

For the PyTorch-based MGCE classifier with GPU support:

```bash
pip install -r requirements-pytorch.txt
```

### Full Installation

For all features (may require additional setup):

```bash
pip install -r requirements.txt
pip install -r requirements-optional.txt
```

## Troubleshooting

### pycddlib Installation Issues

`pycddlib` requires compilation and may fail on some systems. If you encounter issues:

1. **macOS**: Install with Homebrew first
   ```bash
   brew install cddlib
   pip install pycddlib
   ```

2. **Linux**: Install development packages
   ```bash
   sudo apt-get install libgmp-dev
   pip install pycddlib
   ```

3. **Skip if not needed**: LMRC is the only classifier that requires pycddlib

### Commercial Solvers (mosek, gurobipy)

These require licenses and should be installed separately:

- **Mosek**: Get academic license at https://www.mosek.com/products/academic-licenses/
- **Gurobi**: Get academic license at https://www.gurobi.com/academia/

### libsvm Issues

If libsvm installation fails, you can use the official package:

```bash
pip install libsvm-official
```

## Verifying Installation

Test your installation:

```python
# Test core functionality
from MRCpy import MRC
print("✓ Core MRCpy installed")

# Test PyTorch MGCE
try:
    from MRCpy.pytorch import mgce_clf
    print("✓ PyTorch MGCE installed")
except ImportError:
    print("✗ PyTorch MGCE not available")

# Test optional features
try:
    from MRCpy import LMRC
    print("✓ LMRC available (pycddlib installed)")
except (ImportError, AttributeError):
    print("✗ LMRC not available (pycddlib not installed)")
```

## Recommended Installation Order

1. **Start minimal**: `pip install -r requirements-minimal.txt`
2. **Add PyTorch** (if needed): `pip install -r requirements-pytorch.txt`
3. **Add optional** (if needed): Install specific packages from requirements-optional.txt

This approach avoids installation failures from optional dependencies.
