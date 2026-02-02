#!/bin/bash
# Copyright (c) PyPTO Contributors.
# Build script for pto-testing-framework (builds PyPTO and sets up environment)

set -e  # Exit on error

# Color output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}PTO Testing Framework Build Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Get project root directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"
PYPTO_DIR="${PROJECT_ROOT}/3rdparty/pypto"
SIMPLER_DIR="${PROJECT_ROOT}/3rdparty/simpler"
BUILD_DIR="${PROJECT_ROOT}/build/pypto"

# Detect Python executable
if [ -n "$PYTHON_EXE" ]; then
    # Use environment variable if set
    :
elif command -v python3 &> /dev/null; then
    PYTHON_EXE=$(command -v python3)
else
    echo -e "${RED}Error: python3 not found${NC}"
    exit 1
fi

# Parse command line arguments
BUILD_TYPE="RelWithDebInfo"
CLEAN_BUILD=false
INSTALL_PACKAGE=false
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Build type: Debug, Release, RelWithDebInfo (default: RelWithDebInfo)"
    echo "  -c, --clean            Clean build directory before building"
    echo "  -i, --install          Install pto-test package in editable mode (pip install -e .)"
    echo "  -j, --jobs N           Number of parallel jobs (default: auto-detect)"
    echo "  -h, --help             Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     # Build with default settings"
    echo "  $0 --clean             # Clean build"
    echo "  $0 --type Debug        # Build in Debug mode"
    echo "  $0 --install           # Build and install pto-test package"
    echo ""
    echo "After building, set up the environment:"
    echo "  source ${PROJECT_ROOT}/build/setup_env.sh"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BUILD_TYPE="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -i|--install)
            INSTALL_PACKAGE=true
            shift
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Validate build type
if [[ ! "$BUILD_TYPE" =~ ^(Debug|Release|RelWithDebInfo)$ ]]; then
    echo -e "${RED}Error: Invalid build type '$BUILD_TYPE'${NC}"
    echo "Valid types: Debug, Release, RelWithDebInfo"
    exit 1
fi

# Check submodules
echo -e "${BLUE}Checking submodules...${NC}"
if [ ! -d "$PYPTO_DIR/src" ]; then
    echo -e "${YELLOW}PyPTO submodule not initialized. Initializing...${NC}"
    git -C "$PROJECT_ROOT" submodule update --init --recursive 3rdparty/pypto
fi
if [ ! -d "$SIMPLER_DIR/python" ]; then
    echo -e "${YELLOW}Simpler submodule not initialized. Initializing...${NC}"
    git -C "$PROJECT_ROOT" submodule update --init --recursive 3rdparty/simpler
fi
echo -e "${GREEN}✓ Submodules ready${NC}"

# Detect Python paths
echo -e "${BLUE}Detecting Python installation...${NC}"
PYTHON_INCLUDE_DIR=$($PYTHON_EXE -c "import sysconfig; print(sysconfig.get_path('include'))")

# Detect OS and set appropriate library extension
if [[ "$OSTYPE" == "darwin"* ]]; then
    PYTHON_LIBRARY=$($PYTHON_EXE -c "import sysconfig; import os; libdir = sysconfig.get_config_var('LIBDIR'); version = sysconfig.get_config_var('VERSION'); print(os.path.join(libdir, f'libpython{version}.dylib'))")
else
    # Linux
    PYTHON_LIBRARY=$($PYTHON_EXE -c "import sysconfig; import os; libdir = sysconfig.get_config_var('LIBDIR'); version = sysconfig.get_config_var('VERSION'); print(os.path.join(libdir, f'libpython{version}.so'))")
fi

# Detect nanobind location
echo -e "${BLUE}Detecting nanobind installation...${NC}"
NANOBIND_DIR=$($PYTHON_EXE -c "import nanobind; import os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))" 2>/dev/null || echo "")

if [ -z "$NANOBIND_DIR" ]; then
    echo -e "${YELLOW}Warning: nanobind not found. Installing...${NC}"
    $PYTHON_EXE -m pip install nanobind
    NANOBIND_DIR=$($PYTHON_EXE -c "import nanobind; import os; print(os.path.join(os.path.dirname(nanobind.__file__), 'cmake'))")
fi

echo -e "${GREEN}Configuration:${NC}"
echo "  Project root:      ${PROJECT_ROOT}"
echo "  PyPTO source:      ${PYPTO_DIR}"
echo "  Build directory:   ${BUILD_DIR}"
echo "  Build type:        ${BUILD_TYPE}"
echo "  Clean build:       ${CLEAN_BUILD}"
echo "  Install package:   ${INSTALL_PACKAGE}"
echo "  Parallel jobs:     ${JOBS}"
echo "  Python exe:        ${PYTHON_EXE}"
echo "  Python include:    ${PYTHON_INCLUDE_DIR}"
echo "  Python library:    ${PYTHON_LIBRARY}"
echo "  nanobind_DIR:      ${NANOBIND_DIR}"
echo ""

# Step 1: Clean build directory if requested
if [ "$CLEAN_BUILD" = true ]; then
    echo -e "${BLUE}[1/5] Cleaning build directory...${NC}"
    if [ -d "$BUILD_DIR" ]; then
        rm -rf "$BUILD_DIR"
        echo -e "${GREEN}✓ Build directory cleaned${NC}"
    else
        echo -e "${GREEN}✓ Build directory does not exist, skipping${NC}"
    fi
else
    echo -e "${BLUE}[1/5] Skipping clean (use --clean to clean build)${NC}"
fi

# Step 2: Create build directory
echo -e "${BLUE}[2/5] Creating build directory...${NC}"
mkdir -p "$BUILD_DIR"
echo -e "${GREEN}✓ Build directory ready${NC}"

# Step 3: Configure with CMake
echo -e "${BLUE}[3/5] Configuring PyPTO with CMake...${NC}"

# Get Python prefix (conda environment root)
PYTHON_PREFIX=$($PYTHON_EXE -c "import sys; print(sys.prefix)")

cmake -S "$PYPTO_DIR" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -Dnanobind_DIR="$NANOBIND_DIR" \
    -DPython_EXECUTABLE="$PYTHON_EXE" \
    -DPython_ROOT_DIR="$PYTHON_PREFIX"
echo -e "${GREEN}✓ CMake configuration complete${NC}"

# Step 4: Build
echo -e "${BLUE}[4/5] Building PyPTO...${NC}"
cmake --build "$BUILD_DIR" -j"$JOBS"
echo -e "${GREEN}✓ Build complete${NC}"

# Verify build output
echo ""
echo -e "${BLUE}Verifying build output...${NC}"
SO_FILE=$(find "$PYPTO_DIR/python/pypto" -name "pypto_core*.so" 2>/dev/null | head -n 1)
if [ -n "$SO_FILE" ]; then
    echo -e "${GREEN}✓ Built module found: ${SO_FILE}${NC}"
else
    echo -e "${RED}✗ Built module not found${NC}"
    exit 1
fi

# Step 5: Optional package installation
if [ "$INSTALL_PACKAGE" = true ]; then
    echo -e "${BLUE}[5/5] Installing pto-test package...${NC}"
    $PYTHON_EXE -m pip install -e "$PROJECT_ROOT"
    echo -e "${GREEN}✓ pto-test package installed${NC}"
else
    echo -e "${BLUE}[5/5] Skipping package installation (use --install to install)${NC}"
fi

# Test import
echo ""
echo -e "${BLUE}Testing imports...${NC}"
export PYTHONPATH="${PYPTO_DIR}/python:${SIMPLER_DIR}/python:${PROJECT_ROOT}/src:${PYTHONPATH}"

if $PYTHON_EXE -c "import pypto; print(f'PyPTO version: {pypto.__version__}')"; then
    echo -e "${GREEN}✓ PyPTO successfully importable${NC}"
else
    echo -e "${RED}✗ Failed to import PyPTO${NC}"
    echo -e "${YELLOW}Try: source ${PROJECT_ROOT}/build/setup_env.sh${NC}"
    exit 1
fi

# Generate setup_env.sh for convenience
SETUP_ENV_FILE="${PROJECT_ROOT}/build/setup_env.sh"
cat > "$SETUP_ENV_FILE" << EOF
#!/bin/bash
# Auto-generated environment setup script for pto-testing-framework
# Source this file to set up the environment: source build/setup_env.sh

export PYTHONPATH="${PYPTO_DIR}/python:${SIMPLER_DIR}/python:${PROJECT_ROOT}/src:\${PYTHONPATH}"
export PYPTO_DIR="${PYPTO_DIR}"
export SIMPLER_DIR="${SIMPLER_DIR}"

echo "Environment configured for pto-testing-framework"
echo "  PYPTO_DIR:   \$PYPTO_DIR"
echo "  SIMPLER_DIR: \$SIMPLER_DIR"
EOF
chmod +x "$SETUP_ENV_FILE"

# Auto-execute the setup script
echo ""
echo -e "${BLUE}Executing environment setup...${NC}"
source "$SETUP_ENV_FILE"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${GREEN}Environment has been automatically configured!${NC}"
echo ""
echo -e "${YELLOW}If you open a new terminal, source the environment again:${NC}"
echo ""
echo "  source ${PROJECT_ROOT}/build/setup_env.sh"
echo ""
echo "Then you can run tests:"
echo "  pytest tests/"
echo ""
