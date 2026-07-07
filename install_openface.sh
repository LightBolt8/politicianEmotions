#!/usr/bin/env bash
# Build OpenFace natively on macOS (requires ~5-10 GB free disk space).
set -euo pipefail

OPENFACE_DIR="${OPENFACE_DIR:-$HOME/OpenFace}"

echo "Checking disk space..."
AVAIL_KB=$(df -k / | awk 'NR==2 {print $4}')
if (( AVAIL_KB < 5000000 )); then
  echo "Error: need at least ~5 GB free. Currently: $(( AVAIL_KB / 1024 / 1024 )) GB"
  exit 1
fi

echo "Installing Homebrew dependencies..."
brew install cmake wget boost tbb openblas opencv dlib

if [[ ! -d "$OPENFACE_DIR/.git" ]]; then
  echo "Cloning OpenFace into $OPENFACE_DIR"
  git clone https://github.com/TadasBaltrusaitis/OpenFace.git "$OPENFACE_DIR"
fi

cd "$OPENFACE_DIR"
bash download_models.sh

export OMP_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

OPENBLAS_PREFIX=$(brew --prefix openblas)

rm -rf build
mkdir build && cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D OpenBLAS_INCLUDE_DIR="$OPENBLAS_PREFIX/include" \
  -D OpenBLAS_LIB="$OPENBLAS_PREFIX/lib/libopenblas.dylib" \
  -D Boost_ROOT=/opt/homebrew \
  ..
make -j"$(sysctl -n hw.ncpu)"

echo ""
echo "OpenFace installed at: $OPENFACE_DIR/build/bin/FeatureExtraction"
echo "Run on your exports with:"
echo "  cd $(dirname "$0")"
echo "  .venv/bin/python run_openface.py --openface-dir $OPENFACE_DIR"
