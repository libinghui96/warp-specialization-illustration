#!/bin/bash
# Build all examples and dump SASS. Usage: bash build.sh   (CUDA_HOME defaults to /usr/local/cuda, needs CUDA >= 12.8)
set -e
cd "$(dirname "$0")"
NVCC=${CUDA_HOME:-/usr/local/cuda}/bin/nvcc
CUOBJDUMP=${CUDA_HOME:-/usr/local/cuda}/bin/cuobjdump

# section 1 comparison; setmaxnreg needs the arch-specific target sm_120a
$NVCC -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a ws_gemm.cu -o ws_gemm -lcublas -lcuda

# same 128x128x64 WS GEMM for three targets; only the binary matching the local GPU can run
for t in "120a sm120_5080" "90a sm90_h100" "100a sm100_b200"; do
  set -- $t
  $NVCC -O3 -std=c++17 -gencode arch=compute_$1,code=sm_$1 arch/$2.cu -o arch/$2 -lcuda
  $CUOBJDUMP -sass arch/$2 > arch/$2.sass
done
echo "built: ws_gemm arch/sm120_5080 arch/sm90_h100 arch/sm100_b200"
