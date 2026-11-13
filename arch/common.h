// Shared by the per-arch warp-specialization examples: PTX helpers valid on sm_90+ and a host driver.
// Every kernel computes C[M,N] (fp32) = A[M,K] (fp16) * B[N,K]^T (fp16) with one CTA per 128x128 output tile.
#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

constexpr int BM = 128, BN = 128, BK = 64;  // BK halfs = 128B = one swizzle row
constexpr int TILE_A_BYTES = BM * BK * 2, TILE_B_BYTES = BN * BK * 2;
constexpr int STAGE_BYTES = TILE_A_BYTES + TILE_B_BYTES;  // 32KB per pipeline stage

// ------------------------------ device: mbarrier + TMA ------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// TMA / wgmma / tcgen05 swizzle patterns assume 1024B-aligned tiles
__device__ __forceinline__ uint32_t aligned_smem_base(uint8_t* raw) { return (smem_u32(raw) + 1023) & ~1023u; }

__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_init_fence() {
  asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n"
      ".reg .pred P1;\n"
      "LAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\n"
      "bra LAB_WAIT;\n"
      "DONE:\n"
      "}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
// One thread asks the TMA engine to copy a [rows x BK] box at (x=k, y=row) into smem; `bar` gets the byte count.
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tmap, int x, int y, uint32_t bar) {
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}

// ---------------------------------- host driver ----------------------------------
#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

// K-major tile [box_rows x BK] with 128B swizzle: the layout mma/wgmma/tcgen05 all read from
static CUtensorMap make_tmap(const half* ptr, int rows, int cols, int box_rows) {
  CUtensorMap m;
  cuuint64_t gdim[2] = {(cuuint64_t)cols, (cuuint64_t)rows};
  cuuint64_t gstride[1] = {(cuuint64_t)cols * sizeof(half)};
  cuuint32_t box[2] = {BK, (cuuint32_t)box_rows};
  cuuint32_t estride[2] = {1, 1};
  CUresult r = cuTensorMapEncodeTiled(&m, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*)ptr, gdim, gstride, box,
                                      estride, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    printf("cuTensorMapEncodeTiled failed: %d\n", (int)r);
    exit(1);
  }
  return m;
}

// launch(grid, tmA, tmB, C, M, N, K) must start the kernel; `kernel` is only used to probe loadability.
template <typename Launch>
int run_example(const char* built_for, const void* kernel, int smem_bytes, int argc, char** argv, Launch launch) {
  int S = argc > 1 ? atoi(argv[1]) : 4096;
  if (S <= 0 || S % 128) {
    printf("size must be a positive multiple of 128\n");
    return 1;
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("built for %s | running on %s (sm_%d%d)\n", built_for, prop.name, prop.major, prop.minor);
  cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  if (e != cudaSuccess) {
    printf("kernel cannot be loaded on this GPU (%s): compile-only here\n", cudaGetErrorString(e));
    return 0;
  }

  int M = S, N = S, K = S;
  size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
  std::vector<half> hA(nA), hB(nB);
  srand(42);
  for (auto& x : hA) x = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);
  for (auto& x : hB) x = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);
  half *dA, *dB;
  float* dC;
  CK(cudaMalloc(&dA, nA * sizeof(half)));
  CK(cudaMalloc(&dB, nB * sizeof(half)));
  CK(cudaMalloc(&dC, nC * sizeof(float)));
  CK(cudaMemcpy(dA, hA.data(), nA * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), nB * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, nC * sizeof(float)));
  CUtensorMap tmA = make_tmap(dA, M, K, BM), tmB = make_tmap(dB, N, K, BN);
  dim3 grid(N / BN, M / BM);

  const int iters = 10;
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; ++i) launch(grid, tmA, tmB, dC, M, N, K);
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) launch(grid, tmA, tmB, dC, M, N, K);
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaGetLastError());
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  ms /= iters;

  std::vector<float> out(nC);
  CK(cudaMemcpy(out.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
  double err = 0;
  for (int t = 0; t < 256; ++t) {
    int i = rand() % M, j = rand() % N;
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)__half2float(hA[(size_t)i * K + k]) * (double)__half2float(hB[(size_t)j * K + k]);
    err = fmax(err, fabs(s - out[(size_t)i * N + j]) / (fabs(s) + 1.0));
  }
  printf("%dx%dx%d  %.3f ms  %.1f TFLOPS  max_rel_err(256 samples vs CPU)=%.2e  %s\n", M, N, K, ms,
         2.0 * M * N * K / ms / 1e9, err, err < 1e-3 ? "PASS" : "FAIL");
  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  return err < 1e-3 ? 0 : 1;
}
