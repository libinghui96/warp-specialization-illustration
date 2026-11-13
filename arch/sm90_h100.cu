// H100 (sm_90a): TMA + mbarrier + wgmma + setmaxnreg. The Tensor Core reads A/B straight from smem through
// 64-bit descriptors (no ldmatrix); a warpgroup (4 warps) issues one async MMA and keeps the accumulators in registers.
// build: nvcc -O3 -std=c++17 -gencode arch=compute_90a,code=sm_90a sm90_h100.cu -o sm90_h100 -lcuda
#include "common.h"

constexpr int STAGES = 4;  // H100 has 227KB smem per CTA
constexpr int CONSUMER_WGS = 2;  // each warpgroup owns 64 rows of the 128x128 tile
constexpr int THREADS = (CONSUMER_WGS + 1) * 128;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + 1024;
constexpr int PRODUCER_REGS = 40, CONSUMER_REGS = 232;  // 128*40 + 256*232 <= 65536 registers per CTA

// K-major operand in a 128B-swizzled tile: stride between 8-row groups = 1024B, layout type 1 = SW128
__device__ __forceinline__ uint64_t gmma_desc(uint32_t smem_addr) {
  return uint64_t((smem_addr & 0x3FFFF) >> 4) | (uint64_t(1024 >> 4) << 32) | (1ull << 62);
}

#define D4(i) "+f"(d[i]), "+f"(d[i + 1]), "+f"(d[i + 2]), "+f"(d[i + 3])
#define D16(i) D4(i), D4(i + 4), D4(i + 8), D4(i + 12)
// D[64x128] += A[64x16] * B[16x128]; every thread of the warpgroup holds 64 of the 8192 accumulators
__device__ __forceinline__ void wgmma_m64n128k16(float (&d)[64], uint64_t desc_a, uint64_t desc_b) {
#ifdef __CUDA_ARCH__
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %66, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
      "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
      "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
      "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
      "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},"
      " %64, %65, p, 1, 1, 0, 0;\n"  // scale_d=p, scale_a=1, scale_b=1, A and B both K-major
      "}\n"
      : D16(0), D16(16), D16(32), D16(48)
      : "l"(desc_a), "l"(desc_b), "r"(1));
#endif
}
__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wgmma_commit_group() {
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}
__device__ __forceinline__ void wgmma_wait_group0() {
  asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
}
template <uint32_t R>
__device__ __forceinline__ void reg_dealloc() {
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R));
}
template <uint32_t R>
__device__ __forceinline__ void reg_alloc() {
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R));
}

__global__ void __launch_bounds__(THREADS, 1)
    gemm_ws_sm90(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C, int M,
                 int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wg = warp >> 2;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_u32(&full_bar[s]), 1);
      mbar_init(smem_u32(&empty_bar[s]), CONSUMER_WGS * 4);  // one arrive per consumer warp
    }
    mbar_init_fence();
  }
  __syncthreads();

  if (wg == CONSUMER_WGS) {
    // ---------------- producer warpgroup: gives its registers away, one thread drives TMA ----------------
    reg_dealloc<PRODUCER_REGS>();
    if (tid == CONSUMER_WGS * 128) {
      for (int kt = 0; kt < KT; ++kt) {
        int s = kt % STAGES;
        uint32_t ph = (kt / STAGES) & 1;
        mbar_wait(smem_u32(&empty_bar[s]), ph ^ 1);
        uint32_t sA = base + s * STAGE_BYTES, fb = smem_u32(&full_bar[s]);
        mbar_arrive_expect_tx(fb, STAGE_BYTES);
        tma_load_2d(sA, &tmA, kt * BK, m0, fb);
        tma_load_2d(sA + TILE_A_BYTES, &tmB, kt * BK, n0, fb);
      }
    }
  } else {
    // ---------------- consumer warpgroups: 4 async wgmma per k-tile, no ldmatrix ----------------
    reg_alloc<CONSUMER_REGS>();
    float acc[64] = {};
    for (int kt = 0; kt < KT; ++kt) {
      int s = kt % STAGES;
      mbar_wait(smem_u32(&full_bar[s]), (kt / STAGES) & 1);
      uint32_t sA = base + s * STAGE_BYTES + wg * 64 * 128, sB = base + s * STAGE_BYTES + TILE_A_BYTES;
      wgmma_fence();  // registers written by normal code become visible to the async MMA
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk)
        wgmma_m64n128k16(acc, gmma_desc(sA + kk * 32), gmma_desc(sB + kk * 32));  // k16 fp16 = 32B per step
      wgmma_commit_group();
      wgmma_wait_group0();  // accumulators and smem stage are safe to touch again
      if (lane == 0) mbar_arrive(smem_u32(&empty_bar[s]));
    }
    // per warp the 64xN accumulator layout is mma.m16n8's, repeated N/8 times along N
    int r = m0 + wg * 64 + (warp & 3) * 16 + (lane >> 2);
#pragma unroll
    for (int v = 0; v < BN / 8; ++v) {
      int c = n0 + v * 8 + 2 * (lane & 3);
      *reinterpret_cast<float2*>(&C[(size_t)r * N + c]) = make_float2(acc[4 * v], acc[4 * v + 1]);
      *reinterpret_cast<float2*>(&C[(size_t)(r + 8) * N + c]) = make_float2(acc[4 * v + 2], acc[4 * v + 3]);
    }
  }
}

int main(int argc, char** argv) {
  return run_example("sm_90a (H100/H200)", (const void*)gemm_ws_sm90, SMEM_BYTES, argc, argv,
                     [](dim3 grid, CUtensorMap a, CUtensorMap b, float* C, int M, int N, int K) {
                       gemm_ws_sm90<<<grid, THREADS, SMEM_BYTES>>>(a, b, C, M, N, K);
                     });
}
