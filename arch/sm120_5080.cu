// RTX 5080 (sm_120): TMA + mbarrier + mma.sync. Operands go smem -> registers (ldmatrix) -> Tensor Core,
// and the accumulators live in the registers of each consumer warp.
// build: nvcc -O3 -std=c++17 -gencode arch=compute_120a,code=sm_120a sm120_5080.cu -o sm120_5080 -lcuda
#include "common.h"

constexpr int STAGES = 3;          // 3 x 32KB fits the 99KB smem of the 5080
constexpr int CONSUMER_WARPS = 8;  // 2(M) x 4(N), each warp owns a 64x32 piece of the tile
constexpr int THREADS = (CONSUMER_WARPS + 1) * 32;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + 1024;

// Address of 16B chunk `c` of `row` inside a 128B-swizzled tile (same mapping TMA writes with)
__device__ __forceinline__ uint32_t swz(int row, int c) { return row * 128 + ((c ^ (row & 7)) << 4); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void __launch_bounds__(THREADS, 1)
    gemm_ws_sm120(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C, int M,
                  int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_u32(&full_bar[s]), 1);                // completed by the TMA byte count
      mbar_init(smem_u32(&empty_bar[s]), CONSUMER_WARPS);  // one arrive per consumer warp
    }
    mbar_init_fence();
  }
  __syncthreads();

  if (warp == CONSUMER_WARPS) {
    // ---------------- producer: a single thread drives the TMA engine ----------------
    if (lane == 0) {
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
    // ---------------- consumers: 8 warps, each issues its own mma.sync ----------------
    int wm = warp >> 2, wn = warp & 3;
    float acc[4][4][4] = {};  // 64x32 outputs = 16 m16n8 fragments x 4 values per thread
    for (int kt = 0; kt < KT; ++kt) {
      int s = kt % STAGES;
      mbar_wait(smem_u32(&full_bar[s]), (kt / STAGES) & 1);
      uint32_t sA = base + s * STAGE_BYTES, sB = sA + TILE_A_BYTES;
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        uint32_t a[4][4], b[4][2];
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
          ldmatrix_x4(sA + swz(wm * 64 + mi * 16 + (lane & 15), kk * 2 + (lane >> 4)), a[mi][0], a[mi][1], a[mi][2],
                      a[mi][3]);
#pragma unroll
        for (int nj = 0; nj < 2; ++nj)
          ldmatrix_x4(sB + swz(wn * 32 + nj * 16 + (lane & 7) + ((lane >> 4) << 3), kk * 2 + ((lane >> 3) & 1)),
                      b[2 * nj][0], b[2 * nj][1], b[2 * nj + 1][0], b[2 * nj + 1][1]);
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
          for (int ni = 0; ni < 4; ++ni) mma_m16n8k16(acc[mi][ni], a[mi], b[ni]);
      }
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&empty_bar[s]));
    }
#pragma unroll
    for (int mi = 0; mi < 4; ++mi)
#pragma unroll
      for (int ni = 0; ni < 4; ++ni) {
        int r = m0 + wm * 64 + mi * 16 + (lane >> 2), c = n0 + wn * 32 + ni * 8 + 2 * (lane & 3);
        *reinterpret_cast<float2*>(&C[(size_t)r * N + c]) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
        *reinterpret_cast<float2*>(&C[(size_t)(r + 8) * N + c]) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
      }
  }
}

int main(int argc, char** argv) {
  return run_example("sm_120a (RTX 50)", (const void*)gemm_ws_sm120, SMEM_BYTES, argc, argv,
                     [](dim3 grid, CUtensorMap a, CUtensorMap b, float* C, int M, int N, int K) {
                       gemm_ws_sm120<<<grid, THREADS, SMEM_BYTES>>>(a, b, C, M, N, K);
                     });
}
