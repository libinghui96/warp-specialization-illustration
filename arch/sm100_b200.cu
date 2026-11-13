// B200 (sm_100a): TMA + mbarrier + tcgen05. A single thread issues the MMA for the whole 128x128 tile; operands are
// read from smem through descriptors and the accumulators live in Tensor Memory (TMEM), not in registers.
// build: nvcc -O3 -std=c++17 -gencode arch=compute_100a,code=sm_100a sm100_b200.cu -o sm100_b200 -lcuda
#include "common.h"

constexpr int STAGES = 4;
constexpr int TMEM_COLS = BN;  // one 32-bit TMEM column per output column, 128 TMEM lanes = 128 output rows
// warps 0-3: epilogue (warp w may only read TMEM lanes 32w..32w+31), warp 4: MMA issuer, warp 5: TMA producer
constexpr int MMA_WARP = 4, TMA_WARP = 5;
constexpr int THREADS = 6 * 32;
constexpr int SMEM_BYTES = STAGES * STAGE_BYTES + 1024;

// K-major operand in a 128B-swizzled tile: SBO = 1024B, descriptor version 1, layout type 2 = SW128
__device__ __forceinline__ uint64_t umma_desc(uint32_t smem_addr) {
  return uint64_t((smem_addr & 0x3FFFF) >> 4) | (uint64_t(1024 >> 4) << 32) | (1ull << 46) | (2ull << 61);
}
// instruction descriptor: D=f32 (bit 4), A/B=f16 K-major (0), N>>3 at bit 17, M>>4 at bit 24
constexpr uint32_t IDESC = (1u << 4) | (uint32_t(BN >> 3) << 17) | (uint32_t(BM >> 4) << 24);

__device__ __forceinline__ void tmem_alloc(uint32_t* dst_smem, uint32_t cols) {
  asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;\n" ::"r"(smem_u32(dst_smem)),
               "r"(cols));
  asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n" ::);
}
__device__ __forceinline__ void tmem_dealloc(uint32_t taddr, uint32_t cols) {
  asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;\n" ::"r"(taddr), "r"(cols));
}
// TMEM[taddr] (128xBN f32) = A[128x16] * B[16xBN] (+ TMEM[taddr] if accumulate)
__device__ __forceinline__ void tcgen05_mma_f16(uint32_t taddr, uint64_t desc_a, uint64_t desc_b, uint32_t idesc,
                                                uint32_t accumulate) {
  asm volatile(
      "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %4, 0;\n"
      "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n"
      "}\n" ::"r"(taddr),
      "l"(desc_a), "l"(desc_b), "r"(idesc), "r"(accumulate));
}
// `bar` receives one arrive once every MMA issued so far by this thread has completed
__device__ __forceinline__ void tcgen05_commit(uint32_t bar) {
  asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];\n" ::"r"(bar)
               : "memory");
}
__device__ __forceinline__ void tcgen05_fence_after_sync() {
  asm volatile("tcgen05.fence::after_thread_sync;\n" ::: "memory");
}
// each thread reads 16 consecutive f32 columns of its own TMEM lane
__device__ __forceinline__ void tmem_load_x16(uint32_t taddr, uint32_t (&r)[16]) {
  asm volatile(
      "tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];\n"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]), "=r"(r[8]),
        "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
      : "r"(taddr));
  asm volatile("tcgen05.wait::ld.sync.aligned;\n" ::: "memory");
}

__global__ void __launch_bounds__(THREADS, 1)
    gemm_ws_sm100(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C, int M,
                  int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES], acc_bar;
  __shared__ uint32_t tmem_base;
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN, KT = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_u32(&full_bar[s]), 1);   // TMA bytes arrived
      mbar_init(smem_u32(&empty_bar[s]), 1);  // MMAs reading this stage have finished (tcgen05.commit)
    }
    mbar_init(smem_u32(&acc_bar), 1);  // the whole tile is accumulated in TMEM
    mbar_init_fence();
  }
  if (warp == MMA_WARP) tmem_alloc(&tmem_base, TMEM_COLS);
  __syncthreads();
  uint32_t tmem = tmem_base;

  if (warp == TMA_WARP) {
    // ---------------- role 1: TMA producer (one thread) ----------------
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
  } else if (warp == MMA_WARP) {
    // ---------------- role 2: MMA issuer (one thread, never touches the data) ----------------
    if (lane == 0) {
      for (int kt = 0; kt < KT; ++kt) {
        int s = kt % STAGES;
        mbar_wait(smem_u32(&full_bar[s]), (kt / STAGES) & 1);
        tcgen05_fence_after_sync();
        uint32_t sA = base + s * STAGE_BYTES, sB = sA + TILE_A_BYTES;
#pragma unroll
        for (int kk = 0; kk < BK / 16; ++kk)
          tcgen05_mma_f16(tmem, umma_desc(sA + kk * 32), umma_desc(sB + kk * 32), IDESC, kt > 0 || kk > 0);
        tcgen05_commit(smem_u32(&empty_bar[s]));  // hands the smem stage back to the producer
      }
      tcgen05_commit(smem_u32(&acc_bar));
    }
  } else {
    // ---------------- role 3: epilogue, TMEM -> registers -> global ----------------
    mbar_wait(smem_u32(&acc_bar), 0);
    tcgen05_fence_after_sync();
    float* out = C + (size_t)(m0 + warp * 32 + lane) * N + n0;
#pragma unroll
    for (int c0 = 0; c0 < BN; c0 += 16) {
      uint32_t r[16];
      tmem_load_x16(tmem + c0, r);
#pragma unroll
      for (int j = 0; j < 16; j += 4)
        *reinterpret_cast<float4*>(out + c0 + j) = make_float4(__uint_as_float(r[j]), __uint_as_float(r[j + 1]),
                                                               __uint_as_float(r[j + 2]), __uint_as_float(r[j + 3]));
    }
  }
  __syncthreads();
  if (warp == MMA_WARP) tmem_dealloc(tmem, TMEM_COLS);
}

int main(int argc, char** argv) {
  return run_example("sm_100a (B200/GB200)", (const void*)gemm_ws_sm100, SMEM_BYTES, argc, argv,
                     [](dim3 grid, CUtensorMap a, CUtensorMap b, float* C, int M, int N, int K) {
                       gemm_ws_sm100<<<grid, THREADS, SMEM_BYTES>>>(a, b, C, M, N, K);
                     });
}
