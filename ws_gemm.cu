// Warp specialization 对比实验 (RTX 5080 / sm_120)
// C[M,N] (fp32) = A[M,K] (fp16, row-major) * B[N,K]^T (fp16, row-major)
//
// 四个 kernel 使用完全相同的 tile 尺寸、smem 布局 (128B swizzle) 和 MMA 计算代码，
// 唯一区别是 "数据如何进入 shared memory、以及谁来搬":
//   1. sync        : 所有 warp 先搬 -> __syncthreads -> 再算，单 buffer，无重叠
//   2. multistage  : 所有 warp 既搬又算，cp.async 多级流水 (Ampere 风格 software pipelining)
//   3. ws_cpasync  : 1 个 producer warp 用 cp.async 搬，8 个 consumer warp 只算，mbarrier 同步
//   4. ws_tma      : 1 个 producer 线程用 TMA 搬，8 个 consumer warp 只算，mbarrier 同步
//   5. ws256_*     : 128x256 大 tile (每个 consumer warp 64x64)，对比 setmaxnreg 寄存器重分配的作用
//                    需用 -gencode arch=compute_120a,code=sm_120a 编译才会真正启用 setmaxnreg
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cmath>
#include <vector>

#define CK(x)                                                                         \
  do {                                                                                \
    cudaError_t e_ = (x);                                                             \
    if (e_ != cudaSuccess) {                                                          \
      printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(1);                                                                        \
    }                                                                                 \
  } while (0)

constexpr int BM = 128, BN = 128, BK = 64;  // BK*2B = 128B，正好一行一个 128B swizzle atom
constexpr int STAGES = 3;
constexpr int CONSUMER_WARPS = 8;  // 2(M) x 4(N)，每个 warp 计算 64x32
constexpr int COMPUTE_THREADS = CONSUMER_WARPS * 32;
constexpr int WS_THREADS = COMPUTE_THREADS + 32;  // + 1 producer warp
constexpr int TILE_A_BYTES = BM * BK * 2;
constexpr int TILE_B_BYTES = BN * BK * 2;
constexpr int STAGE_BYTES = TILE_A_BYTES + TILE_B_BYTES;  // 32KB

// ------------------------------- PTX helpers -------------------------------
__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
// 与 TMA CU_TENSOR_MAP_SWIZZLE_128B 一致的地址映射 (要求 tile 基址 1024B 对齐)
__device__ __forceinline__ uint32_t swz(int row, int chunk16) {
  return row * 128 + ((chunk16 ^ (row & 7)) << 4);
}
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(uint32_t addr, uint32_t& r0, uint32_t& r1, uint32_t& r2,
                                            uint32_t& r3) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "r"(addr));
}
__device__ __forceinline__ void mma_16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
      "{%0,%1,%2,%3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mbar_init(uint32_t bar, uint32_t count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count));
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_arrive_expect_tx(uint32_t bar, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(bytes)
               : "memory");
}
// 本线程之前发出的所有 cp.async 完成后，异步地对 bar 做一次 arrive
__device__ __forceinline__ void cp_async_mbar_arrive_noinc(uint32_t bar) {
  asm volatile("cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n" ::"r"(bar) : "memory");
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
__device__ __forceinline__ void tma_load_2d(uint32_t dst, const CUtensorMap* tmap, int x, int y,
                                            uint32_t bar) {
  // 用 .shared::cta 而不是 .shared::cluster: 后者会生成跨 CTA 的 CALL 慢路径，并导致 setmaxnreg 被忽略
  asm volatile(
      "cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
      " [%0], [%1, {%2, %3}], [%4];\n" ::"r"(dst),
      "l"(reinterpret_cast<uint64_t>(tmap)), "r"(x), "r"(y), "r"(bar)
      : "memory");
}

#if defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM100_ALL) || \
    defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define WS_HAS_SETMAXNREG 1
#endif
// 同一 warpgroup (4 个 warp) 必须一起执行
template <uint32_t R>
__device__ __forceinline__ void reg_dealloc() {
#ifdef WS_HAS_SETMAXNREG
  asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R));
#endif
}
template <uint32_t R>
__device__ __forceinline__ void reg_alloc() {
#ifdef WS_HAS_SETMAXNREG
  asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R));
#endif
}
__global__ void probe_setmaxnreg(int* out) {
#ifdef WS_HAS_SETMAXNREG
  *out = 1;
#else
  *out = 0;
#endif
}

// ------------------------- 共用的 load / compute / store -------------------------
// 由 [tid, nthr) 这些线程协作把一个 k-tile 的 A、B 用 cp.async 搬进 smem
__device__ __forceinline__ void load_tile_cpasync(uint32_t sA, uint32_t sB, const half* A, const half* B,
                                                  int K, int m0, int n0, int k0, int tid, int nthr) {
#pragma unroll 4
  for (int i = tid; i < BM * BK / 8; i += nthr) {
    int row = i >> 3, c = i & 7;
    cp_async16(sA + swz(row, c), A + (size_t)(m0 + row) * K + k0 + c * 8);
  }
#pragma unroll 4
  for (int i = tid; i < BN * BK / 8; i += nthr) {
    int row = i >> 3, c = i & 7;
    cp_async16(sB + swz(row, c), B + (size_t)(n0 + row) * K + k0 + c * 8);
  }
}

// 一个 warp 在一个 k-tile 上计算 (MI*16) x (NI*8) 的输出；KG>1 时一次预取 KG 个 k16 的片段 (更多 ILP，更多寄存器)
template <int MI, int NI, int KG = 1>
__device__ __forceinline__ void mma_tile(uint32_t sA, uint32_t sB, float (&acc)[MI][NI][4], int wm, int wn,
                                         int lane) {
#pragma unroll
  for (int kk0 = 0; kk0 < BK / 16; kk0 += KG) {
    uint32_t a[KG][MI][4], b[KG][NI][2];
#pragma unroll
    for (int g = 0; g < KG; ++g) {
      int kk = kk0 + g;
#pragma unroll
      for (int mi = 0; mi < MI; ++mi) {
        int row = wm * (MI * 16) + mi * 16 + (lane & 15);
        int c = kk * 2 + (lane >> 4);
        ldmatrix_x4(sA + swz(row, c), a[g][mi][0], a[g][mi][1], a[g][mi][2], a[g][mi][3]);
      }
#pragma unroll
      for (int nj = 0; nj < NI / 2; ++nj) {
        int row = wn * (NI * 8) + nj * 16 + (lane & 7) + ((lane >> 4) << 3);
        int c = kk * 2 + ((lane >> 3) & 1);
        ldmatrix_x4(sB + swz(row, c), b[g][2 * nj][0], b[g][2 * nj][1], b[g][2 * nj + 1][0],
                    b[g][2 * nj + 1][1]);
      }
    }
#pragma unroll
    for (int g = 0; g < KG; ++g)
#pragma unroll
      for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int ni = 0; ni < NI; ++ni) mma_16816(acc[mi][ni], a[g][mi], b[g][ni]);
  }
}

// GUARD=true 时对越界行/列不写 (要求 N 为偶数)
template <bool GUARD = false, int MI, int NI>
__device__ __forceinline__ void store_c(float* C, int N, int m0, int n0, const float (&acc)[MI][NI][4], int wm,
                                        int wn, int lane, int M = 0) {
  int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mi = 0; mi < MI; ++mi)
#pragma unroll
    for (int ni = 0; ni < NI; ++ni) {
      int r = m0 + wm * (MI * 16) + mi * 16 + g;
      int c = n0 + wn * (NI * 8) + ni * 8 + 2 * t;
      if (GUARD && c >= N) continue;
      if (!GUARD || r < M)
        *reinterpret_cast<float2*>(&C[(size_t)r * N + c]) = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
      if (!GUARD || r + 8 < M)
        *reinterpret_cast<float2*>(&C[(size_t)(r + 8) * N + c]) = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
    }
}

__device__ __forceinline__ uint32_t aligned_smem_base(uint8_t* raw) { return (smem_u32(raw) + 1023) & ~1023u; }

// ------------------------------- Kernel 1: sync -------------------------------
__global__ void __launch_bounds__(COMPUTE_THREADS) gemm_sync(const half* A, const half* B, float* C, int M,
                                                             int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  uint32_t sA = aligned_smem_base(smem_raw), sB = sA + TILE_A_BYTES;
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp >> 2, wn = warp & 3;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  float acc[4][4][4] = {};

  for (int k0 = 0; k0 < K; k0 += BK) {
    load_tile_cpasync(sA, sB, A, B, K, m0, n0, k0, tid, COMPUTE_THREADS);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();  // 等搬运完成: 这段时间 tensor core 空闲
    mma_tile(sA, sB, acc, wm, wn, lane);
    __syncthreads();  // 等计算完成才能覆盖 buffer: 这段时间内存系统空闲
  }
  store_c(C, N, m0, n0, acc, wm, wn, lane);
}

// ---------------------------- Kernel 2: multistage ----------------------------
__global__ void __launch_bounds__(COMPUTE_THREADS) gemm_multistage(const half* A, const half* B, float* C,
                                                                   int M, int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp >> 2, wn = warp & 3;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int KT = K / BK;
  float acc[4][4][4] = {};

#pragma unroll
  for (int s = 0; s < STAGES - 1; ++s) {
    if (s < KT) load_tile_cpasync(base + s * STAGE_BYTES, base + s * STAGE_BYTES + TILE_A_BYTES, A, B, K, m0, n0,
                                  s * BK, tid, COMPUTE_THREADS);
    cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    int pf = kt + STAGES - 1;  // 预取的 slot 正是上一轮算完的那个
    if (pf < KT) {
      uint32_t s = base + (pf % STAGES) * STAGE_BYTES;
      load_tile_cpasync(s, s + TILE_A_BYTES, A, B, K, m0, n0, pf * BK, tid, COMPUTE_THREADS);
    }
    cp_async_commit();
    uint32_t s = base + (kt % STAGES) * STAGE_BYTES;
    mma_tile(s, s + TILE_A_BYTES, acc, wm, wn, lane);
  }
  store_c(C, N, m0, n0, acc, wm, wn, lane);
}

// ------------------ 对照组: TMA 多级流水，但不做 warp specialization ------------------
// 线程 0 负责发 TMA，但它同时也是 consumer；每个 k-tile 全 block __syncthreads 一次
__global__ void __launch_bounds__(COMPUTE_THREADS)
    gemm_multistage_tma(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                        float* C, int M, int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int wm = warp >> 2, wn = warp & 3;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int KT = K / BK;
  float acc[4][4][4] = {};

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) mbar_init(smem_u32(&full_bar[s]), 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();
  if (tid == 0) {
    for (int s = 0; s < STAGES - 1 && s < KT; ++s) {
      uint32_t fb = smem_u32(&full_bar[s]), sA = base + s * STAGE_BYTES;
      mbar_arrive_expect_tx(fb, STAGE_BYTES);
      tma_load_2d(sA, &tmA, s * BK, m0, fb);
      tma_load_2d(sA + TILE_A_BYTES, &tmB, s * BK, n0, fb);
    }
  }
  for (int kt = 0; kt < KT; ++kt) {
    int s = kt % STAGES;
    mbar_wait(smem_u32(&full_bar[s]), (kt / STAGES) & 1);
    __syncthreads();  // 保证所有 warp 都算完了 kt-1，才能覆盖它的 slot
    int pf = kt + STAGES - 1;
    if (tid == 0 && pf < KT) {
      int ps = pf % STAGES;
      uint32_t fb = smem_u32(&full_bar[ps]), sA = base + ps * STAGE_BYTES;
      mbar_arrive_expect_tx(fb, STAGE_BYTES);
      tma_load_2d(sA, &tmA, pf * BK, m0, fb);
      tma_load_2d(sA + TILE_A_BYTES, &tmB, pf * BK, n0, fb);
    }
    uint32_t sA = base + s * STAGE_BYTES;
    mma_tile(sA, sA + TILE_A_BYTES, acc, wm, wn, lane);
  }
  store_c(C, N, m0, n0, acc, wm, wn, lane);
}

// ------------------------ Kernel 3/4: warp specialized ------------------------
template <bool USE_TMA>
__global__ void __launch_bounds__(WS_THREADS)
    gemm_ws(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, const half* A,
            const half* B, float* C, int M, int N, int K) {
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES], empty_bar[STAGES];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
  int KT = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES; ++s) {
      mbar_init(smem_u32(&full_bar[s]), USE_TMA ? 1 : 32);  // TMA: 1 次 expect_tx；cp.async: 32 个线程各 arrive 一次
      mbar_init(smem_u32(&empty_bar[s]), CONSUMER_WARPS);    // 每个 consumer warp 释放一次
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();  // 这是 kernel 内唯一一次全 block 同步

  if (warp == CONSUMER_WARPS) {
    // ============================ producer ============================
    if (USE_TMA) {
      if (lane == 0) {  // 只需要 1 个线程发 TMA，其余 31 个线程空闲
        for (int kt = 0; kt < KT; ++kt) {
          int s = kt % STAGES;
          uint32_t ph = (kt / STAGES) & 1;
          mbar_wait(smem_u32(&empty_bar[s]), ph ^ 1);  // 首轮 parity=1 直接通过
          uint32_t sA = base + s * STAGE_BYTES;
          uint32_t fb = smem_u32(&full_bar[s]);
          mbar_arrive_expect_tx(fb, STAGE_BYTES);
          tma_load_2d(sA, &tmA, kt * BK, m0, fb);
          tma_load_2d(sA + TILE_A_BYTES, &tmB, kt * BK, n0, fb);
        }
      }
    } else {
      for (int kt = 0; kt < KT; ++kt) {
        int s = kt % STAGES;
        uint32_t ph = (kt / STAGES) & 1;
        mbar_wait(smem_u32(&empty_bar[s]), ph ^ 1);
        uint32_t sA = base + s * STAGE_BYTES;
        load_tile_cpasync(sA, sA + TILE_A_BYTES, A, B, K, m0, n0, kt * BK, lane, 32);
        cp_async_mbar_arrive_noinc(smem_u32(&full_bar[s]));
      }
      cp_async_wait<0>();
    }
  } else {
    // ============================ consumers ============================
    int wm = warp >> 2, wn = warp & 3;
    float acc[4][4][4] = {};
    for (int kt = 0; kt < KT; ++kt) {
      int s = kt % STAGES;
      uint32_t ph = (kt / STAGES) & 1;
      mbar_wait(smem_u32(&full_bar[s]), ph);  // 只等 "这一份数据到了"，不等其他 warp
      uint32_t sA = base + s * STAGE_BYTES;
      mma_tile(sA, sA + TILE_A_BYTES, acc, wm, wn, lane);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&empty_bar[s]));  // 归还 buffer 给 producer
    }
    store_c(C, N, m0, n0, acc, wm, wn, lane);
  }
}

// --------------- Kernel 5: 大 tile warp specialized (+ setmaxnreg) ---------------
constexpr int BM_BIG = 192, BN_BIG = 192, STAGES_BIG = 2;
constexpr int TILE_A_BIG_BYTES = BM_BIG * BK * 2;
constexpr int TILE_B_BIG_BYTES = BN_BIG * BK * 2;
constexpr int STAGE_BIG_BYTES = TILE_A_BIG_BYTES + TILE_B_BIG_BYTES;  // 48KB
constexpr int PRODUCER_REGS = 40, CONSUMER_REGS = 232;           // 128*40 + 256*232 <= 64K

// PRODUCER_WARPS=1: 288 线程，__launch_bounds__ 允许每线程 224 寄存器
// PRODUCER_WARPS=4: 384 线程，只允许 168 寄存器，consumer 需靠 setmaxnreg 从 producer 借寄存器
template <int PRODUCER_WARPS, bool SETMAXNREG>
__global__ void __launch_bounds__((CONSUMER_WARPS + PRODUCER_WARPS) * 32, 1)
    gemm_ws_big(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB, float* C, int M,
                int N, int K) {
  static_assert(!SETMAXNREG || PRODUCER_WARPS == 4, "setmaxnreg works per warpgroup");
  extern __shared__ uint8_t smem_raw[];
  __shared__ __align__(8) uint64_t full_bar[STAGES_BIG], empty_bar[STAGES_BIG];
  uint32_t base = aligned_smem_base(smem_raw);
  int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  int m0 = blockIdx.y * BM_BIG, n0 = blockIdx.x * BN_BIG;
  int KT = K / BK;

  if (tid == 0) {
    for (int s = 0; s < STAGES_BIG; ++s) {
      mbar_init(smem_u32(&full_bar[s]), 1);
      mbar_init(smem_u32(&empty_bar[s]), CONSUMER_WARPS);
    }
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  __syncthreads();

  if (warp >= CONSUMER_WARPS) {
    if constexpr (SETMAXNREG) reg_dealloc<PRODUCER_REGS>();
    if (warp == CONSUMER_WARPS && lane == 0) {
      for (int kt = 0; kt < KT; ++kt) {
        int s = kt % STAGES_BIG;
        uint32_t ph = (kt / STAGES_BIG) & 1;
        mbar_wait(smem_u32(&empty_bar[s]), ph ^ 1);
        uint32_t sA = base + s * STAGE_BIG_BYTES;
        uint32_t fb = smem_u32(&full_bar[s]);
        mbar_arrive_expect_tx(fb, STAGE_BIG_BYTES);
        tma_load_2d(sA, &tmA, kt * BK, m0, fb);
        tma_load_2d(sA + TILE_A_BIG_BYTES, &tmB, kt * BK, n0, fb);
      }
    }
  } else {
    if constexpr (SETMAXNREG) reg_alloc<CONSUMER_REGS>();
    int wm = warp >> 2, wn = warp & 3;
    float acc[6][6][4] = {};  // warp tile 96x48: 144 个累加器
    for (int kt = 0; kt < KT; ++kt) {
      int s = kt % STAGES_BIG;
      uint32_t ph = (kt / STAGES_BIG) & 1;
      mbar_wait(smem_u32(&full_bar[s]), ph);
      uint32_t sA = base + s * STAGE_BIG_BYTES;
      mma_tile<6, 6, 2>(sA, sA + TILE_A_BIG_BYTES, acc, wm, wn, lane);
      __syncwarp();
      if (lane == 0) mbar_arrive(smem_u32(&empty_bar[s]));
    }
    store_c<true>(C, N, m0, n0, acc, wm, wn, lane, M);
  }
}

// ---------------------------------- host ----------------------------------
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

template <typename F>
static float time_ms(F&& launch, int iters) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  for (int i = 0; i < 3; ++i) launch();
  CK(cudaEventRecord(e0));
  for (int i = 0; i < iters; ++i) launch();
  CK(cudaEventRecord(e1));
  CK(cudaEventSynchronize(e1));
  CK(cudaGetLastError());
  float ms;
  CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaEventDestroy(e0));
  CK(cudaEventDestroy(e1));
  return ms / iters;
}

static double max_rel_err(const std::vector<float>& ref, const std::vector<float>& out) {
  double max_diff = 0, max_ref = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    max_diff = fmax(max_diff, fabs((double)ref[i] - out[i]));
    max_ref = fmax(max_ref, fabs((double)ref[i]));
  }
  return max_diff / max_ref;
}

static void run_size(int M, int N, int K, int iters, cublasHandle_t handle) {
  printf("\n===== M=%d N=%d K=%d =====\n", M, N, K);
  size_t nA = (size_t)M * K, nB = (size_t)N * K, nC = (size_t)M * N;
  std::vector<half> hA(nA), hB(nB);
  srand(42);
  for (auto& x : hA) x = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);
  for (auto& x : hB) x = __float2half((rand() / (float)RAND_MAX) * 2.f - 1.f);
  half *dA, *dB;
  float *dC, *dRef;
  CK(cudaMalloc(&dA, nA * sizeof(half)));
  CK(cudaMalloc(&dB, nB * sizeof(half)));
  CK(cudaMalloc(&dC, nC * sizeof(float)));
  CK(cudaMalloc(&dRef, nC * sizeof(float)));
  CK(cudaMemcpy(dA, hA.data(), nA * sizeof(half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hB.data(), nB * sizeof(half), cudaMemcpyHostToDevice));

  CUtensorMap tmA = make_tmap(dA, M, K, BM), tmB = make_tmap(dB, N, K, BN);
  CUtensorMap tmA192 = make_tmap(dA, M, K, BM_BIG), tmB192 = make_tmap(dB, N, K, BN_BIG);
  double flops = 2.0 * M * N * K;
  std::vector<float> ref(nC), out(nC);

  // cuBLAS 参考: 列主序视角下 C^T(NxM) = B(NxK) * A^T
  float alpha = 1.f, beta = 0.f;
  auto cublas = [&] {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha, dB, CUDA_R_16F, K, dA, CUDA_R_16F, K, &beta,
                 dRef, CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
  };
  float ms = time_ms(cublas, iters);
  CK(cudaMemcpy(ref.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));
  double spot_err = 0;  // 用 CPU 抽查 cuBLAS 参考结果本身
  for (int t = 0; t < 64; ++t) {
    int i = rand() % M, j = rand() % N;
    double s = 0;
    for (int k = 0; k < K; ++k)
      s += (double)__half2float(hA[(size_t)i * K + k]) * (double)__half2float(hB[(size_t)j * K + k]);
    spot_err = fmax(spot_err, fabs(s - ref[(size_t)i * N + j]) / (fabs(s) + 1.0));
  }
  printf("%-14s %8.3f ms  %7.2f TFLOPS  (CPU spot-check err=%.2e)\n", "cuBLAS", ms, flops / ms / 1e9, spot_err);

  dim3 grid(N / BN, M / BM);
  int smem_sync = TILE_A_BYTES + TILE_B_BYTES + 1024;
  int smem_pipe = STAGES * STAGE_BYTES + 1024;

  auto report = [&](const char* name, const void* fn, int threads, int smem, auto&& launch) {
    CK(cudaMemset(dC, 0, nC * sizeof(float)));
    float t = time_ms(launch, iters);
    CK(cudaMemcpy(out.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
    int blocks_per_sm = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, fn, threads, smem));
    cudaFuncAttributes fa;
    CK(cudaFuncGetAttributes(&fa, fn));
    double err = max_rel_err(ref, out);
    printf("%-14s %8.3f ms  %7.2f TFLOPS  blk/SM=%d warps=%2d regs=%3d spill=%4zuB  err=%.1e %s\n", name, t,
           flops / t / 1e9, blocks_per_sm, threads / 32, fa.numRegs, fa.localSizeBytes, err,
           err < 1e-3 ? "PASS" : "FAIL");
  };

  report("sync", (const void*)gemm_sync, COMPUTE_THREADS, smem_sync,
         [&] { gemm_sync<<<grid, COMPUTE_THREADS, smem_sync>>>(dA, dB, dC, M, N, K); });
  // 多申请 smem 强制 1 block/SM，去掉 "多 CTA 互相掩盖" 的效果
  report("sync_1blk", (const void*)gemm_sync, COMPUTE_THREADS, smem_pipe,
         [&] { gemm_sync<<<grid, COMPUTE_THREADS, smem_pipe>>>(dA, dB, dC, M, N, K); });
  report("multistage", (const void*)gemm_multistage, COMPUTE_THREADS, smem_pipe,
         [&] { gemm_multistage<<<grid, COMPUTE_THREADS, smem_pipe>>>(dA, dB, dC, M, N, K); });
  report("multistage_tma", (const void*)gemm_multistage_tma, COMPUTE_THREADS, smem_pipe,
         [&] { gemm_multistage_tma<<<grid, COMPUTE_THREADS, smem_pipe>>>(tmA, tmB, dC, M, N, K); });
  report("ws_cpasync", (const void*)gemm_ws<false>, WS_THREADS, smem_pipe,
         [&] { gemm_ws<false><<<grid, WS_THREADS, smem_pipe>>>(tmA, tmB, dA, dB, dC, M, N, K); });
  report("ws_tma", (const void*)gemm_ws<true>, WS_THREADS, smem_pipe,
         [&] { gemm_ws<true><<<grid, WS_THREADS, smem_pipe>>>(tmA, tmB, dA, dB, dC, M, N, K); });

  {
    dim3 grid_big((N + BN_BIG - 1) / BN_BIG, (M + BM_BIG - 1) / BM_BIG);
    int smem_big = STAGES_BIG * STAGE_BIG_BYTES + 1024;
    report("ws192_1warp", (const void*)gemm_ws_big<1, false>, 9 * 32, smem_big,
           [&] { gemm_ws_big<1, false><<<grid_big, 9 * 32, smem_big>>>(tmA192, tmB192, dC, M, N, K); });
    report("ws192_wg", (const void*)gemm_ws_big<4, false>, 12 * 32, smem_big,
           [&] { gemm_ws_big<4, false><<<grid_big, 12 * 32, smem_big>>>(tmA192, tmB192, dC, M, N, K); });
    report("ws192_wg_reg", (const void*)gemm_ws_big<4, true>, 12 * 32, smem_big,
           [&] { gemm_ws_big<4, true><<<grid_big, 12 * 32, smem_big>>>(tmA192, tmB192, dC, M, N, K); });
  }

  CK(cudaFree(dA));
  CK(cudaFree(dB));
  CK(cudaFree(dC));
  CK(cudaFree(dRef));
}

int main(int argc, char** argv) {
  int iters = 20;
  std::vector<int> sizes = {2048, 4096, 8192};
  if (argc > 1) {
    sizes.clear();
    for (int i = 1; i < argc; ++i) sizes.push_back(atoi(argv[i]));
  }
  cudaDeviceProp prop;
  CK(cudaGetDeviceProperties(&prop, 0));
  printf("GPU: %s  SMs=%d  smem/block(optin)=%zu KB\n", prop.name, prop.multiProcessorCount,
         prop.sharedMemPerBlockOptin / 1024);

  int smem_pipe = STAGES * STAGE_BYTES + 1024;
  CK(cudaFuncSetAttribute(gemm_sync, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pipe));
  CK(cudaFuncSetAttribute(gemm_multistage_tma, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pipe));
  CK(cudaFuncSetAttribute(gemm_multistage, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pipe));
  CK(cudaFuncSetAttribute(gemm_ws<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pipe));
  CK(cudaFuncSetAttribute(gemm_ws<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_pipe));
  int smem_big = STAGES_BIG * STAGE_BIG_BYTES + 1024;
  CK(cudaFuncSetAttribute(gemm_ws_big<1, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_big));
  CK(cudaFuncSetAttribute(gemm_ws_big<4, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_big));
  CK(cudaFuncSetAttribute(gemm_ws_big<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_big));

  int *d_flag, has_smr = 0;
  CK(cudaMalloc(&d_flag, sizeof(int)));
  probe_setmaxnreg<<<1, 1>>>(d_flag);
  CK(cudaMemcpy(&has_smr, d_flag, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaFree(d_flag));
  printf("CUDA runtime %d.%d, setmaxnreg %s\n", CUDART_VERSION / 1000, CUDART_VERSION % 1000 / 10,
         has_smr ? "ENABLED" : "DISABLED (build with -gencode arch=compute_120a,code=sm_120a)");

  cublasHandle_t handle;
  cublasCreate(&handle);
  for (int s : sizes) {
    if (s % 128 != 0) {
      printf("size %d must be a multiple of 128, skipped\n", s);
      continue;
    }
    run_size(s, s, s, iters, handle);
  }
  cublasDestroy(handle);
  return 0;
}
