//Code rewritten from tiny_keccak::keccakp in Rust https://github.com/debris/tiny-keccak

#include <stdint.h>

#include <cuda_runtime.h>

#include "xelis-hash.cuh"

#define STATE_SIZE 25
#define ROUNDS 12

__constant__ static const uint64_t RC[ROUNDS] = {
    0x000000008000808bULL,
    0x800000000000008bULL,
    0x8000000000008089ULL,
    0x8000000000008003ULL,
    0x8000000000008002ULL,
    0x8000000000000080ULL,
    0x000000000000800aULL,
    0x800000008000000aULL,
    0x8000000080008081ULL,
    0x8000000000008080ULL,
    0x0000000080000001ULL,
    0x8000000080008008ULL,
};

__constant__ static const uint8_t RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
};

__constant__ static const uint8_t PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};

//same as PI, but with extra 1 at pos 0
__constant__ static const uint8_t PI2[25] = {
    1, 10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};

__device__ __forceinline__ uint64_t rotate_left_u64(uint64_t result, uint32_t j) {
    if (j == 0) return result;
    return (result << j) | (result >> (64 - j));
}


__device__ __forceinline__ void theta(uint64_t *__restrict__ A, uint64_t *__restrict__ array) {
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            array[x] ^= A[x + y];
        }
    }
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            A[y + x] ^= array[(x + 4) % 5] ^ rotate_left_u64(array[(x + 1) % 5], 1);
        }
    }
}

__device__ __forceinline__ void theta_copy(uint64_t *__restrict__ A, uint64_t *__restrict__ O, uint64_t *array) {
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            array[x] ^= A[x + y];
        }
    }
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            O[y + x] = A[y + x] ^ array[(x + 4) % 5] ^ rotate_left_u64(array[(x + 1) % 5], 1);
        }
    }
}


__device__ __forceinline__ void rho_and_pi(uint64_t *__restrict__ A, uint64_t *__restrict__ array) {
    uint64_t last = A[1];
    for (int t = 0; t < 24; t++) {
        uint8_t index = PI[t];
        array[0] = A[index];
        A[index] = rotate_left_u64(last, RHO[t]);
        last = array[0];
    }
}

__device__ __forceinline__ void chi_copy(uint64_t *__restrict__ A,
        uint64_t *__restrict__ O, uint64_t *__restrict__ array) {
    for (int y = 0; y < 25; y+=5) {
        for (int x = 0; x < 5; x++) {
            array[x] = A[x + y];
        }
        for (int x = 0; x < 5; x++) {
            O[x + y] = array[x] ^ ((~array[(x + 1) % 5]) & (array[(x + 2) % 5]));
        }
    }
}

__device__ __forceinline__ void chi(uint64_t *__restrict__ A, uint64_t *__restrict__ array) {
    for (int y = 0; y < 25; y+=5) {
        for (int x = 0; x < 5; x++) {
            array[x] = A[x + y];
        }
        for (int x = 0; x < 5; x++) {
            A[x + y] = array[x] ^ ((~array[(x + 1) % 5]) & (array[(x + 2) % 5]));
        }
    }
}

__device__ __forceinline__ void iota(uint64_t *__restrict__ A, int round) {
    A[0] ^= RC[round];
}

// Original implementation with extra copy to local buffer inside theta() and chi()
__device__ void keccakp0(uint64_t * __restrict__ I, uint64_t *__restrict__ O) {
    uint64_t A[25];
    {
        uint64_t array[5] = {0};

        theta_copy(I, A, array);
        rho_and_pi(A, array);
        chi(A, array);
        iota(A, 0);
    }
    for (int round = 1; round < ROUNDS - 1; round++) {
        uint64_t array[5] = {0};

        theta(A, array);
        rho_and_pi(A, array);
        chi(A, array);
        iota(A, round);
    }
    {
        uint64_t array[5] = {0};

        theta(A, array);
        rho_and_pi(A, array);
        chi_copy(A, O, array);
        iota(O, ROUNDS - 1);
    }
}

// I -> A -> A2 -> O, buffers can be reused
__device__ __forceinline__ void keccakp_impl(uint64_t * I, uint64_t * A,
        uint64_t * A2, uint64_t *O, uint64_t *array, int round) {
/*    size_t tid = threadIdx.x / STAGE_0_WARP;
    size_t num_threads = blockDim.x / STAGE_0_WARP;
    size_t num_warps = STAGE_0_WARP;
    size_t wid = threadIdx.x % STAGE_0_WARP;*/
    /////////////////////////////////////////
    uint64_t tmp_a;

    for (int x = 0; x < 5; x++) {
        array[x] = I[x + 0] ^ I[x + 5] ^ I[x + 10] ^ I[x + 15] ^ I[x + 20];
    }
    for (int x = 0; x < 5; x++) {
        tmp_a = array[(x + 4) % 5] ^ rotate_left_u64(array[(x + 1) % 5], 1);
        for (int y = 0; y < 25; y+=5) {
            A[y + x] = I[y + x] ^ tmp_a;
        }
    }
    for (int t = 1; t < 25; t++) {
        A2[PI2[t]] = rotate_left_u64(A[PI2[t - 1]], RHO[t - 1]);
    }
    A2[0] = A[0];
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            O[x + y] = A2[x + y] ^ ((~A2[y + (x + 1) % 5]) & (A2[y + (x + 2) % 5]));
        }
    }
    O[0] ^= RC[round];
}

// I -> O, with local/shared buffers
__device__ void keccakp(uint64_t * __restrict__ I, uint64_t *__restrict__ O) {
#if 0
    size_t tid = threadIdx.x / STAGE_0_WARP;
    __shared__ uint64_t _A[KECCAK_WORDS * KECCAK_THREADS];
    __shared__ uint64_t _A2[KECCAK_WORDS * KECCAK_THREADS];
    __shared__ uint64_t array[5 * KECCAK_THREADS];
    uint64_t *A = &_A[tid * KECCAK_WORDS];
    uint64_t *A2 = &_A2[tid * KECCAK_WORDS];
#else
    uint64_t A[KECCAK_WORDS];
    uint64_t A2[KECCAK_WORDS];
    uint64_t array[5];
#endif
    keccakp_impl(I, A, A2, A, array, 0);
    keccakp_impl(A, A2, A, A2, array, 1);

    for (int round = 1; round < ROUNDS/2 - 1; round++) {
        keccakp_impl(A2, A, A2, A, array, round * 2);
        keccakp_impl(A, A2, A, A2, array, round * 2 + 1);
    }
    keccakp_impl(A2, A, A2, A, array, ROUNDS - 2);
    keccakp_impl(A, A2, A, O, array, ROUNDS - 1);
}

////////////////////////////////////////////////////

__device__ __forceinline__ void keccakp2_impl(uint64_t *__restrict__ I, uint64_t *__restrict__ O,
        uint64_t *__restrict__ B, uint64_t *__restrict__ array, int round) {
    /////////////////////////////////////////
    uint64_t tmp_a;

    for (int x = 0; x < 5; x++) {
        array[x] = I[x + 0] ^ I[x + 5] ^ I[x + 10] ^ I[x + 15] ^ I[x + 20];
    }
    for (int x = 0; x < 5; x++) {
        tmp_a = array[(x + 4) % 5] ^ rotate_left_u64(array[(x + 1) % 5], 1);
        for (int y = 0; y < 25; y+=5) {
            O[y + x] = I[y + x] ^ tmp_a;
        }
    }
    for (int t = 1; t < 25; t++) {
        B[PI2[t]] = rotate_left_u64(O[PI2[t - 1]], RHO[t - 1]);
    }
    B[0] = O[0];
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 25; y+=5) {
            O[x + y] = B[x + y] ^ ((~B[y + (x + 1) % 5]) & (B[y + (x + 2) % 5]));
        }
    }
    O[0] ^= RC[round];
}

// I -> O, with buffer B, also using I as buffer
__device__ void keccakp2(uint64_t * __restrict__ I, uint64_t *__restrict__ O, uint64_t *__restrict__ B) {
    uint64_t array[5];

    for (int round = 0; round < ROUNDS/2; round++) {
        keccakp2_impl(I, B, O, array, round * 2);
        keccakp2_impl(B, O, I, array, round * 2 + 1);
    }
}

