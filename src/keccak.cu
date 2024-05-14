//keccak.cu

#include <stdint.h>

#include <cuda_runtime.h>

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

__constant__ static const uint32_t RHO[24] = {
    1, 3, 6, 10, 15, 21, 28, 36, 45, 55, 2, 14, 27, 41, 56, 8, 25, 43, 62, 18, 39, 61, 20, 44,
};

__constant__ static const size_t PI[24] = {
    10, 7, 11, 17, 18, 3, 5, 16, 8, 21, 24, 4, 15, 23, 19, 13, 12, 2, 20, 14, 22, 9, 6, 1,
};

__device__ __forceinline__ uint64_t rotate_left_u64(uint64_t result, uint32_t j) {
    if (j == 0) return result;
    return (result << j) | (result >> (64 - j));
}

__device__ void theta(uint64_t *A, uint64_t *array) {
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            int y5= y * 5;
            array[x] ^= A[x + y5];
        }
    }
    for (int x = 0; x < 5; x++) {
        for (int y = 0; y < 5; y++) {
            int y5 = y * 5;
            A[y5 + x] ^= array[(x + 4) % 5] ^ rotate_left_u64(array[(x + 1) % 5], 1);
        }
    }
}


__device__ void rho_and_pi(uint64_t *A, uint64_t *array) {
    uint64_t last = A[1];
    for (int t = 0; t < 24; t++) {
        size_t index = PI[t];
        array[0] = A[index];
        A[index] = rotate_left_u64(last, RHO[t]);
        last = array[0];
    }
}

__device__ void chi(uint64_t *A, uint64_t *array) {
    for (int y = 0; y < 5; y++) {
        int y5 = 5 * y;
        for (int x = 0; x < 5; x++) {
            array[x] = A[x + y5];
        }
        for (int x = 0; x < 5; x++) {
            A[x + y5] = array[x] ^ ((~array[(x + 1) % 5]) & (array[(x + 2) % 5]));
        }
    }
}

__device__ void iota(uint64_t *A, int round) {
    A[0] ^= RC[round];
}

__device__ void keccakp(uint64_t *A) {
    for (int round = 0; round < ROUNDS; round++) {
        uint64_t array[5] = {0};

        theta(A, array);
        rho_and_pi(A, array);
        chi(A, array);
        iota(A, round);
    }
}


