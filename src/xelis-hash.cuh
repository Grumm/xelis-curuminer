//xelis-hash.h

#ifndef XELIS_HASH_H
#define XELIS_HASH_H

#include <stdint.h>


#define KECCAK_WORDS 25
#define MEMORY_SIZE 32768
#define BYTES_ARRAY_INPUT KECCAK_WORDS * 8
#define SLOT_LENGTH 256
#define SLOT_LENGTH_64 (SLOT_LENGTH / 2)
#define HASH_SIZE 32
#define BUFFER_SIZE 42
#define STAGE_1_MAX (MEMORY_SIZE / KECCAK_WORDS)
#define MEMORY_PAD SLOT_LENGTH

#define ITERS 1
#define SCRATCHPAD_ITERS 5000

extern __device__ void keccakp(uint64_t * I, uint64_t *A);
extern __device__ void keccakp2(uint64_t * I, uint64_t * O, uint64_t * B);
extern __device__ void gpu_cipher_round(void* _block);

////////////////////////////////////////////////////////////////


#define FLAG_CMP(mask, flag) (!!((mask) & (val)))
#define FLAG_CMP_BIT(mask, flag) (-(FLAG_CMP(mask, flag)))
#define FLAG_CMP_BIT_NE(mask, flag) (-(!FLAG_CMP(mask, flag)))

#define MASK_CMP(mask, val) (!((mask) - (val)))
#define MASK_CMP64_BIT(mask, val) (-((uint64_t)MASK_CMP(mask, val)))
#define MASK_CMP32_BIT_NE(mask, val) (~(-((uint32_t)MASK_CMP(mask, val))))
#define MASK_CMP_BIT(mask, val) (-(MASK_CMP(mask, val)))
#define MASK_CMP_BIT32_SIGN(cmp_val, res) (((cmp_val << 1) - 1)*(res))
#define MASK_CMP_BIT64_SIGN(cmp_val, res) (((cmp_val << 1) - 1ULL)*(res))


#define INDEX3(X, Y, Z, i, j, k) ((i) * ((Z) * (Y)) + (j)*(Z) + (k))
#define INDEX2(X, Y, i, j) ((i)*(Y) + (j))


#define RET_ON_CUDA_FAIL(err) do{ if ((err) != cudaSuccess) goto cuda_free; }while(0)

////////////////////////////////////////////////////////////////

#define MAX_GPUS    32
#define MAX_STATES  64

struct cude_buffers{
    uint64_t *scratch_pad, *int_input;
    uint8_t *d_output;
    uint64_t *n_output;
    uint8_t *d_output_st2;
    uint64_t *n_output_st2;
    uint8_t *d_output_result;
    uint64_t *n_output_result;
    uint8_t *difficulty;
};
struct host_buffers{
    uint64_t nonce;
    uint8_t hash[HASH_SIZE];
};

struct runtime_state{
    struct cude_buffers buf[MAX_GPUS];
    struct host_buffers hbuf[MAX_GPUS];
};

struct run_config{
    /*size_t stage_1_threads;
    size_t stage_2_threads;
    size_t stage_3_threads;
    size_t stage_final_threads;*/

    size_t batch_size;
};

struct cuda_state{
    bool init;
    size_t batch_size;
    int num_gpus;
    int num_states;
    struct runtime_state states[MAX_STATES];
    struct run_config configs[MAX_GPUS];
};

#define NUM_THREADS_PER_SM 1536
#define NUM_BLOCKS_PER_SM 24
#define OPTIMAL_NUM_THREADS (NUM_THREADS_PER_SM / NUM_THREADS_PER_SM) //64

#define CUDA_WARP_SIZE 32
#define CUDA_HALF_WARP_SIZE 16
#define NUM_HASH_STATES 16

#define NUM_KECCAK_CHUNKS 2

#define NONCE_STEP 1


#define STAGE_GEN_THREADS KECCAK_WORDS
#define STAGE_0_WARP 2
#define KECCAK_THREADS 64
#define STAGE_1_WARP 16
#define STAGE_1_THREADS 8
#define STAGE_2_WARP 8
#define STAGE_2_THREADS 8
#define STAGE_3_WARP 8
#define STAGE_3_THREADS 32
#define STAGE_HASH_THREADS 64


#define RUN_GEN_HEADERS 1
#define RUN_STAGE_1     1
#define RUN_STAGE_2     1
#define RUN_STAGE_3     1
#define RUN_FIND_MIN    1

#define USE_PRINTF      0


#endif