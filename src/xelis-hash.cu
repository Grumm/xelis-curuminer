//xelis_hash.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "xelis-hash.cuh"

static struct cuda_state g_state;

////////////////////////////////////////////////////////////////


#if USE_PRINTF
__global__ void print_buffer(void *_buffer, size_t len, bool use_u8=false){
    if (use_u8){
        uint8_t *buffer = (uint8_t *)_buffer;
        for (size_t i = 0; i < len; i++){
            printf("%02x", buffer[i]);
        }
    } else {

        uint64_t *buffer = (uint64_t *)_buffer;
        for (size_t i = 0; i < len/8; i++){
            printf("%016lx", buffer[i]);
        }
    }
    printf("\n");
}

__host__ void print_buffer_host(void *_buffer, size_t len, bool use_u8=false){
    if (use_u8){
        uint8_t *buffer = (uint8_t *)_buffer;
        for (size_t i = 0; i < len; i++){
            printf("%02x", buffer[i]);
        }
    } else {

        uint64_t *buffer = (uint64_t *)_buffer;
        for (size_t i = 0; i < len/8; i++){
            printf("%016lx", buffer[i]);
        }
    }
    printf("\n");
}
#else
//__global__ void print_buffer(void *_buffer, size_t len, bool use_u8=false){ }
//__host__ void print_buffer_host(void *_buffer, size_t len, bool use_u8=false){ }
#endif

__device__ void kernel_memcpy(void *dst, const void *src, size_t len){
    for (size_t i = 0; i < len; i++){
        ((uint8_t *)dst)[i] = ((uint8_t *)src)[i];
    }
}

__device__ __forceinline__ void kernel_memcpy_u64(void *dst, const void *src, size_t len){
    for (size_t i = 0; i < len / sizeof(uint64_t); i++){
        ((uint64_t *)dst)[i] = ((uint64_t *)src)[i];
    }
}

__device__ __forceinline__ uint64_t rotate_left_u64(uint64_t result, uint32_t j) {
    //if (j == 0) return result;  + MASK_CMP(j, 0) * result
    return (result << j) | (result >> (64 - j));
}

__device__ void uint64_to_be_bytes(uint64_t num, uint8_t* bytes) {
    for (int i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = (num >> (56 - 8 * i)) & 0xFF;
    }
}

__device__ int kernel_memcmp(const uint8_t* hash1, const uint8_t* hash2, size_t size) {
    for (int i = 0; i < size; i++) {
        if (hash1[i] < hash2[i]) return -1;
        if (hash1[i] > hash2[i]) return 1;
    }
    return 0;
}


//////////////////////////////////////////////////////////////////////////

#define DEF_MEMCPY_ASYNC(SIZE) \
__device__ __forceinline__ void _memcpy_async##SIZE(void *dest, void *src){ \
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"((uint32_t)__cvta_generic_to_shared(dest)), "l"(src), "n"(SIZE)); \
}
#define DEF_MEMCPY_ASYNC_NOCACHE(SIZE) \
__device__ __forceinline__ void _memcpy_async_nocache##SIZE(void *dest, void *src){ \
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n" :: "r"((uint32_t)__cvta_generic_to_shared(dest)), "l"(src), "n"(SIZE)); \
}

#define DEF_MEMCPY_ASYNC_P(P_SIZE, SIZE) \
__device__ __forceinline__ void _memcpy_async_prefetch##SIZE##_##P_SIZE(void *dest, void *src){ \
    asm volatile("cp.async.ca.shared.global.L2::%0B [%1], [%2], %3;\n" :: \
        "n"(P_SIZE), "r"((uint32_t)__cvta_generic_to_shared(dest)), "l"(src), "n"(SIZE)); \
}

DEF_MEMCPY_ASYNC(16)
DEF_MEMCPY_ASYNC(8)
DEF_MEMCPY_ASYNC(4)
DEF_MEMCPY_ASYNC_NOCACHE(16)
DEF_MEMCPY_ASYNC_P(256, 16)
DEF_MEMCPY_ASYNC_P(256, 8)
DEF_MEMCPY_ASYNC_P(256, 4)

__device__ void memcpy_async32(void *_dest, void *_src){
    uint8_t * src = (uint8_t *)_src;
    uint8_t * dest = (uint8_t *)_dest;
    _memcpy_async16(&dest[0], &src[0]);
    _memcpy_async16(&dest[16], &src[16]);
}

__device__ void memcpy_async1024(void *_dest, void *_src){
    uint8_t * src = (uint8_t *)_src;
    uint8_t * dest = (uint8_t *)_dest;
    for(size_t i = 0; i < 64; i++){
        if ((i % 4) == 0)
            _memcpy_async_prefetch16_256(&dest[i * 16], &src[i * 16]);
        else
            _memcpy_async16(&dest[i * 16], &src[i * 16]);
    }
}

__device__ __forceinline__ void memcpy_async_wait(){
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ __forceinline__ void prefetch_L1(void *addr)
{
  asm("prefetch.global.L1 [%0];" : : "l"(addr));
}

__device__ __forceinline__ void prefetch_L2(void *addr)
{
  asm("prefetch.global.L2 [%0];" : : "l"(addr));
}

__device__ __forceinline__ void prefetchu_L1(void *addr)
{
  asm("prefetchu.global.L1 [%0];" : : "l"(addr));
}

__device__ __forceinline__ void prefetchu_L2(void *addr)
{
  asm("prefetchu.global.L2 [%0];" : : "l"(addr));
}

////////////////////////////////////////////////////////////////

#if RUN_STAGE_1

__device__ void stage_1_small(uint64_t * __restrict__ shared_int_input, uint64_t * __restrict__ shared_scratch_pad){
    uint64_t prev_v = 0;
    uint64_t new_v;
    for (size_t j = 0; j < KECCAK_WORDS; j++) {
        size_t pair_idx = (j + 1) % KECCAK_WORDS;
        size_t pair_idx2 = (j + 2) % KECCAK_WORDS;
        //first pass (1, 2) -> (0, 1)

        uint64_t left = shared_int_input[pair_idx];
        uint64_t right = shared_int_input[pair_idx2];
        uint64_t xor_val = left ^ right;
        uint64_t and_val = left & right;

        uint8_t mask = xor_val & 0x3;
        uint64_t v = ((~and_val) & MASK_CMP_BIT(mask, 1)) |
                     ((~xor_val) & MASK_CMP_BIT(mask, 2)) |
                     (xor_val & MASK_CMP_BIT(mask, 3)) |
                     (and_val & MASK_CMP_BIT(mask, 0));
        new_v = v ^ shared_int_input[j];

        shared_scratch_pad[j] = new_v ^ prev_v;
        prev_v = shared_scratch_pad[j];
    }
}

__global__ void stage_1_keccak(uint64_t *int_input, uint64_t *scratch_pad, size_t batch_size) {
    size_t idx = blockIdx.x;
    size_t atid = threadIdx.x;
    size_t tid = threadIdx.x % KECCAK_THREADS;
    size_t wid = threadIdx.x / KECCAK_THREADS;
    size_t num_threads = KECCAK_THREADS;
    size_t num_warps = STAGE_0_WARP;
    size_t global_idx = idx * num_threads + tid;

    if(global_idx >= batch_size)
        return;

    int_input = &int_input[global_idx * KECCAK_WORDS];
    uint64_t *b_scratch_pad = &scratch_pad[idx * num_threads * (MEMORY_SIZE + 2 * MEMORY_PAD)];
    scratch_pad = &scratch_pad[global_idx * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD];
    ////////////////////////////////////////////
    __shared__ uint64_t shared_int_input[KECCAK_THREADS * KECCAK_WORDS];
    uint64_t shared_int_input2[1 * KECCAK_WORDS];
    __shared__ uint64_t shared_int_input3[KECCAK_THREADS * KECCAK_WORDS];
    uint64_t shared_int_input4[1 * KECCAK_WORDS];
    uint32_t index;

    if(wid == 0){
        kernel_memcpy_u64(&shared_int_input4[0 * KECCAK_WORDS], &int_input[0 * KECCAK_WORDS], KECCAK_WORDS * sizeof(uint64_t));

        keccakp(&shared_int_input4[0 * KECCAK_WORDS], &shared_int_input2[0 * KECCAK_WORDS]);
        stage_1_small(&shared_int_input2[0 * KECCAK_WORDS], &shared_int_input3[tid * KECCAK_WORDS]);
    }

    {
        __syncthreads();
        if(wid == 0){
            keccakp(&shared_int_input2[0 * KECCAK_WORDS], &shared_int_input4[0 * KECCAK_WORDS]);
            stage_1_small(&shared_int_input4[0 * KECCAK_WORDS], &shared_int_input[tid * KECCAK_WORDS]);
        }

        if(wid == 1){
            index = 0;
            for (size_t _tid = 0; _tid < num_threads; _tid++){
                for(size_t j = tid; j < KECCAK_WORDS; j+= num_threads){
                    b_scratch_pad[_tid * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD + index * KECCAK_WORDS + j] = shared_int_input3[_tid * KECCAK_WORDS + j];
                }
            }
        }
    }
    for(size_t i = 1; i <= STAGE_1_MAX / 2; i++){
        {
            __syncthreads();
            if(wid == 0){
                keccakp(&shared_int_input4[0 * KECCAK_WORDS], &shared_int_input2[0 * KECCAK_WORDS]);
                stage_1_small(&shared_int_input2[0 * KECCAK_WORDS], &shared_int_input3[tid * KECCAK_WORDS]);
            }
            if(wid == 1 && i != 0){
                index = i * 2 - 1;
#if 1
                for (size_t _tid = 0; _tid < num_threads; _tid++){
                    for(size_t j = tid; j < KECCAK_WORDS; j+= num_threads){
                        b_scratch_pad[_tid * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD + index * KECCAK_WORDS + j] = shared_int_input[_tid * KECCAK_WORDS + j];
                    }
                }
#else
                kernel_memcpy_u64(&scratch_pad[index * KECCAK_WORDS], &shared_int_input[tid * KECCAK_WORDS], KECCAK_WORDS * sizeof(uint64_t));
#endif
            }
        }
        {
            __syncthreads();
            if(wid == 0){
                keccakp(&shared_int_input2[0 * KECCAK_WORDS], &shared_int_input4[0 * KECCAK_WORDS]);
                stage_1_small(&shared_int_input4[0 * KECCAK_WORDS], &shared_int_input[tid * KECCAK_WORDS]);
            }

            if(wid == 1){
                index = i * 2;
#if 1
                for (size_t _tid = 0; _tid < num_threads; _tid++){
                    for(size_t j = tid; j < KECCAK_WORDS; j+= num_threads){
                        b_scratch_pad[_tid * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD + index * KECCAK_WORDS + j] = shared_int_input3[_tid * KECCAK_WORDS + j];
                    }
                }
#else
                kernel_memcpy_u64(&scratch_pad[index * KECCAK_WORDS], &shared_int_input3[tid * KECCAK_WORDS], KECCAK_WORDS * sizeof(uint64_t));
#endif
            }
        }
    }
}
#endif

//////////////////////////////////////////////////////////////////////////

#if RUN_STAGE_2

__global__ void stage_2_kernel(uint64_t *scratch_pad, size_t batch_size) {
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x / STAGE_2_WARP; // 0->0, 1->0,...8->1, 9->1, ...16->2, 17->2...
    size_t wid = threadIdx.x % STAGE_2_WARP;
    size_t num_threads = blockDim.x / STAGE_2_WARP;
    size_t num_warps = STAGE_2_WARP;
    size_t global_idx = idx * num_threads + tid;

    if(global_idx >= batch_size)
        return;

    scratch_pad = &scratch_pad[global_idx * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD];
    /////////////////////////////////////
    __shared__ uint32_t slots[SLOT_LENGTH * STAGE_2_THREADS];
    __shared__ uint8_t indices[SLOT_LENGTH * STAGE_2_THREADS];
    __shared__ uint32_t shared_small_pad1[SLOT_LENGTH * STAGE_2_THREADS];
    uint32_t *small_pad = (uint32_t *)scratch_pad;

    uint32_t pad_len = MEMORY_SIZE * 2; // 2^16
    uint32_t num_slots = pad_len / SLOT_LENGTH; // 256

    prefetch_L1(&small_pad[0 * SLOT_LENGTH + wid * 16]);
    prefetch_L1(&small_pad[pad_len - SLOT_LENGTH + wid * 16]);

    #pragma unroll
    for(size_t i = wid; i < SLOT_LENGTH; i+=num_warps){
        slots[tid * SLOT_LENGTH + i] = small_pad[pad_len - SLOT_LENGTH + i];
    }

    //for (uint32_t iter = 0; iter < ITERS; iter++)
    {
        for (uint32_t j = 0; j < num_slots; j++) {
            __shared__ uint32_t _sum[STAGE_2_THREADS * STAGE_2_WARP];
            __shared__ uint32_t sum[STAGE_2_THREADS];

            _sum[tid * STAGE_2_WARP + wid] = 0;
            if(wid == 0)
                sum[tid] = 0;
            __syncthreads(); // slots, indices

            #pragma unroll 
            for(size_t i = wid; i < SLOT_LENGTH; i+=num_warps){
                indices[tid * SLOT_LENGTH + i] = i;
                shared_small_pad1[tid * SLOT_LENGTH + i] = small_pad[j * SLOT_LENGTH + i];
                _sum[tid * STAGE_2_WARP + wid] += MASK_CMP_BIT32_SIGN(MASK_CMP((slots[tid * SLOT_LENGTH + i] >> 31), 0), shared_small_pad1[tid * SLOT_LENGTH + i]);
            }
            atomicAdd(&sum[tid], _sum[tid * STAGE_2_WARP + wid]);
            /*if(wid == 0){
                for(size_t i = 0; i < num_warps; i++)
                    sum[tid] += _sum[tid * STAGE_2_WARP + i];
            }*/
            __syncthreads(); // sum
            if (wid == 0){
                #pragma unroll
                for (int16_t slot_idx = SLOT_LENGTH - 1; slot_idx >= 0; slot_idx--){
                    uint32_t index_in_indices = shared_small_pad1[tid * SLOT_LENGTH + slot_idx] % ((uint32_t)(slot_idx + 1));
                    uint16_t index = indices[tid * SLOT_LENGTH + index_in_indices];

                    indices[tid * SLOT_LENGTH + index_in_indices] = indices[tid * SLOT_LENGTH + slot_idx];

                    uint32_t pad = shared_small_pad1[tid * SLOT_LENGTH + index];
                    uint8_t old_flag = MASK_CMP((slots[tid * SLOT_LENGTH + index] >> 31), 0);
                    slots[tid * SLOT_LENGTH + index] += sum[tid] + MASK_CMP_BIT32_SIGN(!old_flag, pad);
                    uint8_t new_flag = MASK_CMP((slots[tid * SLOT_LENGTH + index] >> 31), 0);
                    uint32_t extra_sum = MASK_CMP_BIT32_SIGN(new_flag, pad) << 1;
                    sum[tid] += MASK_CMP32_BIT_NE(old_flag, new_flag) & extra_sum;
                }
            }
            prefetch_L2(&small_pad[(j + 2) * SLOT_LENGTH + (STAGE_2_WARP - wid) * 16]);
            prefetch_L1(&small_pad[(j + 1) * SLOT_LENGTH + (STAGE_2_WARP - wid) * 16]);
        }
    }

    __syncthreads();
    #pragma unroll
    for(size_t i = wid; i < SLOT_LENGTH; i+=num_warps){
        small_pad[pad_len - SLOT_LENGTH + i] = slots[tid * SLOT_LENGTH + i];
    }
}
#endif

////////////////////////////////////////////////////////////////

#if RUN_STAGE_3

__device__ __forceinline__ void aes_cipher_round(uint8_t *block) {
    gpu_cipher_round(block);
}

__device__ __forceinline__ uint64_t calc_hash(uint64_t* __restrict__ mem_buffer_a,
        uint64_t* __restrict__ mem_buffer_b, uint64_t result, uint16_t _i, uint16_t _j) {
    size_t tid = threadIdx.x / STAGE_3_WARP;
    uint16_t i = _i + _j;
    //#pragma unroll
    for (uint16_t j = 0; j < HASH_SIZE; j++) {
        uint64_t a = mem_buffer_a[tid * 1 + STAGE_3_THREADS * ((j + i) % BUFFER_SIZE)];
        uint64_t b = mem_buffer_b[tid * 1 + STAGE_3_THREADS * ((j + i) % BUFFER_SIZE)];

        uint8_t case_index = (result >> (j * 2)) & 0xf;
        uint64_t v = 0;

        switch (case_index) {
            case 0:  v = rotate_left_u64(result, j) ^ b; break;
            case 1:  v = ~(rotate_left_u64(result, j) ^ a); break;
            case 2:  v = ~(result ^ a); break;
            case 3:  v = result ^ b; break;
            case 4:  v = result ^ (a + b); break;
            case 5:  v = result ^ (a - b); break;
            case 6:  v = result ^ (b - a); break;
            case 7:  v = result ^ (a * b); break;
            case 8:  v = result ^ (a & b); break;
            case 9:  v = result ^ (a | b); break;
            case 10: v = result ^ (a ^ b); break;
            case 11: v = result ^ (a - result); break;
            case 12: v = result ^ (b - result); break;
            case 13: v = result ^ (a + result); break;
            case 14: v = result ^ (result - a); break;
            case 15: v = result ^ (result - b); break;
        }
        result = v;
    }

    return result;
}

__global__ void stage_3_kernel(uint64_t *scratch_pad, uint8_t *output, size_t batch_size) {
    size_t idx = blockIdx.x;
#if 1
    size_t tid = threadIdx.x / STAGE_3_WARP;
    size_t wid = threadIdx.x % STAGE_3_WARP;
    size_t num_threads = blockDim.x / STAGE_3_WARP;
    size_t num_warps = STAGE_3_WARP;
#else //different coalescing for threads and warps
    size_t tid = threadIdx.x % STAGE_3_THREADS;
    size_t wid = threadIdx.x / STAGE_3_THREADS;
    size_t num_threads = STAGE_3_THREADS;
    size_t num_warps = STAGE_3_WARP;
#endif
    size_t global_idx = idx * num_threads + tid;

    if(global_idx >= batch_size)
        return;

    scratch_pad = &scratch_pad[global_idx * (MEMORY_SIZE + 2 * MEMORY_PAD) + MEMORY_PAD];
    output = &output[global_idx * (HASH_SIZE)];
    /////////////////////////////////////

    uint64_t block[2 * BUFFER_SIZE];

    __shared__ uint64_t mem_buffer_a[BUFFER_SIZE * STAGE_3_THREADS];
    __shared__ uint64_t mem_buffer_b[BUFFER_SIZE * STAGE_3_THREADS];
    __shared__ uint64_t results[STAGE_3_THREADS * BUFFER_SIZE];
    uint64_t addr_a;
    uint64_t addr_b;

    addr_a = (scratch_pad[MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    addr_b = scratch_pad[MEMORY_SIZE - 1] & 0x7FFF;

    // Populate memory buffers
    for (uint64_t i = wid; i < BUFFER_SIZE; i+=num_warps) {
        mem_buffer_a[tid * 1 + STAGE_3_THREADS * i] = scratch_pad[(addr_a + i) % MEMORY_SIZE];
        mem_buffer_b[tid * 1 + STAGE_3_THREADS * i] = scratch_pad[(addr_b + i) % MEMORY_SIZE];
    }
    __syncthreads();
    //TODO try to parallelize the loop. maybe read the doc to find what is the operation
    uint64_t result;
    for (uint16_t i = 0; i < SCRATCHPAD_ITERS; i+=BUFFER_SIZE) {
        uint16_t num_j = min(BUFFER_SIZE, SCRATCHPAD_ITERS - i);

        for(uint16_t j = wid; j < num_j ; j+=num_warps){
            uint64_t mem_a = mem_buffer_a[tid * 1 + STAGE_3_THREADS * j];
            uint64_t mem_b = mem_buffer_b[tid * 1 + STAGE_3_THREADS * j];

            // Simulating block operations (big-endian handling might need adjustments)
            block[2 * j + 0] = mem_b;
            block[2 * j + 1] = mem_a;
            aes_cipher_round((uint8_t *)&block[2 * j]);  // Assuming 'key' is available

            results[tid * BUFFER_SIZE + j] = block[2 * j + 0] ^ ~(mem_a ^ mem_b);
        }
        __syncthreads();
        if (wid == 0)
        for(uint16_t j = 0; j < num_j ; j++){

            result = calc_hash(mem_buffer_a, mem_buffer_b, results[tid * BUFFER_SIZE + j], i, j);


            addr_b = result & 0x7FFF;
            {
                mem_buffer_a[tid * 1 + STAGE_3_THREADS * j] = result;
                //TODO move in another thread
                mem_buffer_b[tid * 1 + STAGE_3_THREADS * j] = scratch_pad[addr_b];
            }

            addr_a = (result >> 15ULL) & 0x7FFFULL;
            //TODO move in another thread
            scratch_pad[addr_a] = result;

            int64_t index = SCRATCHPAD_ITERS - i - j - 1;
            if (__builtin_expect(index < 4 && i >= 0, 0)) {
                uint64_to_be_bytes(result, &output[index * sizeof(uint64_t)]);
            }
        }
    }
}
#endif


////////////////////////////////////////////////////////////////

static void process_hash(int state, int gpu_id, size_t batch_size, cudaStream_t stream){
    struct run_config *config = &g_state.configs[gpu_id];
    cudaSetDevice(gpu_id);
#if RUN_STAGE_1
    /*
        k = input
        at iter i(0..1309), (1310):
            k = keccak(k)
            at iter j(0..24), (0..18):
                shared_v[j] = f(k[j+1], k[j+2]) ^ k[j]
                {shared_scratch_pad[j] = shared_v[j]^shared_v[j-1]^.....shared_v[0]
                {shared_scratch_pad[j] = f(k[j+1], k[j+2]) ^ k[j]^f(k[j], k[j+1]) ^ k[j - 1]^.....f(k[1], k[2]) ^ k[0]
                {shared_scratch_pad[j] = (k[0]^...^k[j])^(f(k[1], k[2])^...^f(k[j+1], k[j+2]))
                scratch_pad[i * KW + j] = shared_scratch_pad[j]
    */
    {
        stage_1_keccak<<<batch_size/KECCAK_THREADS, KECCAK_THREADS * STAGE_0_WARP, 0, stream>>>(
            g_state.states[state].buf[gpu_id].int_input,
            g_state.states[state].buf[gpu_id].scratch_pad,
            batch_size);
    }
#endif
    /* TODO - stage 12
        #first, get slots(scratch_pad[-SL/2..]), and scratch_pad[..SL/2]
        #then do j=0 iteration, while calculating scratch_pad[SL/2..SL]
        #then do i=1 iteration, while doing scratch_pad[SL..SL*3/2]
        #...
        #until i=255 iteration, while not doing scratch_pad
        #copy slots to scratch_pad[-SL..]

        index = get_index(start)

        doing calc_scratch_pad_iteration(index):
            k = keccak_precalculated[index*KW]
            offset = get_offset(index)
            scratch_pad[offset + j] = (k[0]^...^k[j])^(f(k[1], k[2])^...^f(k[j+1], k[j+2])
            #i * KW + j =(0...32749), (32750..32767)

        doing f(a, b):
            xor_val = a^b
            and_val = a&b
            mask = xor_val & 0x3;
            v = ((~and_val) & MASK_CMP_BIT(mask, 1)) |
                         ((~xor_val) & MASK_CMP_BIT(mask, 2)) |
                         (xor_val & MASK_CMP_BIT(mask, 3)) |
                         (and_val & MASK_CMP_BIT(mask, 0));
    */
#if RUN_STAGE_2
    {
    /*
        small_pad[i] = scratch_pad[i]
        slots[i] = small_pad[-SL + i] #-128 in scratch_pad, -256 in small_pad
        at iter j(0..256):
            shared_pad[i]=small_pad[j*SL+i]
            indices[i]=i
            at iter s(SL..0):
                ii = shared_pad[s]%(s+1)
                i = indices[ii]
                slots[i] = sum(k, f(slots(k!=i)) * shared_pad[k])
                indices[ii] = indices[s]
        small_pad[-SL + i] = slots[i]
    */

        stage_2_kernel<<<batch_size / STAGE_2_THREADS, STAGE_2_THREADS * STAGE_2_WARP, 0, stream>>>(
            g_state.states[state].buf[gpu_id].scratch_pad, batch_size);
    }
#endif
#if RUN_STAGE_3
    {
    /*
        addr_a = scratch_pad[-1].upper
        addr_b = scratch_pad[-1].lower
        mem_buffer_a[i] = scratch_pad[addr_a + i]
        mem_buffer_b[i] = scratch_pad[addr_b + i]
        at iter i(0..5K):
            at iter j(0..BS):
                ar = aes(mem_buffer_b[j]+mem_buffer_a[j])[0]
                r = ar ^ ~(mem_buffer_b[j]^mem_buffer_a[j])
                at iter h(0..32):
                    r = f(r, mem_buffer_a[h + i], mem_buffer_a[h + i])
                mem_buffer_a[j] = r
                mem_buffer_b[j] = scratch_pad[r.lower]
                scratch_pad[r.upper] = r
                last 4: output[k] = r
    */
        stage_3_kernel<<<batch_size / STAGE_3_THREADS, STAGE_3_THREADS * STAGE_3_WARP, 0, stream>>>(
            g_state.states[state].buf[gpu_id].scratch_pad,
            g_state.states[state].buf[gpu_id].d_output,
            batch_size);

    }
#endif
}

#if RUN_GEN_HEADERS

__global__ void generate_headers_kernel(uint64_t *int_input, uint64_t *n_output,
        uint64_t nonce_start, uint64_t nonce_end){
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;

    uint64_t *ref_header = int_input;
    uint64_t *target_header = &int_input[idx * KECCAK_WORDS];

    target_header[tid] = ref_header[tid];
    __syncthreads();
    if(tid == 0){
        uint64_t target_nonce = nonce_start + idx;
        //        writer.write_u64(&self.nonce); // 40 + 8 = 48
        target_header[5] = target_nonce;
        n_output[idx] = target_nonce;
    }
}

static void generate_headers(int state, int gpu_id, uint64_t *nonce_start,
        size_t batch_size, cudaStream_t stream){
    uint64_t nonce_per_gpu = batch_size * NONCE_STEP;

    {
        uint64_t nonce_end = *nonce_start + nonce_per_gpu;

        generate_headers_kernel<<<batch_size, STAGE_GEN_THREADS, 0, stream>>>(
            g_state.states[state].buf[gpu_id].int_input, g_state.states[state].buf[gpu_id].n_output,
            *nonce_start, nonce_end);
        *nonce_start = nonce_end;
    }
}
#endif

#if RUN_FIND_MIN

__global__ void find_min_hash_kernel_2step_1block(
        uint8_t *hashes_in, uint64_t *nonces_in,
        uint8_t *hashes_out, uint64_t *nonces_out,
        uint8_t *difficulty, size_t batch_size){
    size_t idx = blockIdx.x;
    size_t tid = threadIdx.x;
    size_t num_blocks = gridDim.x;
    size_t num_threads = blockDim.x;
    size_t reduce_to = 1;

    uint8_t *hash = &hashes_in[idx * batch_size * HASH_SIZE];
    uint64_t *nonce = &nonces_in[idx * batch_size];
    uint8_t *hash_o = &hashes_out[idx * reduce_to * HASH_SIZE];
    uint64_t *nonce_o = &nonces_out[idx * reduce_to];

    __shared__ size_t found_ind[STAGE_HASH_THREADS];
    __shared__ uint64_t found_nonc[STAGE_HASH_THREADS];

    found_ind[tid] = tid;
    found_nonc[tid] = nonce[tid];

    for(size_t i = tid; i < batch_size; i+=num_threads){
        if (kernel_memcmp(&hash[i * HASH_SIZE], &hash[found_ind[tid] * HASH_SIZE], HASH_SIZE) < 0){
            found_ind[tid] = i;
            found_nonc[tid] = nonce[i];
        }
    }

    __syncthreads();

    if (tid == 0) {
        //found_ind[i]
        size_t min_index = 0;
        for(size_t i = 0; i < STAGE_HASH_THREADS; i++){
            if(kernel_memcmp(&hash[found_ind[i] * HASH_SIZE], &hash[found_ind[min_index] * HASH_SIZE], HASH_SIZE) < 0){
                min_index = i;
            }
        }
        kernel_memcpy_u64(hash_o, &hash[found_ind[min_index] * HASH_SIZE], HASH_SIZE);
        nonce_o[0] = found_nonc[min_index];
    }
}

static void find_min_hash(int state, int gpu_id, size_t batch_size, cudaStream_t stream){
    struct run_config *config = &g_state.configs[gpu_id];
    {
        find_min_hash_kernel_2step_1block<<<1, STAGE_HASH_THREADS, 0, stream>>>(
            g_state.states[state].buf[gpu_id].d_output,
            g_state.states[state].buf[gpu_id].n_output,
            g_state.states[state].buf[gpu_id].d_output_result,
            g_state.states[state].buf[gpu_id].n_output_result,
            g_state.states[state].buf[gpu_id].difficulty, batch_size);
    }
}
#endif

extern "C" {

int xelis_hash_cuda(const uint8_t *input, size_t total_batch_size, uint8_t *output, int state) {
    int num_gpus = g_state.num_gpus;
    if (total_batch_size > g_state.batch_size * g_state.num_gpus){
        return -1;
    }
    if (!g_state.init){
        return -2;
    }

    size_t batch_size = total_batch_size / num_gpus;
    cudaStream_t streams[num_gpus];


    size_t offset = 0;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaStreamCreate(&streams[gpu_id]);

        cudaMemcpyAsync(g_state.states[state].buf[gpu_id].int_input, &input[offset * KECCAK_WORDS * sizeof(uint64_t)],
            batch_size * KECCAK_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice,
            streams[gpu_id]);
        offset += batch_size;
    }

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        process_hash(state, gpu_id, batch_size, streams[gpu_id]);
    }

    offset = 0;
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaStreamSynchronize(streams[gpu_id]);

        cudaMemcpyAsync(&output[offset * HASH_SIZE], g_state.states[state].buf[gpu_id].d_output, 
            batch_size * HASH_SIZE, cudaMemcpyDeviceToHost,
            streams[gpu_id]);
        offset += batch_size;
    }
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaStreamSynchronize(streams[gpu_id]);
    }
    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        cudaStreamDestroy(streams[gpu_id]);
    }

    return 0;
}

int xelis_hash_cuda_nonce(const uint8_t *base_header, uint64_t *nonce_start,
        size_t batch_size, uint8_t *output_hash,
        uint64_t *output_nonce,
        const uint8_t *difficulty, int gpu_id, int state){

    int num_gpus = g_state.num_gpus;
    if (!g_state.init){
        return -1;
    }
    if (batch_size > g_state.batch_size){
        return -2;
    }
    if (state > g_state.num_states){
        return -3;
    }
    cudaStream_t stream;

    {
        cudaSetDevice(gpu_id);
        cudaStreamCreate(&stream);

        cudaMemcpyAsync(g_state.states[state].buf[gpu_id].int_input, base_header,
            KECCAK_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice,
            stream);
    }
#if RUN_GEN_HEADERS
    generate_headers(state, gpu_id, nonce_start, batch_size, stream);
#endif
    process_hash(state, gpu_id, batch_size, stream);
#if RUN_FIND_MIN
    find_min_hash(state, gpu_id, batch_size, stream);
#endif

    {
        cudaMemcpyAsync(output_hash,
            g_state.states[state].buf[gpu_id].d_output_result, HASH_SIZE,
            cudaMemcpyDeviceToHost, stream);
        cudaMemcpyAsync(output_nonce,
            g_state.states[state].buf[gpu_id].n_output_result, sizeof(uint64_t),
            cudaMemcpyDeviceToHost, stream);
    }
    {
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }
    return 0;
}

int deinitialize_cuda(void);

int initialize_cuda(size_t batch_size, int num_states){
    cudaError_t err;
    if (g_state.init){
        return g_state.num_gpus;
    }

    int num_gpus;
    err = cudaGetDeviceCount(&num_gpus);
    RET_ON_CUDA_FAIL(err);

    if (num_gpus < 1) {
        return 0;
    }
    memset(&g_state, 0, sizeof(struct cuda_state));

    g_state.batch_size = batch_size;
    g_state.num_gpus = num_gpus;
    g_state.num_states = num_states;
    g_state.init = true;

    for (int gpu_id = 0; gpu_id < num_gpus; ++gpu_id) {
        err = cudaSetDevice(gpu_id);
        RET_ON_CUDA_FAIL(err);
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, gpu_id);
        RET_ON_CUDA_FAIL(err);
        printf("Initializing cuda device\n");
        printf("Shared memory: %ldKB\n", prop.sharedMemPerBlock/1024);
        printf("SMs: %d Threads: %d Blocks: %d\n",
            prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor, prop.maxBlocksPerMultiProcessor);
        printf("Threads/Block: %d Warp: %d Cores per SM: %d\n",
            prop.maxThreadsPerBlock, prop.warpSize,
            prop.maxThreadsPerMultiProcessor / prop.warpSize);
        for (int state = 0; state < num_states; state++){
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].scratch_pad, batch_size * (MEMORY_SIZE + 2 * MEMORY_PAD) * sizeof(uint64_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].int_input, batch_size * KECCAK_WORDS * sizeof(uint64_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].d_output, batch_size * HASH_SIZE * sizeof(uint8_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].n_output, batch_size * sizeof(uint64_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].d_output_result, HASH_SIZE * sizeof(uint8_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].n_output_result, sizeof(uint64_t));
            RET_ON_CUDA_FAIL(err);
            err = cudaMalloc(&g_state.states[state].buf[gpu_id].difficulty, HASH_SIZE * sizeof(uint8_t));
            RET_ON_CUDA_FAIL(err);
        }

        /*g_state.configs[gpu_id].stage_1_threads = STAGE_1_THREADS;
        g_state.configs[gpu_id].stage_2_threads = STAGE_2_THREADS;
        g_state.configs[gpu_id].stage_3_threads = STAGE_3_THREADS;
        g_state.configs[gpu_id].stage_final_threads = STAGE_HASH_THREADS;*/
        g_state.configs[gpu_id].batch_size = batch_size;
    }
    return num_gpus;
cuda_free:
    deinitialize_cuda();
    return -1;
}

int deinitialize_cuda(void){
    if (!g_state.init){
        return 0;
    }

    for (int gpu_id = 0; gpu_id < g_state.num_gpus; ++gpu_id) {
        cudaSetDevice(gpu_id);
        for (int state = 0; state < g_state.num_states; state++){
            cudaFree(g_state.states[state].buf[gpu_id].scratch_pad);
            cudaFree(g_state.states[state].buf[gpu_id].int_input);
            cudaFree(g_state.states[state].buf[gpu_id].d_output);
            cudaFree(g_state.states[state].buf[gpu_id].n_output);
            cudaFree(g_state.states[state].buf[gpu_id].d_output_result);
            cudaFree(g_state.states[state].buf[gpu_id].n_output_result);
            cudaFree(g_state.states[state].buf[gpu_id].difficulty);
        }
    }
    memset(&g_state, 0, sizeof(struct cuda_state));
    return 0;
}

}




