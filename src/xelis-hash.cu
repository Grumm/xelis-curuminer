//xelis_hash.cu

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define KECCAK_WORDS 25
#define MEMORY_SIZE 32768
#define BYTES_ARRAY_INPUT KECCAK_WORDS * 8
#define SLOT_LENGTH 256
#define HASH_SIZE 32
#define BUFFER_SIZE 42
#define STAGE_1_MAX (MEMORY_SIZE / KECCAK_WORDS)

#define ITERS 1
#define SCRATCHPAD_ITERS 5000

//extern __device__ void keccak(const char *message, int message_len, unsigned char *output, int output_len);
//extern __device__ void Keccak(char *input, int size, uint8_t* state);
extern __device__ void keccakp(uint64_t *A);
//extern __device__ void encrypt(uint8_t *block, uint8_t *rkey, uint32_t offset);
extern __device__ void gpu_cipher_round(void* _block, void* _expandedkeys);

////////////////////////////////////////////////////////////////

//#define USE_PRINTF

#ifdef USE_PRINTF
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
#else
__global__ void print_buffer(void *_buffer, size_t len, bool use_u8=false){ }
#endif

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
    if (j == 0) return result;
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


////////////////////////////////////////////////////////////////

__device__ __forceinline__ void keccakp_kernel(uint64_t *data){
	//return keccak((char *)data, BYTES_ARRAY_INPUT, (unsigned char*)data, BYTES_ARRAY_INPUT);
	//Keccak((char *)data, BYTES_ARRAY_INPUT, (uint8_t *)data);
	keccakp(data);
}

__device__ void stage_1_impl(uint64_t *shared_int_input, uint64_t *scratch_pad, int a0, int a1, int b0, int b1, size_t tid, size_t num_threads) {
    __shared__ uint64_t shared_scratch_pad[STAGE_1_MAX];
    __shared__ uint64_t shared_v[STAGE_1_MAX];

    for (size_t i = a0; i <= a1; i++){
        if (tid == 0)
            keccakp_kernel(shared_int_input);
        if (tid == 1 && i != a0)
        	kernel_memcpy_u64(&scratch_pad[(i - 1) * KECCAK_WORDS + b0], &shared_scratch_pad[b0], sizeof(uint64_t) * (b1 - b0 + 1));

        __syncthreads();

        for (size_t j = b0 + tid; j <= b1; j+=num_threads) {
            size_t pair_idx = (j + 1) % KECCAK_WORDS;
            size_t pair_idx2 = (j + 2) % KECCAK_WORDS;

            uint64_t left = shared_int_input[pair_idx];
            uint64_t right = shared_int_input[pair_idx2];
            uint64_t xor_val = left ^ right;

            uint64_t v = 0;
            switch (xor_val & 0x3) {
                case 0:
                    v = left & right;
                    break;
                case 1:
                    v = ~(left & right);
                    break;
                case 2:
                    v = ~xor_val;
                    break;
                case 3:
                    v = xor_val;
                    break;
            }
            shared_v[j] = v;
        }
    	__syncthreads();
        if (tid == 1){
	        uint64_t rand_int = 0;
	        for (size_t j = b0; j <= b1; j++) {
	            uint64_t a_val = shared_int_input[j] ^ rand_int;
	            uint64_t b_val = a_val ^ shared_v[j];
	            rand_int = b_val;

				shared_scratch_pad[j] = b_val;
				//scratch_pad[i * KECCAK_WORDS + j] = b_val;
	        }
	    }
    }
    __syncthreads();
    if (tid == 0)
    	kernel_memcpy_u64(&scratch_pad[a1 * KECCAK_WORDS + b0], &shared_scratch_pad[b0], sizeof(uint64_t) * (b1 - b0 + 1));

    __syncthreads();
}

__global__ void stage_1(uint64_t *int_input, uint64_t *scratch_pad) {
	size_t idx = blockIdx.x;
	size_t tid = threadIdx.x;
    size_t num_threads = blockDim.x;

	int_input = &int_input[idx * (KECCAK_WORDS)];
	scratch_pad = &scratch_pad[idx * (MEMORY_SIZE)];
	/////////////////////////////////////
    __shared__ uint64_t shared_int_input[KECCAK_WORDS];
    for (size_t i = tid; i < KECCAK_WORDS; i += num_threads) {
        shared_int_input[i] = int_input[i];
    }
    __syncthreads();

    stage_1_impl(shared_int_input, scratch_pad, 0, STAGE_1_MAX - 1, 0, KECCAK_WORDS - 1, tid, num_threads);
    if (tid > 18) return;
    stage_1_impl(shared_int_input, scratch_pad, STAGE_1_MAX, STAGE_1_MAX, 0, 17, tid, min(num_threads, 18UL));
}

////////////////////////////////////////////////////////////////

__global__ void stage2_kernel(uint64_t *scratch_pad) {
	size_t idx = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t num_threads = blockDim.x;

	scratch_pad = &scratch_pad[idx * (MEMORY_SIZE)];
	/////////////////////////////////////
    __shared__ uint32_t slots[SLOT_LENGTH];
    __shared__ uint16_t indices[SLOT_LENGTH];
	__shared__ uint32_t shared_small_pad[SLOT_LENGTH];
    uint32_t *small_pad = (uint32_t *)scratch_pad;

    uint32_t pad_len = MEMORY_SIZE * 2; // 2^16
    uint32_t num_slots = pad_len / SLOT_LENGTH; // 256

	#pragma unroll 16
    for(size_t i = tid; i < SLOT_LENGTH; i+=num_threads){
    	slots[i] = small_pad[pad_len - SLOT_LENGTH + i];
		shared_small_pad[i] = small_pad[i];
    }

    //for (uint32_t iter = 0; iter < ITERS; iter++)
    {
        for (uint32_t j = 0; j < num_slots; j++) {
			#pragma unroll 8
		    for(size_t i = tid; i < SLOT_LENGTH; i+=num_threads){
		    	indices[i] = i;
	        	shared_small_pad[i] = small_pad[j * SLOT_LENGTH + i];
	    	}
        	__syncthreads(); // shared_small_pad, indices

    		#pragma unroll 16
    		for (int16_t slot_idx = SLOT_LENGTH - 1; slot_idx >= 0; slot_idx--){
                uint32_t index_in_indices = shared_small_pad[slot_idx] % ((uint32_t)(slot_idx + 1));
                uint16_t index = indices[index_in_indices];

            	//__syncthreads(); //needed for slots and index updates

                if (tid == 0){
                	indices[index_in_indices] = indices[slot_idx];
                }

                uint32_t sum = 0;
    			#pragma unroll 8
                for (uint16_t k = tid; k < SLOT_LENGTH; k+=num_threads) {
			        uint32_t pad = !!(k != index) * shared_small_pad[k];
			        if ((slots[k] >> 31) == 0) {
			            sum += pad;
			        } else {
			            sum -= pad;
			        }
			    }
                atomicAdd(&slots[index], sum);
                //slots[index] += sum;
            }
        }
    }

    __syncthreads();
    for(size_t i = tid; i < SLOT_LENGTH; i+=num_threads){
    	small_pad[pad_len - SLOT_LENGTH + i] = slots[i];
    }
}

////////////////////////////////////////////////////////////////

__device__ __forceinline__ void aes_cipher_round(uint8_t *block, uint8_t *key) {
    //encrypt(block, key, 0);
    gpu_cipher_round(block, key);
}

__device__ __forceinline__ uint64_t calc_hash(uint64_t* __restrict__ mem_buffer_a, uint64_t* __restrict__ mem_buffer_b, uint64_t result, uint64_t i) {
	//#pragma unroll 8
    for (uint32_t j = 0; j < HASH_SIZE; j++) {
        uint64_t a = mem_buffer_a[(j + i) % BUFFER_SIZE];
        uint64_t b = mem_buffer_b[(j + i) % BUFFER_SIZE];

        // Determine operation based on the condition
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

        result = v; // Update the result
    }

    return result;
}

__global__ void stage3_kernel(uint64_t *scratch_pad, uint8_t *output) {
	size_t idx = blockIdx.x;
	size_t tid = threadIdx.x;
	size_t num_threads = blockDim.x;

	scratch_pad = &scratch_pad[idx * (MEMORY_SIZE)];
	output = &output[idx * (HASH_SIZE)];
	/////////////////////////////////////

    __shared__ uint64_t block[2];
    __shared__ uint64_t key[2];

    __shared__ uint64_t mem_buffer_a[BUFFER_SIZE];
    __shared__ uint64_t mem_buffer_b[BUFFER_SIZE];
    __shared__ uint64_t addr_a;
    __shared__ uint64_t addr_b;

    addr_a = (scratch_pad[MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    addr_b = scratch_pad[MEMORY_SIZE - 1] & 0x7FFF;

    // Populate memory buffers
    for (uint64_t i = tid; i < BUFFER_SIZE; i+=num_threads) {
        mem_buffer_a[i] = scratch_pad[(addr_a + i) % MEMORY_SIZE];
        mem_buffer_b[i] = scratch_pad[(addr_b + i) % MEMORY_SIZE];
    }
    __syncthreads();

    if(tid == 0){
    	key[0] = 0;
    	key[1] = 0;
	    uint64_t result;
	    for (uint64_t i = 0; i < SCRATCHPAD_ITERS; i++) {
	        uint64_t mem_a = mem_buffer_a[i % BUFFER_SIZE];
	        uint64_t mem_b = mem_buffer_b[i % BUFFER_SIZE];

	        // Simulating block operations (big-endian handling might need adjustments)
	        block[0] = mem_b;
	        block[1] = mem_a;
	    	aes_cipher_round((uint8_t *)block, (uint8_t *)key);  // Assuming 'key' is available

	        uint64_t hash1 = block[0]; // Assuming block[0..8] is directly readable
	        uint64_t hash2 = mem_a ^ mem_b;

	        result = ~(hash1 ^ hash2);

	        result = calc_hash(mem_buffer_a, mem_buffer_b, result, i);

	        addr_b = result & 0x7FFF;
	        mem_buffer_a[i % BUFFER_SIZE] = result;
	        mem_buffer_b[i % BUFFER_SIZE] = scratch_pad[addr_b];

	        addr_a = (result >> 15ULL) & 0x7FFFULL;
	        scratch_pad[addr_a] = result;

	        int64_t index = SCRATCHPAD_ITERS - i - 1;
	        if (index < 4) {
	        	uint64_to_be_bytes(result, &output[index * sizeof(uint64_t)]);
	        }
	    }
	}
}

////////////////////////////////////////////////////////////////
/*
__global__ void xelis_hash_cuda_impl(uint64_t *scratch_pad, uint64_t *int_input, uint16_t *ref_indices, uint8_t *d_output) {
	size_t idx = blockIdx.x;


	int_input = &int_input[idx * (KECCAK_WORDS)];	
	scratch_pad = &scratch_pad[idx * (MEMORY_SIZE)];
	d_output = &d_output[idx * (HASH_SIZE)];

    stage_1_impl(int_input, scratch_pad, 0, STAGE_1_MAX - 1, 0, KECCAK_WORDS - 1);
    stage_1_impl(int_input, scratch_pad, STAGE_1_MAX, STAGE_1_MAX, 0, 17); // Only 18 threads for this one

    stage2_kernel(scratch_pad, ref_indices);
    stage3_kernel(scratch_pad, d_output);
}*/

////////////////////////////////////////////////////////////////

#define MAX_GPUS	32
#define MAX_STATES	64

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
	size_t stage_1_threads;
	size_t stage_2_threads;
	size_t stage_3_threads;
	size_t stage_final_threads;

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

static struct cuda_state g_state;

#define STAGE_1_THREADS 128
#define STAGE_2_THREADS 32
#define STAGE_3_THREADS 16
#define NUM_HASH_THREADS 64

static void process_hash(int state, int gpu_id, size_t batch_size, cudaStream_t stream){
	struct run_config *config = &g_state.configs[gpu_id];
	cudaSetDevice(gpu_id);
    {
	    stage_1<<<batch_size, config->stage_1_threads, 0, stream>>>(
	    	g_state.states[state].buf[gpu_id].int_input,
	    	g_state.states[state].buf[gpu_id].scratch_pad);
	}
	{
		size_t shared_size = SLOT_LENGTH * (sizeof(uint32_t) * 2 + sizeof(uint16_t));
	    stage2_kernel<<<batch_size, config->stage_2_threads, shared_size, stream>>>(
	    	g_state.states[state].buf[gpu_id].scratch_pad);
	}
	{
	    stage3_kernel<<<batch_size, config->stage_3_threads, 0, stream>>>(
	    	g_state.states[state].buf[gpu_id].scratch_pad,
	    	g_state.states[state].buf[gpu_id].d_output);
	}
}

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

	__shared__ size_t found_ind[NUM_HASH_THREADS];
	__shared__ uint64_t found_nonc[NUM_HASH_THREADS];

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
		for(size_t i = 0; i < NUM_HASH_THREADS; i++){
			if(kernel_memcmp(&hash[found_ind[i] * HASH_SIZE], &hash[found_ind[min_index] * HASH_SIZE], HASH_SIZE) < 0){
				min_index = i;
			}
		}
		kernel_memcpy_u64(hash_o, &hash[found_ind[min_index] * HASH_SIZE], HASH_SIZE);
		nonce_o[0] = found_nonc[min_index];
	}
}

__global__ void find_min_hash_kernel_naive(
		uint8_t *hashes_in, uint64_t *nonces_in,
		uint8_t *hashes_out, uint64_t *nonces_out,
		uint8_t *difficulty, size_t batch_size){
	size_t reduce_to = 1;
	size_t idx = 0;

	uint8_t *hash = &hashes_in[idx * batch_size * HASH_SIZE];
	uint64_t *nonce = &nonces_in[idx * batch_size];
	uint8_t *hash_o = &hashes_out[idx * reduce_to * HASH_SIZE];
	uint64_t *nonce_o = &nonces_out[idx * reduce_to];

	size_t min_index = 0;
	for(size_t i = 0; i < batch_size; i++){
		if(kernel_memcmp(&hash[i * HASH_SIZE], &hash[min_index * HASH_SIZE], HASH_SIZE) < 0){
			min_index = i;
		}
	}
	kernel_memcpy_u64(hash_o, &hash[min_index * HASH_SIZE], HASH_SIZE);
	nonce_o[0] = nonce[min_index];
}

#define NONCE_STEP 1

static void generate_headers(int state, int gpu_id, uint64_t *nonce_start,
		size_t batch_size, cudaStream_t stream){
	uint64_t nonce_per_gpu = batch_size * NONCE_STEP;

	{
    	uint64_t nonce_end = *nonce_start + nonce_per_gpu;

    	generate_headers_kernel<<<batch_size, KECCAK_WORDS, 0, stream>>>(
    		g_state.states[state].buf[gpu_id].int_input, g_state.states[state].buf[gpu_id].n_output,
    		*nonce_start, nonce_end);
    	*nonce_start = nonce_end;
    }
}

static void find_min_hash(int state, int gpu_id, size_t batch_size, cudaStream_t stream){
	struct run_config *config = &g_state.configs[gpu_id];
#if 1
	{
    	find_min_hash_kernel_2step_1block<<<1, config->stage_final_threads, 0, stream>>>(
    		g_state.states[state].buf[gpu_id].d_output,
    		g_state.states[state].buf[gpu_id].n_output,
    		g_state.states[state].buf[gpu_id].d_output_result,
    		g_state.states[state].buf[gpu_id].n_output_result,
    		g_state.states[state].buf[gpu_id].difficulty, batch_size);
    }
#else
    {
    	find_min_hash_kernel<<<batch_size/NUM_HASH_THREADS, NUM_HASH_THREADS, 0, stream>>>(
    		g_state.states[state].buf[gpu_id].d_output,
    		g_state.states[state].buf[gpu_id].n_output,
    		g_state.states[state].buf[gpu_id].d_output_st2,
    		g_state.states[state].buf[gpu_id].n_output_st2,
    		g_state.states[state].buf[gpu_id].difficulty);
    }
    {
    	find_min_hash_kernel<<<1, batch_size/NUM_HASH_THREADS, 0, stream>>>(
    		g_state.states[state].buf[gpu_id].d_output_st2,
    		g_state.states[state].buf[gpu_id].n_output_st2,
    		g_state.states[state].buf[gpu_id].d_output_result,
    		g_state.states[state].buf[gpu_id].n_output_result,
    		g_state.states[state].buf[gpu_id].difficulty);
    }
#endif
}

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
#if 0
	    cudaMemcpyAsync(g_state.states[state].buf[gpu_id].difficulty, difficulty,
	    	HASH_SIZE, cudaMemcpyHostToDevice,
	    	stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].int_input,
			sizeof(uint64_t) * batch_size * KECCAK_WORDS,
			gpu_id, stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].n_output,
			sizeof(uint64_t) * batch_size,
			gpu_id, stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].scratch_pad,
			sizeof(uint64_t) * batch_size * MEMORY_SIZE,
			gpu_id, stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].d_output,
			sizeof(uint64_t) * batch_size * MEMORY_SIZE,
			gpu_id, stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].d_output_result,
			HASH_SIZE,
			gpu_id, stream);
		cudaMemPrefetchAsync(g_state.states[state].buf[gpu_id].n_output_result,
			sizeof(uint64_t),
			gpu_id, stream);
#endif
    }

    generate_headers(state, gpu_id, nonce_start, batch_size, stream);
    process_hash(state, gpu_id, batch_size, stream);
    find_min_hash(state, gpu_id, batch_size, stream);

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

#define RET_ON_CUDA_FAIL(err) do{ if ((err) != cudaSuccess) goto cuda_free; }while(0)

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
    	for (int state = 0; state < num_states; state++){
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].scratch_pad, batch_size * MEMORY_SIZE * sizeof(uint64_t));
		    RET_ON_CUDA_FAIL(err);
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].int_input, batch_size * KECCAK_WORDS * sizeof(uint64_t));
		    RET_ON_CUDA_FAIL(err);
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].d_output, batch_size * HASH_SIZE * sizeof(uint8_t));
		    RET_ON_CUDA_FAIL(err);
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].n_output, batch_size * sizeof(uint64_t));
		    RET_ON_CUDA_FAIL(err);
		    //cudaMalloc(&g_state.states[state].buf[gpu_id].d_output_st2, batch_size * HASH_SIZE * sizeof(uint8_t));
		    //cudaMalloc(&g_state.states[state].buf[gpu_id].n_output_st2, batch_size * sizeof(uint64_t));
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].d_output_result, HASH_SIZE * sizeof(uint8_t));
		    RET_ON_CUDA_FAIL(err);
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].n_output_result, sizeof(uint64_t));
		    RET_ON_CUDA_FAIL(err);
		    err = cudaMalloc(&g_state.states[state].buf[gpu_id].difficulty, HASH_SIZE * sizeof(uint8_t));
		    RET_ON_CUDA_FAIL(err);
		}

		g_state.configs[gpu_id].stage_1_threads = STAGE_1_THREADS;
		g_state.configs[gpu_id].stage_2_threads = STAGE_2_THREADS;
		g_state.configs[gpu_id].stage_3_threads = STAGE_3_THREADS;
		g_state.configs[gpu_id].stage_final_threads = NUM_HASH_THREADS;
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
		    //cudaMalloc(&g_state.states[state].buf[gpu_id].d_output_st2, batch_size * HASH_SIZE * sizeof(uint8_t));
		    //cudaMalloc(&g_state.states[state].buf[gpu_id].n_output_st2, batch_size * sizeof(uint64_t));
		    cudaFree(g_state.states[state].buf[gpu_id].d_output_result);
		    cudaFree(g_state.states[state].buf[gpu_id].n_output_result);
		    cudaFree(g_state.states[state].buf[gpu_id].difficulty);
		}
	}
	memset(&g_state, 0, sizeof(struct cuda_state));
	return 0;
}

}




