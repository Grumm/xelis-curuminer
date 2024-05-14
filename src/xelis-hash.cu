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

extern __device__ void keccak(const char *message, int message_len, unsigned char *output, int output_len);
//extern __device__ void encrypt(uint8_t *block, uint8_t *rkey, uint32_t offset);
extern __device__ void gpu_cipher(void* _block, void* _expandedkeys);

////////////////////////////////////////////////////////////////

__device__ void keccakp_kernel(uint64_t *data){
	return keccak((char *)data, BYTES_ARRAY_INPUT, (unsigned char*)data, BYTES_ARRAY_INPUT);
}

__global__ void stage_1_impl(uint64_t *int_input, uint64_t *scratch_pad, int a0, int a1, int b0, int b1) {
    for (size_t i = a0; i <= a1; i++){
        //keccakp_kernel(int_input);

        uint64_t rand_int = 0;
        for (size_t j = b0; j <= b1; j++) {
            size_t pair_idx = (j + 1) % KECCAK_WORDS;
            size_t pair_idx2 = (j + 2) % KECCAK_WORDS;

            size_t target_idx = i * KECCAK_WORDS + j;
            uint64_t a_val = int_input[j] ^ rand_int;
            uint64_t left = int_input[pair_idx];
            uint64_t right = int_input[pair_idx2];
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

            uint64_t b_val = a_val ^ v;
            rand_int = b_val;
            scratch_pad[target_idx] = b_val;

            //TODO temp store in __shared__, at the end copy to __global__
        }
    }
}

#define STAGE_1_THREADS 256

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

__host__ void print_buffer_host(void *_buffer, size_t len){
	uint8_t *buffer = (uint8_t *)_buffer;
	for (size_t i = 0; i < len; i++){
		printf("%x", buffer[i]);
	}
	printf("\n");
}

/*
// Assuming `int_input` and `scratch_pad` are already allocated on the device
__device__ void stage1_kernel(uint64_t *int_input, uint64_t *scratch_pad) {
    //dim3 blocks((STAGE_1_MAX + 1 + 255) / 256); // Example calculation for blocks
    //dim3 threads(256); // Number of threads per block

    // First call
    stage_1_impl(int_input, scratch_pad, 0, STAGE_1_MAX - 1, 0, KECCAK_WORDS - 1);
    // Second call
    //print_buffer(scratch_pad, 1);
    stage_1_impl(int_input, scratch_pad, STAGE_1_MAX, STAGE_1_MAX, 0, 17); // Only 18 threads for this one
    //print_buffer(scratch_pad, 1);
    //SYNC
}*/

/*
__device__ void stage1_kernel(uint64_t* scratch_pad, uint64_t* int_input) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < STAGE_1_MAX) {
        keccakp_kernel(&int_input[idx * KECCAK_WORDS]); // Make sure this processes a block of KECCAK_WORDS

        uint64_t rand_int = 0;
        for (int i = 0; i < KECCAK_WORDS; i++) {
            int pair_idx1 = (i + 1) % KECCAK_WORDS;
            int pair_idx2 = (i + 2) % KECCAK_WORDS;
            uint64_t a = int_input[idx * KECCAK_WORDS + i] ^ rand_int;
            uint64_t left = int_input[idx * KECCAK_WORDS + pair_idx1];
            uint64_t right = int_input[idx * KECCAK_WORDS + pair_idx2];
            uint64_t v = left ^ right; // Simplify or expand as needed
            rand_int = a ^ v;
            scratch_pad[idx * KECCAK_WORDS + i] = rand_int;
        }
    }
}*/

__device__ void kernel_memcpy(void *dst, const void *src, size_t len){
	for (size_t i = 0; i < len; i++){
		((uint8_t *)dst)[i] = ((uint8_t *)src)[i];
	}
}

////////////////////////////////////////////////////////////////

__global__ void stage2_kernel(uint64_t *scratch_pad, uint16_t *ref_indices) {
    // Temporary array to store indices - assuming SLOT_LENGTH is small enough for shared memory usage
    __shared__ uint16_t indices[SLOT_LENGTH];
    __shared__ uint32_t slots[SLOT_LENGTH];
    uint32_t *small_pad = (uint32_t *)scratch_pad;

    for(size_t i = 0; i < SLOT_LENGTH; i++){
    	ref_indices[i] = i;
    	slots[i] = 0;
    }

    uint32_t pad_len = MEMORY_SIZE * 2;
    uint32_t num_slots = pad_len / SLOT_LENGTH;

    kernel_memcpy(slots, &small_pad[pad_len - SLOT_LENGTH], SLOT_LENGTH * sizeof(uint32_t));

    for (uint32_t iter = 0; iter < ITERS; iter++) {
        for (uint32_t j = 0; j < num_slots; j++) {
    		kernel_memcpy(indices, ref_indices, SLOT_LENGTH * sizeof(uint16_t));

    		for (int16_t slot_idx = SLOT_LENGTH - 1; slot_idx >= 0; slot_idx--){
                uint32_t index_in_indices = small_pad[j * SLOT_LENGTH + slot_idx] % ((uint32_t)(slot_idx + 1));
                uint16_t index = indices[index_in_indices];
                indices[index_in_indices] = indices[slot_idx];

                uint32_t sum = slots[index];
                uint32_t offset = j * SLOT_LENGTH;
                
                for (uint16_t k = 0; k < SLOT_LENGTH; ++k) {
                	if (k == index) continue;

			        uint32_t pad = small_pad[offset + k];
			        if ((slots[k] >> 31) == 0) {
			            sum += pad;
			        } else {
			            sum -= pad;
			        }
			    }
                slots[index] = sum;
            }
        }
    }

    kernel_memcpy(&small_pad[pad_len - SLOT_LENGTH], slots, SLOT_LENGTH * sizeof(uint32_t));
}

////////////////////////////////////////////////////////////////



////////////////

// Mock AES function - replace with actual AES kernel calls or implementation
__device__ void aes_cipher_round(uint8_t *block, uint8_t *key) {
    //encrypt(block, key, 0);
    //gpu_cipher(block, key);
}

__device__ uint64_t rotate_left(uint64_t result, uint32_t j) {
    if (j == 0) return result;
    return (result << j) | (result >> (64 - j));
}

__device__ uint64_t calc_hash(uint64_t* mem_buffer_a, uint64_t* mem_buffer_b, uint64_t result, uint64_t i) {
    for (uint32_t j = 0; j < HASH_SIZE; j++) {
        uint64_t a = mem_buffer_a[(j + i) % BUFFER_SIZE];
        uint64_t b = mem_buffer_b[(j + i) % BUFFER_SIZE];

        // Determine operation based on the condition
        uint8_t case_index = (result >> (j * 2)) & 0xf;
        uint64_t v = 0; // Default to zero if something goes wrong

        switch (case_index) {
            case 0:  v = rotate_left(result, j) ^ b; break;
            case 1:  v = ~(rotate_left(result, j) ^ a); break;
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

__device__ void uint64_to_be_bytes(uint64_t num, uint8_t* bytes) {
    for (int i = 0; i < sizeof(uint64_t); ++i) {
        bytes[i] = (num >> (56 - 8 * i)) & 0xFF;
    }
}



__global__ void stage3_kernel(uint64_t *scratch_pad, uint8_t *output) {
    uint8_t block[16] = {0};  // Simulating GenericArray from Rust
    uint8_t key[16] = {0};  // Simulating GenericArray from Rust

    uint64_t mem_buffer_a[BUFFER_SIZE] = {0};
    uint64_t mem_buffer_b[BUFFER_SIZE] = {0};
    uint64_t addr_a = (scratch_pad[MEMORY_SIZE - 1] >> 15) & 0x7FFF;
    uint64_t addr_b = scratch_pad[MEMORY_SIZE - 1] & 0x7FFF;

    // Populate memory buffers
    for (uint64_t i = 0; i < BUFFER_SIZE; i++) {
        mem_buffer_a[i] = scratch_pad[(addr_a + i) % MEMORY_SIZE];
        mem_buffer_b[i] = scratch_pad[(addr_b + i) % MEMORY_SIZE];
    }

    uint8_t final_result[HASH_SIZE] = {0};

    uint64_t result;
    for (uint64_t i = 0; i < SCRATCHPAD_ITERS; i++) {
        uint64_t mem_a = mem_buffer_a[i % BUFFER_SIZE];
        uint64_t mem_b = mem_buffer_b[i % BUFFER_SIZE];

        // Simulating block operations (big-endian handling might need adjustments)
        kernel_memcpy(block, &mem_b, sizeof(mem_b));
        kernel_memcpy(block + 8, &mem_a, sizeof(mem_a));

        aes_cipher_round(block, key);  // Assuming 'key' is available

        uint64_t hash1 = *((uint64_t *)block); // Assuming block[0..8] is directly readable
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
        	uint64_to_be_bytes(result, &final_result[index * 8]);
//            kernel_memcpy(&final_result[index * 8], &result, sizeof(result));
        }
    }

    kernel_memcpy(output, final_result, HASH_SIZE);
}

////////////////////////////////////////////////////////////////
/*
__global__ void xelis_hash_cuda_impl(uint64_t *scratch_pad, uint64_t *int_input, uint16_t *ref_indices, uint8_t *d_output) {
    //stage1_kernel(int_input, scratch_pad);

    print_buffer<<<1, 1>>>(scratch_pad, 32*8);
    stage_1_impl(int_input, scratch_pad, 0, STAGE_1_MAX - 1, 0, KECCAK_WORDS - 1);
    // Second call
    print_buffer<<<1, 1>>>(scratch_pad, 32*8);
    stage_1_impl(int_input, scratch_pad, STAGE_1_MAX, STAGE_1_MAX, 0, 17); // Only 18 threads for this one
    print_buffer<<<1, 1>>>(scratch_pad, 32*8);

    //<<<1, 1>>>
    stage2_kernel((uint32_t *)scratch_pad, ref_indices);
    print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE - SLOT_LENGTH/2], 32*4);
    stage3_kernel(scratch_pad, d_output);
    print_buffer<<<1, 1>>>(d_output, 32*1);
}*/

extern "C" {

void xelis_hash_cuda(uint8_t *input, size_t input_size, uint8_t *output) {
    uint64_t *scratch_pad, *int_input;
    uint16_t *ref_indices;
    uint8_t *d_output;


    // TODO preallocate one time only
    cudaMalloc(&scratch_pad, MEMORY_SIZE * sizeof(uint64_t));
    cudaMalloc(&int_input, KECCAK_WORDS * sizeof(uint64_t));
    cudaMalloc(&ref_indices, SLOT_LENGTH * sizeof(uint16_t));
    cudaMalloc(&d_output, HASH_SIZE * sizeof(uint8_t));

    cudaMemset(scratch_pad, 0, MEMORY_SIZE * sizeof(uint64_t));

/*
    uint64_t *scratch_pad_h = (uint64_t *)malloc(MEMORY_SIZE * sizeof(uint64_t));
    cudaMemcpy(scratch_pad_h, scratch_pad, MEMORY_SIZE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    print_buffer_host(scratch_pad_h, 32*8);*/

    cudaMemcpy(int_input, input, KECCAK_WORDS * sizeof(uint64_t), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    //xelis_hash_cuda_impl<<<1, 1>>>(scratch_pad, int_input, ref_indices, d_output);

    //print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE-128], 1024);
    cudaDeviceSynchronize();
    stage_1_impl<<<1, 1>>>(int_input, scratch_pad, 0, STAGE_1_MAX - 1, 0, KECCAK_WORDS - 1);
    // Second call
    cudaDeviceSynchronize();
    //print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE-128], 1024);
    cudaDeviceSynchronize();
    stage_1_impl<<<1, 1>>>(int_input, scratch_pad, STAGE_1_MAX, STAGE_1_MAX, 0, 17); // Only 18 threads for this one
    cudaDeviceSynchronize();
    //print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE / 2], 1024);
    print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE-128], 1024);

    cudaDeviceSynchronize();
    //<<<1, 1>>>
    stage2_kernel<<<1, 1>>>(scratch_pad, ref_indices);
    cudaDeviceSynchronize();
    print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE - 128], 1024);
    cudaDeviceSynchronize();
    stage3_kernel<<<1, 1>>>(scratch_pad, d_output);
    print_buffer<<<1, 1>>>(&scratch_pad[MEMORY_SIZE - 128], 1024);
    cudaDeviceSynchronize();
    print_buffer<<<1, 1>>>(d_output, 32*1, true);
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_output, HASH_SIZE * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaFree(d_output);
    cudaFree(ref_indices);
    cudaFree(int_input);
    cudaFree(scratch_pad);
}


int init_cuda(void){
    int numDevices;
    cudaGetDeviceCount(&numDevices);
    if (numDevices < 1) {
        return -1;
    }
    // Set the device to use - for example, GPU 0
    int deviceId = 6; // You can select any device from 0 to numDevices - 1
    cudaError_t err = cudaSetDevice(deviceId);
    if (err != cudaSuccess) {
        return -1;
    }
    return 0;
}

}