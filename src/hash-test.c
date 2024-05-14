

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_runtime.h>

extern void xelis_hash_cuda(uint8_t *input, size_t input_size, uint8_t *output);
extern int init_cuda(void);


typedef struct { uint8_t u[32]; } Hash;
extern Hash xelis_hash_cpu(unsigned char* input, size_t input_len, uint64_t* scratch_pad);


#define KECCAK_WORDS 25
#define BYTES_ARRAY_INPUT KECCAK_WORDS * 8
#define HASH_SIZE 32
#define MEMORY_SIZE 32768

void test_cuda(){
    if (init_cuda() != 0) return;

    uint8_t input[BYTES_ARRAY_INPUT]; // Ensure enough input for full processing
    uint8_t output[HASH_SIZE];
    for (int i = 0; i < sizeof(input); i++) {
        input[i] = i % 256;
    }
    printf("Started CUDA\n");
    xelis_hash_cuda(input, sizeof(input), output);
    printf("Completed. Hash:\n");
    for (int i = 0; i < HASH_SIZE; i++) {
        printf("%02x", output[i]);
    }
    printf("\n");
}

void test_cpu(){
    uint8_t input[BYTES_ARRAY_INPUT]; // Ensure enough input for full processing
    for (int i = 0; i < sizeof(input); i++) {
        input[i] = i % 256;
    }
    uint64_t *scratch_pad = malloc(sizeof(uint64_t) * MEMORY_SIZE);

    printf("Started CPU\n");
    Hash hash = xelis_hash_cpu(input, sizeof(input), scratch_pad);
    printf("Completed. Hash:\n");
    for (int i = 0; i < HASH_SIZE; i++) {
        printf("%02x", hash.u[i]);
    }
    printf("\n");
}

int main() {
    test_cpu();
    test_cuda();
    return 0;
}
