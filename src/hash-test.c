

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <pthread.h>

#include <gmp.h>

#include <cuda_runtime.h>

extern int xelis_hash_cuda(const uint8_t *input, size_t batch_size, uint8_t *output, int state);
extern int initialize_cuda(size_t batch_size, int num_states);
extern int deinitialize_cuda(void);
extern int xelis_hash_cuda_nonce(const uint8_t *base_header, uint64_t *nonce_start,
        size_t batch_size, uint8_t *output_hash,
        uint64_t *output_nonce,
        const uint8_t *difficulty, int gpu_id, int state);

typedef struct { uint8_t u[32]; } Hash;
extern Hash xelis_hash_cpu(uint8_t* input, size_t input_len, uint64_t* scratch_pad);


#define KECCAK_WORDS 25
#define BYTES_ARRAY_INPUT KECCAK_WORDS * 8
#define HASH_SIZE 32
#define MEMORY_SIZE 32768

uint8_t *fill_mem(uint32_t seed, uint64_t num){
    size_t count = num * BYTES_ARRAY_INPUT;

    uint8_t *mem = malloc(count);

    srand(seed);
    for (size_t i = 0; i < count; i++){
        mem[i] = rand();
    }

    return mem;
}

void print_buffer(void *_buffer, size_t len){
    uint8_t *buffer = (uint8_t *)_buffer;
    for (size_t i = 0; i < len; i++){
        printf("%x", buffer[i]);
    }
}

///////////////////////////////////////////////////////////////////

int verify_correctness(uint32_t seed, uint64_t num){
    uint8_t *cuda_hash = malloc(HASH_SIZE * num);
    uint64_t *scratch_pad = calloc(sizeof(uint64_t) * MEMORY_SIZE, num);
    uint8_t *test_samples_cuda = fill_mem(seed, num);
    uint8_t *test_samples_cpu = malloc(BYTES_ARRAY_INPUT * num);
    memcpy(test_samples_cpu, test_samples_cuda, BYTES_ARRAY_INPUT * num);

    printf("Verifyig...\n");
    xelis_hash_cuda(test_samples_cuda, num, cuda_hash, 0);

    int failed = 0;
    for (size_t i = 0; i < num; i++){
    uint8_t *input_cpu = &test_samples_cpu[i * BYTES_ARRAY_INPUT];

        Hash cpu_hash = xelis_hash_cpu(input_cpu, BYTES_ARRAY_INPUT,
            &scratch_pad[i * MEMORY_SIZE]);

        if(memcmp(&cuda_hash[HASH_SIZE * i], cpu_hash.u, HASH_SIZE) != 0){
            if(failed < 2){
                printf("For input ");
                print_buffer(input_cpu, HASH_SIZE);
                printf(" at index %ld incorrect hash\nCPU: ", i);
                print_buffer(&cpu_hash, HASH_SIZE);
                printf("\nGPU: ");
                print_buffer(cuda_hash, HASH_SIZE);
                printf("\n");
            }
            //return -1;
            failed++;
        }
    }
    printf("Failed %d/%ld\n", failed, num);
    

    free(test_samples_cuda);
    free(test_samples_cpu);
    free(scratch_pad);
    free(cuda_hash);
    return 0;
}

///////////////////////////////////////////////////////////////////

#define DO_NOT_OPTIMIZE(value) \
    asm volatile("" : "+m"(value) : : "memory")

int benchmark_speed(uint32_t seed, uint64_t num, size_t iters){
    uint8_t *cuda_hash = malloc(HASH_SIZE * num);
    uint64_t *scratch_pad = calloc(sizeof(uint64_t) * MEMORY_SIZE, num + 1);
    uint8_t *test_samples = fill_mem(seed, num);
    float time_taken;
    struct timespec start, end;

    printf("Benchmarking with batch %ld for %ld iter\n", num, iters);

    //cuda
    int result;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for(size_t i = 0; i < iters; i++){
        result = xelis_hash_cuda(test_samples, num, cuda_hash, 0);
        DO_NOT_OPTIMIZE(cuda_hash);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("GPU: %.4fs\n", time_taken);
    printf("Result: %d\n", result);


    //cpu
    test_samples = fill_mem(seed, num);
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < num; i++){
        uint8_t *input = &test_samples[i * BYTES_ARRAY_INPUT];

        Hash cpu_hash = xelis_hash_cpu(input, BYTES_ARRAY_INPUT, &scratch_pad[MEMORY_SIZE * i]);
        DO_NOT_OPTIMIZE(cpu_hash);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("CPU: %.4fs\n", time_taken);

    free(test_samples);
    free(scratch_pad);
    free(cuda_hash);
}

///////////////////////////////////////////////////////////////////

static void init_mpz_from_bytes(mpz_t result, const uint8_t* bytes, size_t num_bytes) {
    mpz_import(result, num_bytes, 1, sizeof(bytes[0]), 1, 0, bytes);
}
static void export_mpz_to_bytes(const mpz_t num, uint8_t* bytes, size_t num_bytes) {
    memset(bytes, 0, num_bytes);  // Clear the buffer, padding with zeros
    size_t count;
    mpz_export(bytes + (num_bytes - mpz_sizeinbase(num, 256) + 1), &count, 1, 1, 1, 0, num);
}
void compute_difficulty_target(uint64_t difficulty_uint, uint8_t *buf) {
    // Check if difficulty is zero
    if (difficulty_uint == 0) {
        return;
    }

    mpz_t target;
    mpz_init(target);
    init_mpz_from_bytes(target, buf, HASH_SIZE);
    mpz_t difficulty;
    mpz_init(difficulty);
    mpz_set_ui(difficulty, difficulty_uint);

    // Initialize maximum U256 value
    mpz_t max_u256;
    mpz_init(max_u256);
    mpz_ui_pow_ui(max_u256, 2, 256); // 2^256
    mpz_sub_ui(max_u256, max_u256, 1); // 2^256 - 1

    // Compute target = max_u256 / difficulty
    mpz_tdiv_q(target, max_u256, difficulty);

    export_mpz_to_bytes(target, buf, HASH_SIZE);
    // Clean up
    mpz_clear(max_u256);
    mpz_clear(target);
    mpz_clear(difficulty);
}

// Compute difficulty from hash and return as uint64_t
uint64_t difficulty_from_hash(const uint8_t* hash_bytes) {
    mpz_t hash_value, difficulty;
    mpz_init(hash_value);
    mpz_init(difficulty);

    // Convert hash from big-endian bytes to an mpz_t
    init_mpz_from_bytes(hash_value, hash_bytes, HASH_SIZE);

    // Define the maximum U256 value: 2^256 - 1
    mpz_t max_u256;
    mpz_init(max_u256);
    mpz_ui_pow_ui(max_u256, 2, 256);
    mpz_sub_ui(max_u256, max_u256, 1);  // max_u256 = 2^256 - 1

    // Calculate difficulty: max_u256 / hash_value
    if (mpz_cmp_ui(hash_value, 0) == 0) {
        mpz_clears(hash_value, difficulty, max_u256, NULL);
        return 0;  // Avoid division by zero, returning zero difficulty
    } else {
        mpz_tdiv_q(difficulty, max_u256, hash_value);
    }

    // Convert mpz difficulty to uint64_t
    uint64_t difficulty_uint64 = mpz_get_ui(difficulty);

    // Clean up mpz_t variables
    mpz_clears(hash_value, difficulty, max_u256, NULL);

    return difficulty_uint64;
}

struct mythr_data{
    uint8_t *test_samples;
    size_t iters;
    size_t num;
    size_t state;
};

void *worker_thread(void *_data){
    struct mythr_data *data = _data;
    uint8_t difficulty_buf[HASH_SIZE];
    uint64_t output_nonce;
    uint8_t output_hash[HASH_SIZE];
    uint64_t nonce_start = 0;

    for(size_t i = 0; i < data->iters; i++){
        xelis_hash_cuda_nonce(data->test_samples, &nonce_start,
            data->num, output_hash, &output_nonce,
            difficulty_buf, 0, data->state);
    }

    DO_NOT_OPTIMIZE(output_hash);
}

int verify_difficulty(uint32_t seed, size_t num, size_t num_thr, size_t iters, uint64_t nonce_start, uint64_t difficulty){
    uint8_t output_hash[HASH_SIZE];
    uint64_t output_nonce;
    uint8_t *test_samples = fill_mem(seed, num_thr);
    float time_taken;
    struct timespec start, end;
    uint8_t difficulty_buf[HASH_SIZE];

    compute_difficulty_target(difficulty, difficulty_buf);

    printf("Benchmarking nonce with batch %ld for %ld iters in %ld threads\n", num, iters, num_thr);

    //cuda
    pthread_t threads[128];
    struct mythr_data _data[128];

    for (int i = 0; i < num_thr; ++i){
        _data[i].test_samples = &test_samples[i * KECCAK_WORDS];
        _data[i].iters = iters;
        _data[i].num = num;
        _data[i].state = i;
    }

    clock_gettime(CLOCK_MONOTONIC, &start);
    if(num_thr == 1){
        for(size_t i = 0; i < iters; i++){
            xelis_hash_cuda_nonce(test_samples, &nonce_start,
                num, output_hash, &output_nonce,
                difficulty_buf, 0, 0);
            DO_NOT_OPTIMIZE(output_hash);
        }
    }else {
        for (int i = 0; i < num_thr; ++i)
            pthread_create(&threads[i], NULL, worker_thread, &_data[i]);
        for (int i = 0; i < num_thr; ++i)
            pthread_join(threads[i], NULL);
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    time_taken = end.tv_sec - start.tv_sec + (end.tv_nsec - start.tv_nsec) / 1e9;

    printf("GPU: %.4fs, %ldKH/s\n", time_taken, (uint64_t)(num * num_thr * iters / time_taken) / 1024);
    printf("Found hash with nonce %lu:\n", output_nonce);
    print_buffer(output_hash, HASH_SIZE);
    printf("\nDifficulty: %lu, expected %lu\n",
        difficulty_from_hash(output_hash), difficulty);
}

#define NUM_THREADS_PER_BLOCK 1024
#define NUM_THREADS_PER_SM 1536
#define NUM_BLOCKS_PER_SM 24
#define NUM_SM 128
#define NUM_SM_PER_KACCEK (NUM_SM / 8)
#define NUM_THREADS_PER_BLOCK (NUM_THREADS_PER_SM / NUM_BLOCKS_PER_SM) //64
#define MAX_NUM_BLOCKS NUM_BLOCKS_PER_SM * NUM_SM //3K
#define MAX_NUM_THREADS NUM_SM * NUM_THREADS_PER_SM //196608
//stage 1 pralellism: 1024

int main() {
    size_t num_iters = 10;
    size_t num_threads = 1;
    size_t batch_size = 1024 * 32;
    initialize_cuda(batch_size, num_threads);
    verify_correctness(100, 256);
    srand(time(0));
    int seed = rand();
    verify_difficulty(100, batch_size, num_threads, num_iters, 0, 512);
    //benchmark_speed(100, batch_size, num_iters);
    deinitialize_cuda();
}
