//Code taken from https://github.com/Rvch7/AES128_CUDA/blob/master/AES_CUDA/AES128_functions.cu

#include <stdint.h>
#include <cuda_runtime.h>


#ifdef __CUDA_ARCH__
#define CONSTANT __constant__ 
#else
#define CONSTANT const
#endif

// Definations
typedef unsigned char BYTE;
extern CONSTANT BYTE SBOX[256];

#define BLOCKSIZE 16
#define NUMOFKEYS 11

//const char Nk = 4;    // Number of keys
//const char Nb = 4;    // Number of words in a block
const char Nr = 10; // Number of rounds

struct word {
    BYTE byte[4];
    __device__ word operator^(word x) {
        word z;
        z.byte[0] = x.byte[0] ^ this->byte[0];
        z.byte[1] = x.byte[1] ^ this->byte[1];
        z.byte[2] = x.byte[2] ^ this->byte[2];
        z.byte[3] = x.byte[3] ^ this->byte[3];
        return z;
    }
};

struct block_t {
    word state[4] = {};
    __device__ block_t operator^(block_t x) {
        block_t z;
        z.state[0] = x.state[0] ^ this->state[0];
        z.state[1] = x.state[1] ^ this->state[1];
        z.state[2] = x.state[2] ^ this->state[2];
        z.state[3] = x.state[3] ^ this->state[3];
        return z;
    }
};


CONSTANT BYTE SBOX[256] = {
 0x63 ,0x7c ,0x77 ,0x7b ,0xf2 ,0x6b ,0x6f ,0xc5 ,0x30 ,0x01 ,0x67 ,0x2b ,0xfe ,0xd7 ,0xab ,0x76
,0xca ,0x82 ,0xc9 ,0x7d ,0xfa ,0x59 ,0x47 ,0xf0 ,0xad ,0xd4 ,0xa2 ,0xaf ,0x9c ,0xa4 ,0x72 ,0xc0
,0xb7 ,0xfd ,0x93 ,0x26 ,0x36 ,0x3f ,0xf7 ,0xcc ,0x34 ,0xa5 ,0xe5 ,0xf1 ,0x71 ,0xd8 ,0x31 ,0x15
,0x04 ,0xc7 ,0x23 ,0xc3 ,0x18 ,0x96 ,0x05 ,0x9a ,0x07 ,0x12 ,0x80 ,0xe2 ,0xeb ,0x27 ,0xb2 ,0x75
,0x09 ,0x83 ,0x2c ,0x1a ,0x1b ,0x6e ,0x5a ,0xa0 ,0x52 ,0x3b ,0xd6 ,0xb3 ,0x29 ,0xe3 ,0x2f ,0x84
,0x53 ,0xd1 ,0x00 ,0xed ,0x20 ,0xfc ,0xb1 ,0x5b ,0x6a ,0xcb ,0xbe ,0x39 ,0x4a ,0x4c ,0x58 ,0xcf
,0xd0 ,0xef ,0xaa ,0xfb ,0x43 ,0x4d ,0x33 ,0x85 ,0x45 ,0xf9 ,0x02 ,0x7f ,0x50 ,0x3c ,0x9f ,0xa8
,0x51 ,0xa3 ,0x40 ,0x8f ,0x92 ,0x9d ,0x38 ,0xf5 ,0xbc ,0xb6 ,0xda ,0x21 ,0x10 ,0xff ,0xf3 ,0xd2
,0xcd ,0x0c ,0x13 ,0xec ,0x5f ,0x97 ,0x44 ,0x17 ,0xc4 ,0xa7 ,0x7e ,0x3d ,0x64 ,0x5d ,0x19 ,0x73
,0x60 ,0x81 ,0x4f ,0xdc ,0x22 ,0x2a ,0x90 ,0x88 ,0x46 ,0xee ,0xb8 ,0x14 ,0xde ,0x5e ,0x0b ,0xdb
,0xe0 ,0x32 ,0x3a ,0x0a ,0x49 ,0x06 ,0x24 ,0x5c ,0xc2 ,0xd3 ,0xac ,0x62 ,0x91 ,0x95 ,0xe4 ,0x79
,0xe7 ,0xc8 ,0x37 ,0x6d ,0x8d ,0xd5 ,0x4e ,0xa9 ,0x6c ,0x56 ,0xf4 ,0xea ,0x65 ,0x7a ,0xae ,0x08
,0xba ,0x78 ,0x25 ,0x2e ,0x1c ,0xa6 ,0xb4 ,0xc6 ,0xe8 ,0xdd ,0x74 ,0x1f ,0x4b ,0xbd ,0x8b ,0x8a
,0x70 ,0x3e ,0xb5 ,0x66 ,0x48 ,0x03 ,0xf6 ,0x0e ,0x61 ,0x35 ,0x57 ,0xb9 ,0x86 ,0xc1 ,0x1d ,0x9e
,0xe1 ,0xf8 ,0x98 ,0x11 ,0x69 ,0xd9 ,0x8e ,0x94 ,0x9b ,0x1e ,0x87 ,0xe9 ,0xce ,0x55 ,0x28 ,0xdf
,0x8c ,0xa1 ,0x89 ,0x0d ,0xbf ,0xe6 ,0x42 ,0x68 ,0x41 ,0x99 ,0x2d ,0x0f ,0xb0 ,0x54 ,0xbb ,0x16 };

__device__ __forceinline__ BYTE xtimes(BYTE x) {
    return ((x << 1) ^ (((x >> 7) & 1) * 0x1b));
}

__device__ __forceinline__ void mix_columns(block_t* block) {
    block_t out;

    for (int i = 0; i < BLOCKSIZE; i += 4) {
        out.state->byte[i + 0] = xtimes(block->state->byte[i + 0]) ^ xtimes(block->state->byte[i + 1]) ^ block->state->byte[i + 1] ^ block->state->byte[i + 2] ^ block->state->byte[i + 3];
        out.state->byte[i + 1] = block->state->byte[i + 0] ^ xtimes(block->state->byte[i + 1]) ^ xtimes(block->state->byte[i + 2]) ^ block->state->byte[i + 2] ^ block->state->byte[i + 3];
        out.state->byte[i + 2] = block->state->byte[i + 0] ^ block->state->byte[i + 1] ^ xtimes(block->state->byte[i + 2]) ^ xtimes(block->state->byte[i + 3]) ^ block->state->byte[i + 3];
        out.state->byte[i + 3] = xtimes(block->state->byte[i + 0]) ^ block->state->byte[i + 0] ^ block->state->byte[i + 1] ^ block->state->byte[i + 2] ^ xtimes(block->state->byte[i + 3]);
    }

    *block = out;
}

__device__ __forceinline__ void shift_rows(block_t* block) {  // Performs shift rows operation on a block
    block_t out;
    //On per-row basis (+1 shift X each row)
    //Row 1
    out.state->byte[0] = block->state->byte[0];
    out.state->byte[4] = block->state->byte[4];
    out.state->byte[8] = block->state->byte[8];
    out.state->byte[12] = block->state->byte[12];
    //Row 2
    out.state->byte[1] = block->state->byte[5];
    out.state->byte[5] = block->state->byte[9];
    out.state->byte[9] = block->state->byte[13];
    out.state->byte[13] = block->state->byte[1];
    //Row 3
    out.state->byte[2] = block->state->byte[10];
    out.state->byte[6] = block->state->byte[14];
    out.state->byte[10] = block->state->byte[2];
    out.state->byte[14] = block->state->byte[6];
    //Row 4
    out.state->byte[3] = block->state->byte[15];
    out.state->byte[7] = block->state->byte[3];
    out.state->byte[11] = block->state->byte[7];
    out.state->byte[15] = block->state->byte[11];

    *block = out;
}

__device__ __forceinline__ void sbox_substitute(block_t* block) {
//Performs an S-box substitution on a block
    for (int i = 0; i < 16; i++) {
        uint8_t index = block->state->byte[i];
        block->state->byte[i] = SBOX[index];
    }
}

__device__ __forceinline__ void addroundkey(block_t* block, const block_t* expandedkeys) {
    *block = *block ^ *expandedkeys;
}

__device__ void gpu_cipher(void* _block, void* _expandedkeys) {
    block_t *block = (block_t *)_block;
    block_t *expandedkeys = (block_t *)_expandedkeys;

    unsigned int i = 0;
    addroundkey((block + i), expandedkeys);
    for (int round = 1; round < Nr; round++) {
        sbox_substitute((block + i));
        shift_rows((block + i));
        mix_columns((block + i));
        addroundkey((block + i), (expandedkeys + round));
    }
    sbox_substitute((block + i));
    shift_rows((block + i));
    addroundkey((block + i),(expandedkeys + Nr));

}

__device__ void gpu_cipher_round(void* __restrict__ _block) {
    block_t *block = (block_t *)_block;

    sbox_substitute(block);
    shift_rows(block);
    mix_columns(block);
    //addroundkey(block, round_key); - we assume key is always 0, so skipping this
}