#!/bin/bash
set -e

CUDA_HOME=/usr/local/cuda-12/

ROOT_PATH=$PWD
BUILD_PATH="${ROOT_PATH}/build"
HASH_PATH="${ROOT_PATH}/xelis-hash"
SRC_PATH="${ROOT_PATH}/src"

rm -rf ${BUILD_PATH}/* > /dev/null 2>&1
mkdir -p ${BUILD_PATH}

cd ${HASH_PATH}
cargo build --release

cd ${BUILD_PATH}

rm -rf *.o
rm -rf hash-test
${CUDA_HOME}/bin/nvcc -dc ${SRC_PATH}/keccak.cu -o keccak.o
${CUDA_HOME}/bin/nvcc -dc ${SRC_PATH}/aes128.cu -o aes128.o
${CUDA_HOME}/bin/nvcc -dc ${SRC_PATH}/xelis-hash.cu -o xelis-hash.o

gcc -c ${SRC_PATH}/hash-test.c -o hash-test.o -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcudart

${CUDA_HOME}/bin/nvcc keccak.o aes128.o xelis-hash.o hash-test.o -o hash-test -L${HASH_PATH}/target/release -l:libhelishash.a


