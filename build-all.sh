#!/bin/bash
set -e

#CUDA_HOME=/usr/local/cuda-12.4/
CUDA_HOME=/usr/local/cuda-12/
#CUDA_HOME=/usr/local/cuda-11.8/

BASE_FLAGS="--use_fast_math -Xcompiler -Wall,-O3 -Xcompiler -fPIC"

#Ampere
#TARGET_FLAGS="-arch=sm_89"
TARGET_FLAGS="-arch=sm_89 -dlto"
#Ampere
#TARGET_FLAGS="-arch=sm_75 -dlto"
#Volta
#TARGET_FLAGS="-arch=sm_70 -dlto"
#Pascal
#TARGET_FLAGS="-arch=sm_60 -dlto"

#EXTRA_OPT_OPTIONS="-G -g -lineinfo"
EXTRA_OPT_OPTIONS=""
# -maxrregcount=128
# --ptxas-options=-v
# --extra-device-vectorization

OPT_FLAGS=" -Xptxas=-O3,-dlcm=ca ${BASE_FLAGS} ${EXTRA_OPT_OPTIONS} ${TARGET_FLAGS}"
APP_OPT_FLAGS="-O2 -Wall"


ROOT_PATH=$PWD
BUILD_PATH="${ROOT_PATH}/build"
HASH_PATH="${ROOT_PATH}/xelis-hash"
MINER_PATH="${ROOT_PATH}/xelis-miner/xelis_miner"
SRC_PATH="${ROOT_PATH}/src"



rm -rf ${BUILD_PATH}/* > /dev/null 2>&1
mkdir -p ${BUILD_PATH}

cd ${HASH_PATH}
cargo clean
cargo build --release

cd ${BUILD_PATH}

rm -rf *.o
rm -rf *.a
rm -rf *.so
rm -rf hash-test
${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} -dc ${SRC_PATH}/keccak.cu -o keccak.o
${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} -dc ${SRC_PATH}/aes128.cu -o aes128.o
${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} -dc ${SRC_PATH}/xelis-hash.cu -o xelis-hash.o

gcc ${APP_OPT_FLAGS} -c ${SRC_PATH}/hash-test.c -o hash-test.o -I${CUDA_HOME}/include -L${CUDA_HOME}/lib64 -lcudart  -lgmp

${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} keccak.o aes128.o xelis-hash.o hash-test.o -o hash-test -L${HASH_PATH}/target/release -l:libhelishash.a  -lgmp

${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} -dlink keccak.o aes128.o xelis-hash.o -o all-devices.o
${CUDA_HOME}/bin/nvcc ${OPT_FLAGS} -lib -o libxelis_hash_cuda.a keccak.o aes128.o xelis-hash.o all-devices.o

cd ${MINER_PATH}
cargo clean
cargo build --release


