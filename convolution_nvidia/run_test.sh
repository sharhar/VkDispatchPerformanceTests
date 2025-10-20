#! /bin/bash

if [ ! -d "../convolution_nvidia_dependencies" ]; then
    echo "Error: Nvidia convolution dependencies not found. Please run the setup_nvidia_dependencies.sh script"
    exit 1
fi

mkdir -p test_results

DATA_SIZE=$1
ITER_COUNT=$2
BATCH_SIZE=$3
REPEATS=$4
ARCH=$5

echo "Running NVIDIA Convolution tests for arch SM_$ARCH"
echo "This may take several minutes..."

bash exec_tests.sh $ARCH $REPEATS > test_results/test.log

python3 parse_stdout_log.py test_results/test.log
#python3 make_graph.py