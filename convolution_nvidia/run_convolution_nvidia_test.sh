#! /bin/bash

if [ ! -d "../convolution_nvidia_dependencies" ]; then
    echo "Error: Nvidia convolution dependencies not found. Please run the setup_nvidia_dependencies.sh script"
    exit 1
fi

mkdir -p test_results

arch="$1"

echo "Running NVIDIA Convolution tests for arch SM_$arch"
echo "This may take several minutes..."

bash exec_tests.sh $arch $2 > test_results/test.log

python3 parse_stdout_log.py test_results/test.log
python3 make_graph.py