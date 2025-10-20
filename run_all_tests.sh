#! /bin/bash

DATA_SIZE=134217728
#DATA_SIZE=67108864
#DATA_SIZE=33554432
ITER_COUNT=200
BATCH_SIZE=20
REPEATS=3
ARCH=89

run_test() {
    local test_name=$1
    local title =$2
    local xlabel=$3
    local ylabel=$4

    echo "Running $test_name test..."
    cd $test_name
    #bash run_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
    cd ..

    python3 make_graph.py "$test_name" "$title" "$xlabel" "$ylabel"

    #cp $test_name/test_results/${test_name}_graph.png test_results/${test_name}_graph.png

    #mkdir -p test_results/$test_name

    #cp $test_name/test_results/*.csv test_results/$test_name/
    #cp $test_name/test_results/*.png test_results/$test_name/
}

mkdir -p test_results

echo "Running all tests with DATA_SIZE=$DATA_SIZE, ITER_COUNT=$ITER_COUNT, BATCH_SIZE=$BATCH_SIZE, REPEATS=$REPEATS, ARCH=$ARCH"

run_test "fft_nonstrided" "Nonstrided FFT Performance" "FFT Size" "GB/s (higher is better)"
run_test "convolution_nvidia" "NVidia Scaled Convolution Performance" "Convolution Size (FFT size)" "ms (lower is better)"
run_test "convolution_nonstrided_scaled" "Control Scaled Convolution Performance" "Convolution Size (FFT size)" "s (lower is better)"

#echo "Running fft_nonstrided test..."
#cd fft_nonstrided
#bash run_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
#cd ..
#cp fft_nonstrided/test_results/fft_nonstrided_graph.png test_results/fft_nonstrided_graph.png

# mkdir -p test_results/fft_nonstrided

# cp fft_nonstrided/test_results/*.csv test_results/fft_nonstrided/
# cp fft_nonstrided/test_results/*.png test_results/fft_nonstrided/

# echo "Running convolution_nvidia test..."
# cd convolution_nvidia
# bash run_convolution_nvidia_test.sh $ARCH $REPEATS
# cd ..
# cp convolution_nvidia/test_results/convolution_nvidia_graph.png test_results/convolution_nvidia_graph.png

# echo "Running convolution_nonstrided_scaled test..."
# cd convolution_nonstrided_scaled
# bash run_convolution_nonstrided_scaled_test.sh $DATA_SIZE $ITER_COUNT $BATCH_SIZE $REPEATS $ARCH
# cd ..
# cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_graph.png test_results/convolution_nonstrided_scaled_graph.png
# cp convolution_nonstrided_scaled/test_results/convolution_nonstrided_scaled_ratios_graph.png test_results/convolution_nonstrided_scaled_ratios_graph.png