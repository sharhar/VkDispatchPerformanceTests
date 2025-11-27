#! /bin/bash

cd cufftdx_validation
python3 make_tables.py
cd ..

cd vulkan_cuda_calibration
python3 make_tables.py
cd ..