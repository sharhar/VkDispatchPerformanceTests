#! /bin/bash

cd scaled_convolution
python3 make_figure.py
cd ..

cd 2d_convolution
python3 make_figure.py
cd ..

cd 2d_padded_convolution
python3 make_figure.py
cd ..