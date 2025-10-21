#! /bin/bash

bash compile_and_run.sh 64 $1 $2
bash compile_and_run.sh 128 $1 $2
bash compile_and_run.sh 256 $1 $2
bash compile_and_run.sh 512 $1 $2
bash compile_and_run.sh 1024 $1 $2
bash compile_and_run.sh 2048 $1 $2
bash compile_and_run.sh 4096 $1 $2