#!/bin/bash

S=$1
E=$2
for ((i=S; i<=E; i++)); do
  python exp_main.py --max_gen 30 --log_folder "./logs_statistics_${i}" > "output_statistics_${i}"
done
