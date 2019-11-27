#!/bin/bash

S=$i
E=$2
for ((i=S; i<=E; i++)); do
  python exp_main.py --max_gen 30 --log_folder "./log_${i}" > "output_${i}"
done
