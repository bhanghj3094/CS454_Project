#!/bin/bash

for ((i=0;i<=29;i++))
do
  cd logs_statistics_${i}
  ls | cut -d . -f 1 | while read line; do mv $line.txt $line.csv; done
  cd ..
done

