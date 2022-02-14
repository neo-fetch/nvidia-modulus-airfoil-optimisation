#!/bin/bash
input="/examples/ldc/angles"
i=0
while IFS= read -r line
do
  python ldc_2d.py $line &
  sleep $((60*10))
  kill "$!"
  a="network_checkpoint_ldc_2d"
  b="${a}_new_${i}"
  i=`expr $i + 1`
  mv $a $b
done < "$input"
