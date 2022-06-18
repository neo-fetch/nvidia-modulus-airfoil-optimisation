#!/bin/bash
# while true
# do
#     docker cp thirsty_golick:"/examples/ldc/" "simnet/ldc_10mps_new_${i}/" &
#     sleep $((2*60))
#     # kill "$!"
#     i=$((i+1))
# done
while true
do
    steps=$(cat ~/simnet/examples/ldc/network_checkpoint_sine_k_all_5/checkpoint | grep model_checkpoint_path: | grep -Eo '[0-9]*')
    # If steps is empty then sleep for 2 seconds and skip the iteration
    if [ -z "$steps" ]; then
        echo "No steps found. Sleeping for 2 seconds"
        sleep 2
        continue
    fi
    echo "steps: ${steps}"
    sleep $((1))
    # convert steps to integer
    steps=$(($steps))
    # check if steps is divisible by 20000
    if [ $(($steps % 1000)) -eq 0 ]; then
        # Check if steps is 0
        if [ $steps -eq 0 ]; then
            echo "0 steps done. Starting from 1000"
        else
            echo "1000 steps done. Backing up files..."
        fi
        # backup files
        echo "Copying checkpoint at ${steps}..."
        cp -r ~/simnet/examples/ldc/network_checkpoint_sine_k_all_5/ ~/simnet/results/sine_checkpoint_results_${steps}
        echo "Done! saved to ~/simnet/results/sine_checkpoint_results_${steps}"
        # echo "Result directory contents:"
        # ls ~/simnet/results/ | grep sine_checkpoint_results 
        echo "Sleeping for 2 seconds..."
        sleep $((2))
    fi
done
