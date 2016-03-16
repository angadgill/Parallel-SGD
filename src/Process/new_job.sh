#!/bin/bash
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --job-name="python node"
#SBATCH --output="python_node.out"
#SBATCH -t 00:45:00

sleep 45m
