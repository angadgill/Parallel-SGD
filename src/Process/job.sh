#!/bin/bash
#SBATCH --job-name="driver"
#SBATCH --output="sgd_compare.out"
#SBATCH --partition=compute
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=12
#SBATCH --export=ALL
#SBATCH -t 00:10:00

python driver.py
