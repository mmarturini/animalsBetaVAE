#!/bin/bash
#PBS -q dssc_gpu
#PBS -l nodes=1:ppn=1
#PBS -l walltime=12:00:00

cd $PBS_O_WORKDIR
/u/dssc/mmarturi/.conda/envs/DeepLearning/bin/python3.9 catVAEb120.py > catVAEb120.out


