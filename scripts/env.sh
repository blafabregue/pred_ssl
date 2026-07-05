#!/bin/bash
# Per-cluster environment setup. Sourced by the SLURM job scripts (sbatch_pretrain.slurm,
# sbatch_eval.slurm) and run_pipeline.sh before any python call, so the interpreter with
# torch is on PATH on the compute node. EDIT THIS for your cluster (module load / conda /
# venv activate). Delete the file to fall back to the scripts' default `conda activate`.
#
# Configured for hpc-login1: modules provide Python 3.12 + torch 2.5.1 (CUDA 12.1).

# make `module` available in a non-interactive (batch) shell if it isn't already
module load python/3.12.8
source /home2020/home/icube/blafabre/pred_ssl/venv/bin/activate
