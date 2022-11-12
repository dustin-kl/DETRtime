#!/bin/bash

#SBATCH --mail-type=ALL                     # mail configuration: NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --output=logs/%j.out                 # where to store the output (%j is the JOBID), subdirectory must exist
#SBATCH --error=logs/%j.err                  # where to store error messages
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G

echo "Running on host: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# Exit on errors
set -o errexit

# Set a directory for temporary files unique to the job with automatic removal at job termination
TMPDIR=$(mktemp -d)
if [[ ! -d ${TMPDIR} ]]; then
    echo 'Failed to create temp directory' >&2
    exit 1
fi
trap "exit 1" HUP INT TERM
trap 'rm -rf "${TMPDIR}"' EXIT
export TMPDIR

# Binary or script to execute
python main.py
	--data_path /itet-stor/klebed/net_scratch/segmentation/ICML_dots_min_segmentation_minseq_4000_margin_2_amp_thresh_150/tensors \
	--backbone inception_time \
	--lr_backbone 1e-4 \
	--nb_filters 16 \
	--use_residual False \
	--backbone_depth 6 \
	--batch_size 32 \
	--bbox_loss_coef 10 \
	--giou_loss_coef 2 \
	--eos_coef 0.4 \
	--hidden_dim 128 \
	--dim_feedforward 512 \
	--dropout 0.1 \
	--wandb_dir movie \
	--num_queries 30 \
	--lr_drop 50 \
	--output_dir ./runs/"$modelname" & \

echo "Finished at:     $(date)"
exit 0