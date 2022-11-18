#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=kitti
#SBATCH --output=pvrcnn_waymo_result/pvrcnn_resized_gt_eval_on_resized_kirk_%j.%N.out
# SBATCH --output=pvrcnn_kitti_result/H125_N5_F1_no_GT_yes_Shape_%j.%N.out
# SBATCH --output=votr_kitti_result/H125_N4_voxelrcnn_head_noaug_%j.%N.out
#SBATCH --nodes=1
#SBATCH --ntasks=3
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --mem=120G
# SBATCH --gpus=rtx_a5000:2
#SBATCH --gpus=geforce_rtx_2080_ti:3
#SBATCH --qos=batch

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT

# Activate everything you need
module load cuda/11.1
pyenv activate open

CUDA_VISIBLE_DEVICES=0, 1, 2,
# train in waymo
#srun python train.py \
#--launcher slurm \
#--tcp_port $PORT \
#--cfg_file cfgs/waymo_models/pv_rcnn.yaml \
#--batch_size 4 \
#--extra_tag pvrcnn_resized_gt \
#--max_ckpt_save_num 5

# test waymo model in multi gpu
srun python test.py \
--launcher slurm \
--tcp_port $PORT \
--cfg_file cfgs/waymo_models/pv_rcnn.yaml \
--ckpt /no_backups/s1420/open_output/waymo_models/pv_rcnn/pvrcnn_resized_gt/ckpt/checkpoint_epoch_30.pth \
--extra_tag pvrcnn_resized_gt_eval_on_resized_kirk



# train in kitti
#srun python train.py \
#--launcher slurm \
#--tcp_port $PORT \
#--cfg_file cfgs/kitti_models/pv_rcnn_2.yaml \
#--extra_tag H125_N5_F1_no_GT_yes_Shape \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

# test kitti model in multi gpu
#srun python test.py \
#--launcher slurm \
#--tcp_port $PORT \
#--cfg_file cfgs/kitti_models/pv_rcnn.yaml \
#--ckpt /no_backups/s1420/open_output/kitti_models/pv_rcnn/pvrcnn_filter_by_1/ckpt/checkpoint_epoch_72.pth \
#--extra_tag pvrcnn_filter_by_1_eval_on_16

