#!/bin/bash -l

# Slurm parameters
#SBATCH --job-name=kitti
#SBATCH --output=votr_kitti_result/only_Downsample_eval_on_32_%j.%N.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=168:00:00
#SBATCH --mem=50G
#SBATCH --gpus=rtx_a5000:1
#SBATCH --qos=batch

# Activate everything you need
module load cuda/11.1
pyenv activate open


#python resize_label.py --cfg_file=cfgs/waymo_models/pv_rcnn.yaml --extra_tag=resize_box


# generate waymo infos
#python -m pcdet.datasets.waymo.waymo_dataset --func create_waymo_infos --cfg_file cfgs/dataset_configs/waymo_dataset.yaml


# train votr
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--cfg_file cfgs/kitti_models/votr_tsd_0.yaml \
#--batch_size 2 \
#--extra_tag only_Downsample \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

# test votr
python test.py --cfg_file cfgs/kitti_models/votr_tsd_0.yaml \
--ckpt /no_backups/s1420/open_output/kitti_models/votr_tsd_0/only_Downsample/ckpt/checkpoint_epoch_76.pth \
--extra_tag only_Downsample_eval_on_32



# train pvrcnn
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--cfg_file cfgs/kitti_models/pv_rcnn_2.yaml \
#--batch_size 2 \
#--extra_tag H125_N5_F1_no_GT_yes_Shape_yes_Downsample \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

# test pvrcnn
#python test.py --cfg_file cfgs/kitti_models/pv_rcnn_3.yaml \
#--ckpt /no_backups/s1420/open_output/kitti_models/pv_rcnn_3/H125_N5_F5_no_GT_yes_Shape/ckpt/checkpoint_epoch_67.pth \
#--extra_tag H125_N5_F5_no_GT_yes_Shape_eval_on_16



#train second
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--cfg_file cfgs/kitti_models/second.yaml \
#--extra_tag default_lr_no_gt_H0625_N9 \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

# test second
#python test.py --cfg_file cfgs/kitti_models/second.yaml \
#--ckpt /no_backups/s1420/open_output/kitti_models/second/default_lr_no_gt_H0625_N4/ckpt/checkpoint_epoch_77.pth \
#--extra_tag default_lr_no_gt_H0625_N4_eval_on_16



#train secondiou
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--cfg_file cfgs/kitti_models/second_iou_0.yaml \
#--extra_tag smaller_lr_cia_ssd_only_Downsample \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

#test secondiou
#python test.py --cfg_file cfgs/kitti_models/second_iou_aware.yaml \
#--ckpt /no_backups/s1420/open_output/kitti_models/second_iou/smaller_lr_cia_ssd_H1_N5_F5_no_GT_yes_Shape_yes_Downsample/ckpt/checkpoint_epoch_78.pth \
#--extra_tag smaller_lr_cia_ssd_H1_N5_F5_no_GT_yes_Shape_yes_Downsample_eval_on_16



#train voxel rcnn
#CUDA_VISIBLE_DEVICES=0 python train.py \
#--cfg_file cfgs/kitti_models/voxel_rcnn_3class_.yaml \
#--extra_tag only_Downsample \
#--max_ckpt_save_num 15 \
#--num_epochs_to_eval 15

#test voxel rcnn
#python test.py --cfg_file cfgs/kitti_models/voxel_rcnn_3class_.yaml \
#--ckpt /no_backups/s1420/open_output/kitti_models/voxel_rcnn_3class_/only_Downsample/ckpt/checkpoint_epoch_77.pth \
#--extra_tag only_Downsample_eval_on_16