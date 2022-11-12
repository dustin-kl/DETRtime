date=$(date +%y-%m-%d-%H-%M)    
modelname=$'movie_3class'

# nohup python main.py \
python backbone_embedding.py
	--data_path /home/ubuntu/projects/data/ICML_movie_min_segmentation_minseq_500_margin_1_amp_thresh_10000/tensors \
	--backbone inception_time \
	--lr_backbone 1e-4 \
	--nb_filters 16 \
	--use_residuals False \
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
	--output_dir ./runs/"$modelname" &


# lr=0.0001, wandb_dir='noname', lr_backbone=0.0001, batch_size=32, weight_decay=0.0001, epochs=300, lr_drop=50, clip_max_norm=0.1, 
# backbone='inception_time', kernel_size=16, nb_filters=16, in_channels=129, out_channels=1, backbone_depth=6, use_residual=True, 
# position_embedding='sine', back_channels=16, back_layers=12, enc_layers=6, dec_layers=6, dim_feedforward=2048, hidden_dim=128, 
# dropout=0.1, nheads=8, num_queries=20, pre_norm=False, set_cost_class=1, set_cost_bbox=10, set_cost_giou=2, bbox_loss_coef=10.0, 
# giou_loss_coef=2.0, eos_coef=0.2, num_classes=3, timestamps=500, timestamps_output=500, data_path='/home/ubuntu/projects/data/ICML_zuco_min_segmentation_minseq_500_margin_1_amp_thresh_10000/tensors', 
# scaler=None, output_dir='./runs/Resume_dustin_supermodel_zuco_22-05-16-08-24', device='cuda', seed=42, resume='/home/ubuntu/projects/EEGEyeNet_experimental/runs/Resume_dustin_supermodel_zuco/checkpoint_best_val.pth', 
# start_epoch=0, eval=False, num_workers=2