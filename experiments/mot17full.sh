
### full train on MOT17 training set
python main.py tracking --exp_id trk_ltrb_amodal_full_70epoch --dataset mot --dataset_version 17trainval --pre_hm --ltrb_amodal --tracking_ltrb_amodal --iou --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1  --num_epochs 70 --gpus 0,1,2 --batch_size 16 --load_model ../models/crowdhuman.pth

## test on full using single IOU association => CenterTrack++ for MAX_AGE = 30 
python test.py tracking --exp_id trk_ltrb_amodal_full_70_epoch_iou_MA30_SUBMISSION_FINAL --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou  --tracking_ltrb_amodal --iou_first --max_age 30 --load_model ../models/tracking/trk_ltrb_amodal_full_70epoch/model_last.pth

## test on full using original CenterTrack association method for MAX_AGE = 30 
python test.py tracking --exp_id mot17_PAPER --dataset mot --dataset_version 17test --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --max_age 30 --load_model ../models/mot17_fulltrain.pth