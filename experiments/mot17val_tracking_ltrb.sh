cd src
### finetune on the pretrained model trained on CrowdHuman dataset for 70 epoches 
## adding two output branches: iou and tracking_ltrb 
python main.py tracking --exp_id trk_ltrb_70 --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --tracking_ltrb_amodal --iou --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1  --num_epochs 70 --gpus 0,1,2 --batch_size 16 --load_model ../models/crowdhuman.pth


### ablative studies for different association order and distance matrix
### using tracking_ltrb approach
python test.py tracking --exp_id trk_ltrb_base_MG30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --max_age 30 --load_model ../models/tracking/trk_ltrb_70/model_last.pth
python test.py tracking --exp_id trk_ltrb_combined_MG30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou  --tracking_ltrb_amodal --combined --max_age 30 --load_model ../models/tracking/trk_ltrb_70/model_last.pth
python test.py tracking --exp_id trk_ltrb_dis_then_iou_MG30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou  --tracking_ltrb_amodal --second_matching --max_age 30 --load_model ../models/tracking/trk_ltrb_70/model_last.pth
python test.py tracking --exp_id trk_ltrb_iou_then_dis_MG30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou  --tracking_ltrb_amodal --second_matching --iou_first --max_age 30 --load_model ../models/tracking/trk_ltrb_70/model_last.pth
python test.py tracking --exp_id trk_ltrb_iou_MG30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou  --tracking_ltrb_amodal --iou_first --max_age 30 --load_model ../models/tracking/trk_ltrb_70/model_last.pth