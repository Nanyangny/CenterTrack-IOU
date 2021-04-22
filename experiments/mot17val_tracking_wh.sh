cd src
### finetune on the pretrained model trained on CrowdHuman dataset for 70 epoches 
python main.py tracking --exp_id tracking_wh_70 --dataset mot --dataset_version 17halftrain --pre_hm --ltrb_amodal --iou --tracking_wh --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1  --gpus 0,1,2  --num_epochs 70 --batch_size 16 --load_model ../models/crowdhuman.pth



### ablative studies for different association order and distance matrix
### using tracking_wh approach
python test.py tracking --exp_id tracking_wh_iou_MA30 --dataset mot --dataset_version 17halfval --ltrb_amodal --pre_hm --track_thresh 0.4 --pre_thresh 0.5 --iou --tracking_wh --iou_first  --max_age 30 --load_model ../models/tracking/tracking_wh_70/model_last.pth
python test.py tracking --exp_id tracking_wh_dis_MA30 --dataset mot --dataset_version 17halfval --ltrb_amodal --pre_hm --track_thresh 0.4 --pre_thresh 0.5  --max_age 30 --load_model ../models/tracking/tracking_wh_70/model_last.pth
python test.py tracking --exp_id tracking_wh_combined_MA30 --dataset mot --dataset_version 17halfval --ltrb_amodal --pre_hm --track_thresh 0.4 --pre_thresh 0.5 --iou --tracking_wh --max_age 30 --combined --load_model ../models/tracking/tracking_wh_70/model_last.pth
python test.py tracking --exp_id tracking_wh_70_iou_then_dist_MA30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou --tracking_wh --load_model ../models/tracking/tracking_wh_70/model_last.pth --second_matching --iou_first --max_age 30
python test.py tracking --exp_id tracking_wh_70_dis_then_iou_MA30 --dataset mot --dataset_version 17halfval --pre_hm --ltrb_amodal --track_thresh 0.4 --pre_thresh 0.5 --iou --tracking_wh --load_model ../models/tracking/tracking_wh_70/model_last.pth --second_matching  --max_age 30