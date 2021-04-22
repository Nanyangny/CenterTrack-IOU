python eval_motchallenge.py /home/students/acct1001_05/Dataset/mot17/train/ /home/students/acct1001_05/CenterTrack_MOT_Paper/models/tracking/DirectTracking_10_iou_then_dist/results_mot17halfval --eval_official
              '../../data/mot{}/{}/ '.format(self.year, 'train') + \
              '{}/results_mot{}/ '.format(save_dir, self.dataset_version) + \
              gt_type_str + ' --eval_official')