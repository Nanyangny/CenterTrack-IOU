from sklearn import metrics
from scipy import interpolate
import numpy as np
import os
import torch
def eval(gt_path='/home/students/acct1001_05/Dataset/mot17/train', pred_path='/home/students/acct1001_05/CenterTrack_MOT_Paper/models/tracking/CH_Pretrain_iou_apr_cost1_MG30_tracking_wh/results_mot17halfval'):
    seqs = os.listdir(pred_path)

    for seq in seqs:
        print(seq)
        if not seq=="eval_res.txt":
            pred_ids, gt_ids = [], []
            gt_seq_path=os.path.join(gt_path,seq[:-4],'gt','gt.txt')
            gts = np.loadtxt(gt_seq_path,dtype=np.float32, delimiter=',')
            preds = np.loadtxt(os.path.join(pred_path,seq),dtype=np.float32, delimiter=',')
            pred_ids.extend([ item[1] for item in preds ])
            gt_ids.extend([ item[1] for item in gts])
            print(f'{seq} done')

            n = len(pred_ids)
            ids = np.zeros((n,), dtype=np.int64)
            for k in range(len(pred_ids)):
                ids[k] = pred_ids[k]

            id_labels = torch.LongTensor(gt_ids)
            pdist = torch.LongTensor(pred_ids)
            # gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()
            gt = pdist.expand(n, n).eq(pdist.expand(n, n).t()).numpy()
            pdist = pdist.expand(n, n).eq(pdist.expand(n, n).t()).numpy()
            up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
            pdist = pdist[up_triangle]
            gt = gt[up_triangle]

            far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
            far, tar, threshold = metrics.roc_curve(gt, pdist)
            interp = interpolate.interp1d(far, tar)
            tar_at_far = [interp(x) for x in far_levels]
            for f, fa in enumerate(far_levels):
                print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))

    # return tar_at_far


if __name__ =="__main__":
    eval()