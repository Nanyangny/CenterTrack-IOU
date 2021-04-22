import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment
from numba import jit
import copy
from cython_bbox import bbox_overlaps as bbox_ious



class Tracker(object):
    def __init__(self, opt):
        self.opt = opt
        self.reset()

    def init_track(self, results):
        for item in results:
            if item['score'] > self.opt.new_thresh:
                self.id_count += 1
                # active and age are never used in the paper
                item['active'] = 1
                item['age'] = 1
                item['apr'] = item['wh'][0] / item['wh'][1]
                item['tracking_id'] = self.id_count
                item['wh_sum']= item['wh'][0] + item['wh'][1]
                if not ('ct' in item):
                    bbox = item['bbox']
                    item['ct'] = [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
                self.tracks.append(item)

    def reset(self):
        self.id_count = 0
        self.tracks = []

    def step(self, results, public_det=None):
        N = len(results)
        M = len(self.tracks)
        dets = np.array(
            [det['ct'] + det['tracking'] for det in results],
            np.float32)  # N x 2  ## to get back to prev dets for comparison
        track_size = np.array([((track['bbox'][2] - track['bbox'][0]) * \
                                (track['bbox'][3] - track['bbox'][1])) \
                               for track in self.tracks], np.float32)  # M



        ## wh_sum criteria
        det_whsum = np.array([(det['wh'][0] + det['wh'][1]) for det in results], np.float32)
        track_whsum = np.array([track['wh_sum'] for track in self.tracks],np.float)
        whsum_dist = wh_sum_distance(track_whsum,det_whsum)
        whsum_dist[whsum_dist>0.1]= 1e10
        # whsum_dist *=10

        if self.opt.tracking_ltrb_amodal:
            ## consider tracking_wh offset from prev wh directly     version 1
            dets_bboxes = np.array([det['tracking_bboxes'] for det in results ], np.float32)

            ## consider tracking_wh offset from dets (ct + tracking)    version 2
        elif self.opt.tracking_wh:
                dets_bboxes = np.array([ct_ltbr_with_tracking_wh(det['ct'],det['tracking'],det['tracking_wh']) for det in results])
        # else:
            ## consider det and tracked iou directly  version 6, 7
            # dets_bboxes = np.array([det['bbox'] for det in results],np.float)

        # dets_size = np.array([((det_box[2]-det_box[0])*(det_box[3]-det_box[1]))for det_box in dets_bboxes], np.float32)

        if self.opt.tracking_ltrb_amodal or self.opt.tracking_wh:
            track_bboxes = np.array([track['bbox'] \
                                    for track in self.tracks],np.float32)  # M X 4
            iou_distance = ious_distance(dets_bboxes, track_bboxes) # N X M
        if self.opt.iou:
            iou_threshold = np.array([det['iou'] for det in results],np.float32).reshape(-1,1)
            iou_distance_invalid_mask = iou_distance > iou_threshold

        track_cat = np.array([track['class'] for track in self.tracks], np.int32)  # M
        item_size = np.array([((item['bbox'][2] - item['bbox'][0]) * \
                               (item['bbox'][3] - item['bbox'][1])) \
                              for item in results], np.float32)  # N
        item_cat = np.array([item['class'] for item in results], np.int32)  # N

        tracks = np.array(
            [pre_det['ct'] for pre_det in self.tracks], np.float32)  # M x 2

        ## apr criteria
        track_apr = np.array([((track['bbox'][2] - track['bbox'][0]) / \
                                (track['bbox'][3] - track['bbox'][1])) \
                               for track in self.tracks], np.float32)  # M

        det_apr = np.array([(det['wh'][0]/det['wh'][1]) for det in results], np.float32)
        apr_dist = aspect_ratio_distance(track_apr, det_apr) * 10
        apr_dist[apr_dist>self.opt.apr_thresh] = 1e18
        # smooth_sizes = (item_size + dets_size)/2

        size_dist = size_distance(track_size,item_size)
        track_age = np.array([track['age'] for track in self.tracks], np.int32).reshape(1,-1)
        if self.opt.track_age:
            size_dist[(track_age>self.opt.track_age_cut)*size_dist>1]=1e18
            size_dist[(track_age<=self.opt.track_age_cut)*size_dist>0.7]=1e18

        else:
            size_dist[size_dist>self.opt.size_thresh]=1e18

        dist = (((tracks.reshape(1, -1, 2) - \
                  dets.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M



        # dist = iou_distance + iou_distance_invalid_mask * 1e18
        ### combined association

        if self.opt.combined:
            ## version 3
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M)) + \
                       iou_distance_invalid_mask) > 0
            dist = dist + invalid * 1e18

        elif self.opt.iou_first:
            ## iou_dist only version version 2/ 4 / 7 / 8
            if self.opt.iou:
                dist = iou_distance + iou_distance_invalid_mask * 1e18
            else:
                dist = iou_distance

        else:
        ## dist and iou_dist version
            # version 4 /5
            print('dist only')
            invalid = ((dist > track_size.reshape(1, M)) + \
                       (dist > item_size.reshape(N, 1)) + \
                       (item_cat.reshape(N, 1) != track_cat.reshape(1, M))) > 0
            dist = dist + invalid * 1e18

        if self.opt.apr:
            dist += apr_dist

        if self.opt.wh_sum:
            dist += whsum_dist

        if self.opt.size_comparison:
            dist += size_dist



        if self.opt.hungarian:
            item_score = np.array([item['score'] for item in results], np.float32)  # N
            dist[dist > 1e18] = 1e18
            matched_indices = linear_assignment(dist)
        else:
            matched_indices = greedy_assignment(copy.deepcopy(dist))

        unmatched_dets = [d for d in range(dets.shape[0]) \
                          if not (d in matched_indices[:, 0])]
        unmatched_tracks = [d for d in range(tracks.shape[0]) \
                            if not (d in matched_indices[:, 1])]

        if self.opt.hungarian:
            matches = []
            for m in matched_indices:
                if dist[m[0], m[1]] > 1e16:
                    unmatched_dets.append(m[0])
                    unmatched_tracks.append(m[1])
                else:
                    matches.append(m)
            matches = np.array(matches).reshape(-1, 2)
        else:
            matches = matched_indices

        ### reset the list
        ret = []

        ## for matched detection
        for m in matches:
            track = results[m[0]]
            track['tracking_id'] = self.tracks[m[1]]['tracking_id']
            track['age'] = 1
            track['apr'] = (self.tracks[m[1]]['apr'] + det_apr[m[0]])/2
            track['active'] = self.tracks[m[1]]['active'] + 1
            track['wh_sum'] = det_whsum[m[0]]
            ret.append(track)

        ### to further matching using iou
        if self.opt.second_matching:
            if not self.opt.iou_first:
                unmatched_bboxes = dets_bboxes[unmatched_dets]
                unmatched_tracks_bboxes = track_bboxes[unmatched_tracks]

                ## calculate iou_distance
                unmatched_iou_distance = ious_distance(unmatched_bboxes, unmatched_tracks_bboxes)
                unmatched_iou_threshold = iou_threshold[unmatched_dets].reshape(-1, 1)

                unmatched_iou_distance_invalid_mask = unmatched_iou_distance > unmatched_iou_threshold
                unmatched_dist = unmatched_iou_distance + unmatched_iou_distance_invalid_mask * 1e18

            else:
                unmatched_dets2 = dets[unmatched_dets]
                unmatched_tracks2= tracks[unmatched_tracks]

                ## calculate cent_distance
                dist2 = (((unmatched_tracks2.reshape(1, -1, 2) - \
                          unmatched_dets2.reshape(-1, 1, 2)) ** 2).sum(axis=2))  # N x M

                item_cat2 = item_cat[unmatched_dets]
                item_size2 = item_size[unmatched_dets]

                track_cat2 = track_cat[unmatched_tracks]
                track_size2 = track_size[unmatched_tracks]


                invalid = ((dist2 > track_size2.reshape(1, -1)) + \
                           (dist2 > item_size2.reshape(-1, 1)) + \
                           (item_cat2.reshape(-1, 1) != track_cat2.reshape(1, -1))) > 0
                unmatched_dist = dist2 + invalid*1e18


            if self.opt.hungarian:
                matched_indices2 = linear_assignment(unmatched_dist)
            else:
                matched_indices2 = greedy_assignment(copy.deepcopy(unmatched_dist))


            ## association of matched_indices2
            for m in matched_indices2:
                m = np.array([unmatched_dets[m[0]], unmatched_tracks[m[1]]])
                track = results[m[0]]
                track['tracking_id'] = self.tracks[m[1]]['tracking_id']
                track['age'] = 1
                track['apr'] = track['wh'][0]/track['wh'][1]
                track['wh_sum']= det_whsum[m[0]]
                track['active'] = self.tracks[m[1]]['active'] + 1
                ret.append(track)

            ## map back the original unmatched index

            unmatched_dets2 =[unmatched_dets[d] for d in range(len(unmatched_dets)) if not d in matched_indices2[:, 0]]

            unmatched_track2 = [unmatched_tracks[d] for d in range(len(unmatched_tracks)) if not d in matched_indices2[:, 1]]

            ## Private detection: create tracks for all un-matched detections and detection
            for i in unmatched_dets2:
                track = results[i]
                if track['score'] > self.opt.new_thresh:
                    self.id_count += 1
                    track['tracking_id'] = self.id_count
                    track['age'] = 1
                    track['active'] = 1
                    track['apr'] = track['wh'][0]/track['wh'][1]
                    track['wh_sum']= track['wh'][0]+track['wh'][1]
                    ret.append(track)

            for i in unmatched_track2:
                track = self.tracks[i]
                if track['age'] < self.opt.max_age:
                    track['age'] += 1
                    track['active'] = 0
                    bbox = track['bbox']
                    ct = track['ct']
                    v = [0, 0]
                    track['bbox'] = [
                        bbox[0] + v[0], bbox[1] + v[1],
                        bbox[2] + v[0], bbox[3] + v[1]]
                    track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                    ret.append(track)
        else:

            if self.opt.public_det and len(unmatched_dets) > 0:
            # Public detection: only create tracks from provided detections
                pub_dets = np.array([d['ct'] for d in public_det], np.float32)
                dist3 = ((dets.reshape(-1, 1, 2) - pub_dets.reshape(1, -1, 2)) ** 2).sum(
                    axis=2)
                matched_dets = [d for d in range(dets.shape[0]) \
                            if not (d in unmatched_dets)]
                dist3[matched_dets] = 1e18
                for j in range(len(pub_dets)):
                    i = dist3[:, j].argmin()
                    if dist3[i, j] < item_size[i]:
                        dist3[i, :] = 1e18
                        track = results[i]
                        if track['score'] > self.opt.new_thresh:
                            self.id_count += 1
                            track['tracking_id'] = self.id_count
                            track['age'] = 1
                            track['active'] = 1
                            ret.append(track)
            else:
                # Private detection: create tracks for all un-matched detections
                for i in unmatched_dets:
                    track = results[i]
                    if track['score'] > self.opt.new_thresh:
                        self.id_count += 1
                        track['tracking_id'] = self.id_count
                        track['age'] = 1
                        track['wh_sum']=track['wh'][0] + track['wh'][1]
                        track['apr'] = track['wh'][0]/track['wh'][1]
                        track['active'] = 1
                        ret.append(track)

            for i in unmatched_tracks:
                track = self.tracks[i]
                if track['age'] < self.opt.max_age:
                    track['age'] += 1
                    track['active'] = 0
                    bbox = track['bbox']
                    ct = track['ct']
                    v = [0, 0]
                    track['bbox'] = [
                        bbox[0] + v[0], bbox[1] + v[1],
                        bbox[2] + v[0], bbox[3] + v[1]]
                    track['ct'] = [ct[0] + v[0], ct[1] + v[1]]
                    ret.append(track)

        self.tracks = ret
        return ret


def greedy_assignment(dist):
    matched_indices = []
    if dist.shape[1] == 0:
        return np.array(matched_indices, np.int32).reshape(-1, 2)
    for i in range(dist.shape[0]):
        j = dist[i].argmin()
        if dist[i][j] < 1e16:
            dist[:, j] = 1e18
            matched_indices.append([i, j])
    return np.array(matched_indices, np.int32).reshape(-1, 2)


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type detlbrs: list[tlbr] | np.ndarray
    :type tracktlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs),len(btlbrs)))
    if ious.size ==0:
        return ious
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs,dtype=np.float),
        np.ascontiguousarray(btlbrs,dtype=np.float)
    )
    return ious

def ious_distance(atlbrs,btlbrs):
    """
    compute cost based on IoU
    :param atlbrs:
    :param btlbrs:
    :return:
    """
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def ltwh_to_ltbr(bbox_ltwh):
    bbox_ltwh_copy =  bbox_ltwh.copy()
    bbox_ltwh_copy[2:] += bbox_ltwh_copy[:2]
    return bbox_ltwh_copy


def ltbr_with_tracking_wh(det_bbox_ltbr,tracking_wh):
    det_bbox_ltbr_copy = det_bbox_ltbr.copy()
    det_bbox_ltbr_copy += np.array([-tracking_wh[0]/2.0, -tracking_wh[1]/2.0, tracking_wh[0]/2.0, tracking_wh[1]/2.0])
    return det_bbox_ltbr_copy

def ltbr_from_ct_and_wh(det_ct,det_wh):
    dets_ct_ltrb = np.array([det_ct[0] - det_wh[0] / 2.0,
                             det_ct[1] - det_wh[1] / 2.0,
                             det_ct[0] + det_wh[0] / 2.0,
                             det_ct[1] + det_wh[1] / 2.0])
    return dets_ct_ltrb

def ct_ltbr_with_tracking_wh(dets_ct,dets_tracking,tracking_wh=None):
    dets = dets_ct + dets_tracking


    dets_wh_with_tracking_wh = tracking_wh
    dets_ct_ltrb = np.array([dets[0] - dets_wh_with_tracking_wh[0] / 2.0,
                    dets[1] - dets_wh_with_tracking_wh[1] / 2.0,
                    dets[0] + dets_wh_with_tracking_wh[0] / 2.0,
                    dets[1] + dets_wh_with_tracking_wh[1] / 2.0])

    return dets_ct_ltrb


def aspect_ratio_distance(a_aspr,b_aspr):
    return abs(a_aspr.reshape(1,-1) -b_aspr.reshape(-1,1))/a_aspr.reshape(1,-1)


def wh_sum_distance(a_wh_sum,b_wh_sum):
    return abs(a_wh_sum.reshape(1,-1) - b_wh_sum.reshape(-1,1))/a_wh_sum.reshape(1,-1)

def size_distance(a_size,b_size):
    return abs(a_size.reshape(1,-1) - b_size.reshape(-1,1))/a_size.reshape(1,-1)