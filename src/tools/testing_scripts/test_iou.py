from cython_bbox import bbox_overlaps as bbox_ious
import numpy as np
import time

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


tracks = np.array([[0,0,5,5],[5,5,10,10]],dtype=np.float)
dets = np.array([[5,5,9,9],[3.5,3.5,5,5]],dtype=np.float)
threshold = np.array([0.5,0.9]).reshape(-1,1)


cost_matrix=ious_distance(dets,tracks)
# print(cost_matrix)
# print(cost_matrix > threshold)
# print(cost_matrix)


a =np.array([1,2,3,4,5])
b = np.array([2,4,5,6,4])
print((a+b)/2)