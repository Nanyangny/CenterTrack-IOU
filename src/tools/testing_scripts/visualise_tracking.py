import json
import numpy as np
from collections import defaultdict
import os
import cv2
from progress.bar import Bar

# ann_path = "/home/students/acct1001_05/Dataset/mot17/annotations/test.json"
# data_path = "/home/students/acct1001_05/Dataset/Detrac/images/test/" ## UAD
data_path = "/home/students/acct1001_05/Dataset/Own/test/" ## own
# mot17_half_PAPER
# trk_ltrb_amodal_repeat_70_epoch_iou_MG30
result_path = "/home/students/acct1001_05/CenterTrack_MOT_Paper/models/tracking/OwnData_20_FYP_iou/results/"
output_img_path = "/home/students/acct1001_05/CenterTrack_MOT_Paper/models/tracking/OwnData_20_FYP_iou/test_visualize/"
color_list = np.array(
    [1.000, 1.000, 1.000,
     0.850, 0.325, 0.098,
     0.929, 0.694, 0.125,
     0.494, 0.184, 0.556,
     0.466, 0.674, 0.188,
     0.301, 0.745, 0.933,
     0.635, 0.078, 0.184,
     0.300, 0.300, 0.300,
     0.600, 0.600, 0.600,
     1.000, 0.000, 0.000,
     1.000, 0.500, 0.000,
     0.749, 0.749, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 1.000,
     0.667, 0.000, 1.000,
     0.333, 0.333, 0.000,
     0.333, 0.667, 0.000,
     0.333, 1.000, 0.000,
     0.667, 0.333, 0.000,
     0.667, 0.667, 0.000,
     0.667, 1.000, 0.000,
     1.000, 0.333, 0.000,
     1.000, 0.667, 0.000,
     1.000, 1.000, 0.000,
     0.000, 0.333, 0.500,
     0.000, 0.667, 0.500,
     0.000, 1.000, 0.500,
     0.333, 0.000, 0.500,
     0.333, 0.333, 0.500,
     0.333, 0.667, 0.500,
     0.333, 1.000, 0.500,
     0.667, 0.000, 0.500,
     0.667, 0.333, 0.500,
     0.667, 0.667, 0.500,
     0.667, 1.000, 0.500,
     1.000, 0.000, 0.500,
     1.000, 0.333, 0.500,
     1.000, 0.667, 0.500,
     1.000, 1.000, 0.500,
     0.000, 0.333, 1.000,
     0.000, 0.667, 1.000,
     0.000, 1.000, 1.000,
     0.333, 0.000, 1.000,
     0.333, 0.333, 1.000,
     0.333, 0.667, 1.000,
     0.333, 1.000, 1.000,
     0.667, 0.000, 1.000,
     0.667, 0.333, 1.000,
     0.667, 0.667, 1.000,
     0.667, 1.000, 1.000,
     1.000, 0.000, 1.000,
     1.000, 0.333, 1.000,
     1.000, 0.667, 1.000,
     0.167, 0.000, 0.000,
     0.333, 0.000, 0.000,
     0.500, 0.000, 0.000,
     0.667, 0.000, 0.000,
     0.833, 0.000, 0.000,
     1.000, 0.000, 0.000,
     0.000, 0.167, 0.000,
     0.000, 0.333, 0.000,
     0.000, 0.500, 0.000,
     0.000, 0.667, 0.000,
     0.000, 0.833, 0.000,
     0.000, 1.000, 0.000,
     0.000, 0.000, 0.000,
     0.000, 0.000, 0.167,
     0.000, 0.000, 0.333,
     0.000, 0.000, 0.500,
     0.000, 0.000, 0.667,
     0.000, 0.000, 0.833,
     0.000, 0.000, 1.000,
     0.333, 0.000, 0.500,
     0.143, 0.143, 0.143,
     0.286, 0.286, 0.286,
     0.429, 0.429, 0.429,
     0.571, 0.571, 0.571,
     0.714, 0.714, 0.714,
     0.857, 0.857, 0.857,
     0.000, 0.447, 0.741,
     0.50, 0.5, 0
     ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255
colors = [(color_list[i]).astype(np.uint8) for i in range(len(color_list))]
# print(colors[:10])



def add_box(img=None,bbox=None,cat=None,show_dots=True,c=None,conf=1,tracking_id =None,no_bbox=False,show_txt=True):
    if show_dots:
        ct = (int(bbox[0] + (bbox[2]) / 2), int(bbox[1] + (bbox[3]) / 2))
        cv2.circle(
            img, ct, 5, c, -1, lineType=cv2.LINE_AA)
    if conf >= 1:
        ID = int(conf)
        txt = '{}{}'.format(names[cat], ID)
    elif not tracking_id:
        txt = '{}{:.1f}'.format(names[cat], conf)
    else:
        txt = '{}{:.1f} ID:{}'.format(names[cat], conf,tracking_id)
    thickness = 2
    fontsize = 1

    font = cv2.FONT_HERSHEY_SIMPLEX
    cat_size = cv2.getTextSize(txt, font, fontsize, thickness)[0]
    if not no_bbox:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                        c, thickness)

    if show_txt:
        cv2.rectangle(img,
                      (bbox[0], bbox[1] - cat_size[1] - thickness),
                      (bbox[0] + cat_size[0], bbox[1]), c, -1)
        cv2.putText(img,
                    txt,
                    (bbox[0], bbox[1] - thickness - 1),
                    font, fontsize, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
    return img

def draw_bbox(img, bboxes):
  for bbox in bboxes:
    cv2.rectangle(img, (int(bbox[0]), int(bbox[1])),
      (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
      colors[int(bbox[4])% len(colors)].tolist(), 2, lineType=cv2.LINE_AA)
    ct = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
    txt = '{}'.format(bbox[4])
    cv2.putText(img, txt, (int(ct[0]), int(ct[1])),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 122, 255), thickness=2, lineType=cv2.LINE_AA)

def save_img(path=output_img_path, seq=None, out_img=None,imgId=None):
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(os.path.join(path,seq)):
        os.mkdir(os.path.join(path,seq))
    output_modified = os.path.join(path,seq)
    cv2.imwrite(output_modified + '/{:06d}.jpg'.format(imgId), out_img)


category_map = {
    'car': 1,
    'bus': 2,
    'van': 3,
    'others': 4, }


# names = {1: 'Sedan',
#          2: 'Suv',
#          3: 'Van',
#          4: 'Taxi',
#          5: 'Bus',
#          6: 'Truck'}


def visualise_result(data_path= data_path,result_path=result_path,num_seq=None):
    seqs = [seq[:-4] for seq in os.listdir(result_path)]
    print(sorted(seqs))
    num_seq=len(seqs)
    val_dic ={
        # "MOT17-02-FRCNN":299,
        # "MOT17-04-FRCNN":524,
        # "MOT17-05-FRCNN":418,
        # "MOT17-09-FRCNN":263,
        # "MOT17-10-FRCNN":326,
        # "MOT17-13-FRCNN":368,
        ## UA-D viz
        # "MVI_40761":500,
        # "MVI_40863":500,
        # "MVI_40901":500,
        # "MVI_40762":500,
        # "MVI_40864":500,
        # "MVI_40763": 500
        ## Own data
        # "MVI_2168":500
        "MVI_2211":200
    }

    for seq in sorted(seqs)[:num_seq]:

        if seq not in val_dic.keys():
            print(f'{seq} not in val')
            continue
        anns = np.loadtxt(os.path.join(result_path,seq+".txt"), dtype=np.float32, delimiter=',')
        print('anns shape', anns.shape)
        image_to_anns = defaultdict(list)
        if seq.startswith("MVI_2"):
            img_path = data_path + seq + '/img1/'
        elif seq.startswith("MVI_"):
            img_path = data_path + seq +'/'
        else:
            img_path = data_path + seq + '/img1/'
        images = os.listdir(img_path)
        num_images = len([image for image in images if 'jpg' in image])

        for i in range(anns.shape[0]):
            if seq.startswith("MVI_"):
                frame_id = int(anns[i][0])
            else:
                frame_id = int(anns[i][0])+num_images-val_dic[seq]-1
            track_id = int(anns[i][1])
            bbox = (anns[i][2:6]).tolist()

            image_to_anns[frame_id].append(bbox + [track_id])


        bar = Bar('Processing '+ seq, max=val_dic[seq])
        for i in range(val_dic[seq]):
            if seq.startswith("MVI_2"):
                frame_id = i + 1
                file_name = '{}/img1/{:06d}.jpg'.format(seq, frame_id)  ### for UA-D viz
            elif seq.startswith("MVI_"):
                frame_id = i+1
                file_name = '{}/img{:05d}.jpg'.format(seq, frame_id) ### for UA-D viz
            else:
                frame_id = num_images - val_dic[seq] + i
                file_name = '{}/img1/{:06d}.jpg'.format(seq, frame_id)
            file_path = data_path + file_name
            img = cv2.imread(file_path)
            draw_bbox(img, image_to_anns[frame_id])
            # input_img = add_box(img=input_img[:],cat=cat, bbox=bbox, c=c, conf=conf, tracking_id=track_id)
            save_img(imgId=frame_id, out_img=img[:],seq=seq)
            bar.next()
        bar.finish()


if __name__ =="__main__":
    visualise_result()





