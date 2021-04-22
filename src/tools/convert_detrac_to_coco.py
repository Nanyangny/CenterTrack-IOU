import os
import numpy as np
import json
import cv2
import xml.etree.ElementTree as ET

# Use the same script for UA-DETRAC
# DATA_PATH = '../../data/mot16/'
DATA_PATH = '/home/students/acct1001_05/Dataset/Detrac/'
OUT_PATH = DATA_PATH + 'annotations/'
SPLITS = ['test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = False
CREATE_SPLITTED_DET = False
vehicle_map_v3 = {"Sedan": 1, "Police": 1,
               "Suv": 2, "Hatchback": 2,
               "Van": 3, "MiniVan": 3,
               "Taxi": 4,
               "Bus": 5,
               "Truck": 6,
               "Truck-Box-Large": 6,
               "Truck-Box-Med": 6,
               "Truck-Flatbed": 6,
               "Truck-Pickup": 6,
               "Truck-Util": 6}
category_map = {
    'car': 1,
    'bus': 2,
    'van': 3,
    'others': 4, }

if __name__ == '__main__':
    for split in SPLITS:
        data_path = DATA_PATH + "images/" + split
        out_path = OUT_PATH + '{}.json'.format(split)
        out = {'images': [], 'annotations': [],
               'categories': [{'id': 1, 'name': 'car'},
                              {'id': 2, 'name': 'bus'},
                              {'id': 3, 'name': 'van'},
                              {'id': 4, 'name': 'others'}],
               'videos': []}
        seqs = os.listdir(data_path)  ## sequence file
        image_cnt = 0
        ann_cnt = 0
        video_cnt = 0
        ### run each seq
        for seq in sorted(seqs):
            if '.DS_Store' in seq:
                continue
            # if 'mot17' in DATA_PATH and (split != 'test' and not ('FRCNN' in seq)):
            # continue
            video_cnt += 1
            out['videos'].append({
                'id': video_cnt,
                'file_name': seq})
            seq_path = '{}/{}/'.format(data_path, seq)
            img_path = seq_path
            images = os.listdir(img_path)
            num_images = len([image for image in images if 'jpg' in image])
            if HALF_VIDEO and ('half' in split):
                image_range = [0, num_images // 2] if 'train' in split else \
                    [num_images // 2 + 1, num_images - 1]
            else:
                image_range = [0, num_images - 1]
            for i in range(num_images):
                if (i < image_range[0] or i > image_range[1]):
                    continue
                image_info = {'file_name': '{}/img{:05d}.jpg'.format(seq, i + 1),
                              'id': image_cnt + i + 1,
                              'frame_id': i + 1 - image_range[0],
                              'prev_image_id': image_cnt + i if i > 0 else -1,
                              'next_image_id': \
                                  image_cnt + i + 2 if i < num_images - 1 else -1,
                              'video_id': video_cnt}
                out['images'].append(image_info)
            print('{}: {} images'.format(seq, num_images))

            if split != 'test':
                input_ann_path = DATA_PATH + "/origannotation/train/" + seq + ".xml"
                tree = ET.parse(input_ann_path)
                root = tree.getroot()
                ann_total = int(len(root.findall("./frame/target_list/target")))
                print(' {} ann images'.format(ann_total))
                ignored_regions = []
                for child in root:
                    ## find the ignored_region
                    if child.tag == "ignored_region":
                        for node in child:
                            ignored_regions.append(np.array(([float(val) for val in node.attrib.values()])))

                    if child.tag == "frame":
                        target_list = child[0]
                        frame_id = int(child.attrib['num'])

                        for target in target_list:
                            ann_cnt += 1
                            track_id = int(target.attrib['id'])
                            bbox = np.array([float(i) for i in target[0].attrib.values()])
                            # bbox = bbox[[1, 2, 3, 0]]  ## for v3
                            category_id = category_map[target[1].attrib['vehicle_type']]
                            ann = {'id': ann_cnt,
                                   'category_id': category_id,
                                   'image_id': image_cnt + frame_id,
                                   'track_id': track_id,
                                   'bbox': bbox.tolist(),
                                   'conf': 1}
                            out['annotations'].append(ann)

                        for region in ignored_regions:
                                ann_cnt += 1
                                ann = {'id': ann_cnt,
                                       'category_id': 0,
                                       'image_id': image_cnt + frame_id,
                                       'track_id': -1,
                                       'bbox': region.tolist(),
                                       'conf': 1}
                                out['annotations'].append(ann)

            image_cnt += num_images
        print('loaded {} for {} images and {} samples'.format(
            split, len(out['images']), len(out['annotations'])))
        json.dump(out, open(out_path, 'w'))
