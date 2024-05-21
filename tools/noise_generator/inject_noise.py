### Injecting noise into a clean dataset; by Haoran Zhu
from pycocotools.coco import COCO
import os
import mmcv
from tqdm import tqdm
import argparse

import numpy as np   
import json
import numpy as np
from aitod_dataset import AitodDataset  ### must for load AI-TOD-v1.0 or AI-TOD-v2.0 Dataset
from dota_dataset import DotaDataset    ### must for load DOTA-v2.0 Dataset

def compute_iou(rec_1, rec_2):
    rec1 = (rec_1[0], rec_1[1], rec_1[0]+rec_1[2], rec_1[1]+rec_1[3])
    rec2 = (rec_2[0], rec_2[1], rec_2[0]+rec_2[2], rec_2[1]+rec_2[3])
    """
    computing IoU
    rec1: (x0, y0, x1, y1)
    rec2: (x0, y0, x1, y1)
    :return: scala value of IoU
    """
    # computing area of each rectangle
    S_rec1 = (rec1[2] - rec1[0]) * (rec1[3] - rec1[1])
    S_rec2 = (rec2[2] - rec2[0]) * (rec2[3] - rec2[1])

    # computing the sum_area
    sum_area = S_rec1 + S_rec2

    # find the each edge of intersect rectangle
    left_line = max(rec1[1], rec2[1])
    right_line = min(rec1[3], rec2[3])
    top_line = max(rec1[0], rec2[0])
    bottom_line = min(rec1[2], rec2[2])

    # judge if there is an intersect area
    if left_line >= right_line or top_line >= bottom_line:
        return 0
    else:
        intersect = (right_line - left_line) * (bottom_line - top_line)
        return (intersect / (sum_area - intersect)) * 1.0                           


def gen_noise_gt(data_infos, box_noise_level=0.4, mode='first', category_info=None):
    print('\ngenerating noisy ground-truth ...')
    print('noisy degree = ', box_noise_level)
    print('noisy mode = ', mode)
    num_classes = len(category_info)
    data_len = len(data_infos)
    p = [1 - box_noise_level, box_noise_level]     
    p = np.array(p)
    p_third = []
    p_forth = []
    class_id_third = []
    class_id_forth = []
    for k in range(num_classes):
        p_third.append(0.0)
        p_forth.append(1/num_classes)
        class_id_third.append(k)
        class_id_forth.append(k)
    
    p_forth = np.array(p_forth)
    class_id = [0, 1]

    p_judge_extra = [1 - box_noise_level, box_noise_level]  
    p_judge_extra = np.array(p_judge_extra)
    judge_matrix = [0, 1]

    p_judge_bbox = [1 - box_noise_level, box_noise_level]  
    p_judge_bbox = np.array(p_judge_bbox)
    judge_matrix_bbox = [0, 1]

    # for idx in range(data_len):
    for idx in tqdm(range(data_len)):
        img_w, img_h = data_infos[idx]['width'], data_infos[idx]['height']
        anno = data_infos[idx]['ann']
        bboxes = anno['bboxes']
        category_id = anno['labels']
        # perturb bbox
        if box_noise_level > 0:
            noisy_bboxes = []
            category = []
            is_cate_noise = []
            is_bbox_noise = []
            offset = []
            ori_category_id = []
            for i, bbox in enumerate(bboxes):
                x1, y1, x2, y2 = bbox
                x, y, w, h = x1, y1, x2-x1, y2-y1
                cx, cy = (x1+x2)/2, (y1+y2)/2
                cate = category_id[i]
                index = np.random.choice(class_id, p = p.ravel())
                p_third[cate] = 1 - box_noise_level
                for mn in range(0, len(p_third)):
                    if mn != cate:
                        p_third[mn] = box_noise_level/(num_classes-1)
                p_third = np.array(p_third)
                if sum(p_third) != 1:
                    p_third[-1] += 1 - sum(p_third)
                if sum(p_forth) != 1:
                    p_forth[-1] += 1 - sum(p_forth)
                index_third = np.random.choice(class_id_third, p = p_third.ravel())
                index_forth = np.random.choice(class_id_forth, p = p_forth.ravel())
                index_judge = np.random.choice(judge_matrix, p = p_judge_extra.ravel())
                index_judge_bbox = np.random.choice(judge_matrix_bbox, p = p_judge_bbox.ravel())
                # shift bbox x-y
                xy_rand_range, wh_rand_range = box_noise_level, box_noise_level
                x_offset = w * np.random.uniform(-xy_rand_range, xy_rand_range)
                y_offset = h * np.random.uniform(-xy_rand_range, xy_rand_range)
                noisy_cx, noisy_cy = cx + x_offset, cy + y_offset

                # scale bbox w-h
                noisy_w = w * (np.random.uniform(-wh_rand_range, wh_rand_range) + 1.0)
                noisy_h = h * (np.random.uniform(-wh_rand_range, wh_rand_range) + 1.0)

                # noisy coordinates
                noisy_x1, noisy_y1, noisy_x2, noisy_y2 = max(0, noisy_cx - noisy_w/2), max(0, noisy_cy - noisy_h/2), min(noisy_cx + noisy_w/2, img_w - 1), min(noisy_cy + noisy_h/2, img_h-1)
                noisy_x1 = round(noisy_x1)
                noisy_y1 = round(noisy_y1)
                noisy_x2 = round(noisy_x2)
                noisy_y2 = round(noisy_y2)

                # noisy bbox xywh
                noisy_x = noisy_x1
                noisy_y = noisy_y1
                noisy_ww = round(noisy_x2 - noisy_x1)
                noisy_hh = round(noisy_y2 - noisy_y1)
                # eliminate invalid noisy box
                if mode == 'second':
                    if noisy_x2 <= noisy_x1 or noisy_y2 <= noisy_y1:
                        noisy_bboxes.append([x, y, w, h])
                        category.append(cate)
                        ori_category_id.append(cate)
                        is_cate_noise.append(0)
                        is_bbox_noise.append(0)
                        offset.append([0.0, 0.0, 0.0, 0.0])
                        continue
                    ori_category_id.append(cate)
                    is_cate_noise.append(0)
                    is_bbox_noise.append(1)
                    offset.append([noisy_x-x, noisy_y-y, noisy_ww-w, noisy_hh-h])
                    noisy_bboxes.append([noisy_x, noisy_y, noisy_ww, noisy_hh])
                    category.append(cate)
                elif mode == 'first':
                    if index == 0:
                        ori_category_id.append(cate)
                        is_cate_noise.append(0)
                        is_bbox_noise.append(0)
                        offset.append([0.0, 0.0, 0.0, 0.0])
                        noisy_bboxes.append([x, y, w, h])
                        category.append(cate)
                elif mode == 'third':
                    ori_category_id.append(cate)
                    is_cate_noise.append(int(cate != index_third))
                    is_bbox_noise.append(0)
                    offset.append([0.0, 0.0, 0.0, 0.0])
                    noisy_bboxes.append([x, y, w, h])
                    category.append(index_third)
                elif mode == 'st':
                    if noisy_x2 <= noisy_x1 or noisy_y2 <= noisy_y1:
                        noisy_bboxes.append([x, y, w, h])
                        category.append(index_third)
                        ori_category_id.append(cate)
                        is_cate_noise.append(int(cate != index_third))
                        is_bbox_noise.append(0)
                        offset.append([0.0, 0.0, 0.0, 0.0])
                        continue
                    ori_category_id.append(cate)
                    is_cate_noise.append(int(cate != index_third))
                    is_bbox_noise.append(1)
                    offset.append([noisy_x-x, noisy_y-y, noisy_ww-w, noisy_hh-h])
                    noisy_bboxes.append([noisy_x, noisy_y, noisy_ww, noisy_hh])
                    category.append(index_third)
                elif mode == 'forth':
                    ori_category_id.append(cate)
                    is_cate_noise.append(0)
                    is_bbox_noise.append(0)
                    offset.append([0.0, 0.0, 0.0, 0.0])
                    noisy_bboxes.append([x, y, w, h])
                    category.append(cate)
                    if index_judge == 1:
                        while(1):
                            x_background = np.random.randint(0,img_w-w)
                            y_background = np.random.randint(0,img_h-h)
                            ok_add = 1
                            for loop_judge in range(0, len(bboxes)):
                                if compute_iou(bboxes[loop_judge], [x_background, y_background, w, h]) >= 0.2:
                                    ok_add = 0
                                    break
                            if ok_add == 1:
                                ori_category_id.append(8)
                                is_cate_noise.append(1)
                                is_bbox_noise.append(0)
                                offset.append([0.0, 0.0, 0.0, 0.0])
                                noisy_bboxes.append([x_background, y_background, w, h])
                                category.append(index_forth)
                                break   
                elif mode == 'all':
                    if index == 0:
                        if noisy_x2 <= noisy_x1 or noisy_y2 <= noisy_y1:
                            noisy_bboxes.append([x, y, w, h])
                            category.append(index_third)
                            ori_category_id.append(cate)
                            is_cate_noise.append(int(cate != index_third))
                            is_bbox_noise.append(0)
                            offset.append([0.0, 0.0, 0.0, 0.0])
                        else:
                            ori_category_id.append(cate)
                            is_cate_noise.append(int(cate != index_third))
                            is_bbox_noise.append(1)
                            offset.append([noisy_x-x, noisy_y-y, noisy_ww-w, noisy_hh-h])
                            noisy_bboxes.append([noisy_x, noisy_y, noisy_ww, noisy_hh])
                            category.append(index_third)
                    if index_judge == 1:
                        while(1):
                            x_background = np.random.randint(0,img_w-w)
                            y_background = np.random.randint(0,img_h-h)
                            ok_add = 1
                            for loop_judge in range(0, len(bboxes)):
                                if compute_iou(bboxes[loop_judge], [x_background, y_background, w, h]) >= 0.2:
                                    ok_add = 0
                                    break
                            if ok_add == 1:
                                ori_category_id.append(8)
                                is_cate_noise.append(1)
                                is_bbox_noise.append(0)
                                offset.append([0.0, 0.0, 0.0, 0.0])
                                noisy_bboxes.append([x_background, y_background, w, h])
                                category.append(index_forth)
                                break   
            # save noisy gt
            data_infos[idx]['ann']['bboxes'] = np.array(noisy_bboxes).astype(np.float32)
            data_infos[idx]['ann']['labels'] = np.array(category).astype(np.int32)
            data_infos[idx]['ann']['is_cate_noise'] = np.array(is_cate_noise).astype(np.int32)
            data_infos[idx]['ann']['is_bbox_noise'] = np.array(is_bbox_noise).astype(np.int32)
            data_infos[idx]['ann']['ori_category_id'] = np.array(ori_category_id).astype(np.int32)
            data_infos[idx]['ann']['offset'] = np.array(offset).astype(np.float32)
    print('done')
    return data_infos
       
def load_annotations(ann_file):
    return mmcv.load(ann_file)

def save_annotations(outputs, ann_file):
    return mmcv.dump(outputs, ann_file)

def preprocess_dataset(prefix='/none/xxx.json',save_path_train_temp='temp.json'):
    print('\npreprocess dataset ...')
    coco = AitodDataset()
    
    # load training data & save as pkl
    ann_file_train = prefix
    anno_train = coco.load_annotations(ann_file_train)
    
    save_path_train = save_path_train_temp
    save_annotations(anno_train, save_path_train)
    print('\npreprocess done')
    return save_path_train, coco.CLASSES

    
def mkdir(path):
	folder = os.path.exists(path)
	if not folder:                  
		os.makedirs(path)           

def parse_args():
    parser = argparse.ArgumentParser(description='Generate Noisy Datasets')
    parser.add_argument('--noise-level', nargs='+', help='the noise level [0.1, 0.2, 0.3, 0.4 ...]', required=True)
    '''
    noise_type:
    first:  missing labels
    second: inaccurate bounding boxes
    third:  class shifts
    forth:  extra labels
    st:     inaccurate bounding boxes + class shifts
    all:    missing labels + inaccurate bounding boxes + class shifts + extra labels
    '''
    parser.add_argument('--noise-types', nargs='+', 
                        help='the noise types [\'first\', \'second\', \'third\', \'forth\' ...]', 
                        required=True)
    parser.add_argument(
        '--clean-path', help='the clean datasets path')
    parser.add_argument(
        '--store-path', help='the noisy datasets path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # preprocess AI-TOD-v2.0 dataset
    args = parse_args()
    data_prefix = args.clean_path
    anno_prefix = args.store_path
    mkdir(anno_prefix)
    mkdir(os.path.join(anno_prefix, 'log'))
    save_path_train_temp = os.path.join(anno_prefix, 'log', 'temp.json')

    box_noise_level = args.noise_level
    mode = args.noise_types

    save_path_train, category_info = preprocess_dataset(data_prefix, save_path_train_temp)    

    for m in range(0, len(box_noise_level)):
        for n in range(0, len(mode)):
            seed = 45
            np.random.seed(seed)
            data_infos = load_annotations(save_path_train)
            noisy_data_infos = gen_noise_gt(data_infos, float(box_noise_level[m]), str(mode[n]), category_info)
            
            out_ann_file_1 = save_path_train + '_new.json'
            save_annotations(noisy_data_infos, out_ann_file_1)
            coco = mmcv.load(out_ann_file_1)
            
            coconew1 = dict()
            category_matrix = category_info
            coconew1['categories'] = []
            for cate in range(len(category_matrix)):
                temp123 = {}
                temp123["id"] = cate
                temp123["name"] = category_matrix[cate]
                temp123["supercategory"] = "mark"
                coconew1['categories'].append(temp123)
            '''
            info{
            "year" : int,                
            "version" : str,             
            "description" : str,         
            "contributor" : str,         
            "url" : str,                 
            "date_created" : datetime,   
            "noisiness"  :  str          
            "mode"  : str                
            }
            '''
            coconew1['info'] = {}
            coconew1['info']['year'] = 2023
            coconew1['info']['version'] = '1.0'
            coconew1['info']['description'] = "This is a noisy dataset based on AI-TOD-v2.0. For specific noise content, plz see 'noisiness'"
            coconew1['info']['contributor'] = 'Haoran Zhu'
            coconew1['info']['url'] = 'null'
            coconew1['info']['data_created'] = '2023.6.25'
            coconew1['info']['noisiness'] = box_noise_level[m]
            coconew1['info']['mode'] = mode[n]
            coconew1['annotations'] = []  
            coconew1['images'] = []   
            targetidssp1 = 0
            for j in range(0, len(coco)):
                Anns = coco[j]['ann']
                segmentations = []
                areas = []
                image_item = dict()
                image_item['file_name'] = coco[j]['file_name']
                image_item['height'] = coco[j]['height']
                image_item['width'] = coco[j]['width']
                image_item['id'] = coco[j]['id']
                coconew1['images'].append(image_item)
                for k in range(0, len(Anns['labels'])):
                    annotation_item = dict()
                    annotation_item['image_id'] = coco[j]['id']
                    annotation_item['bbox'] = [round(Anns['bboxes'][k][0], 1), round(Anns['bboxes'][k][1], 1), round(Anns['bboxes'][k][2], 1), round(Anns['bboxes'][k][3], 1)]
                    annotation_item['area'] = round(Anns['bboxes'][k][2] * Anns['bboxes'][k][3], 1)
                    annotation_item['segmentation'] = [[Anns['bboxes'][k][0], Anns['bboxes'][k][1], Anns['bboxes'][k][0] + Anns['bboxes'][k][2], Anns['bboxes'][k][1] + Anns['bboxes'][k][3]]]
                    annotation_item['category_id'] = Anns['labels'][k]
                    annotation_item['iscrowd'] = 0
                    annotation_item['ignore'] = 0
                    annotation_item['is_cate_noise'] = Anns['is_cate_noise'][k]
                    annotation_item['is_bbox_noise'] = Anns['is_bbox_noise'][k]
                    annotation_item['ori_category_id'] = Anns['ori_category_id'][k]
                    annotation_item['offset'] = [round(Anns['offset'][k][0], 1), round(Anns['offset'][k][1], 1), round(Anns['offset'][k][2], 1), round(Anns['offset'][k][3], 1)]
                    annotation_item['id'] = targetidssp1 
                    targetidssp1 = targetidssp1 + 1
                    coconew1['annotations'].append(annotation_item)
            out_ann_file = os.path.join(anno_prefix, 'aitodv2_noise_r{:.1f}_{}_train.json'.format(float(box_noise_level[m]), mode[n]))
            print("There are ", targetidssp1, ' labels')
            print('save to {}'.format(out_ann_file))
            json.dump(coconew1, open(out_ann_file, 'w'), indent = 4)


