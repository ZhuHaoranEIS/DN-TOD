import cv2
import os
import aitool
import mmcv

import aitool

if __name__ == '__main__':
    # ann_file = '/data/small/v1/coco/annotations/small_test_v1_1.0.json'
    # res_file = '/home/jwwangchn/Documents/Nutstore/100-Work/110-Projects/2019-Small/ICCV/results/cascade_rcnn_test_results_aitod_dotd_12epoch.bbox.json'
    # img_dir = '/data/small/v1/coco/test'
    # ann_file = '/data/small/v1/coco/annotations/small_val_v1_1.0.json'
    ann_file = '/home/xc/dataset/AI-TOD/AI-TOD/AI-TOD/annotations/small_test_v1_1.0.json'

    res_files = {
        #'iou': '/home/xc/mmdetection-aitod/mmdet-aitod/work_dirs/v001.01.01_aitod_faster_rcnn_r50_baseline/v001.01.01.bbox.json',
        'sota': '/home/xc/mmdetection-aitod/mmdet-aitod/work_dirs/sota_aitod_detectors_r50_rpn_rk3_test/sota_aitod_detectors_r50_rpn_rk3_test.bbox.json'}
    img_dir = '/home/xc/dataset/AI-TOD/AI-TOD/AI-TOD/test'

    sample_basenames = aitool.get_basename_list(
        '/home/xc/dataset/AI-TOD/AI-TOD/AI-TOD/test')

    #samples = ['105__300_600','1065__600_300','106__600_1800','1070__1200_1800','1070__1500_1500','1124__0_2618']
    samples =['74e0fc880','0000126_11844_d_0000130__600_0','322__1200_1200','1362__1200_600','01497','17259','P0023__1.0__1613___1800','P2668__1.0__600___600']
    score = 0.3
    final = dict()

    for method in ['sota']:
        # save_dir = f'/data/small/v1/results/CascadeRCNN/{method}/ship'
        save_dir = f'/home/xc/mmdetection-aitod/mmdet-aitod/show_result/AI-TOD-test-sota/{method}'
        res_file = res_files[method]
        coco_parser = aitool.COCOParser(ann_file)
        objects = coco_parser.objects
        img_name_with_id = coco_parser.img_name_with_id
        coco_result_parser = aitool.COCOJsonResultParser(res_file)
        for img_name in list(objects.keys())[::-1]:
            vehicle_count = 0
            # if img_name not in ['0000182_01220_d_0000039__0_0', '0000225_05003_d_0000016__600_0', '1127__1200_1200', 'P2245__1.0__469___0']:
            # continue
            if img_name not in sample_basenames:
                continue
            image_id = img_name_with_id[img_name]
            prediction = coco_result_parser(image_id)
            if len(prediction) == 0:
                continue
            ground_truth = coco_parser(img_name)

            img = cv2.imread(os.path.join(img_dir, img_name + '.png'))

            gt_bboxes, pred_bboxes = [], []
            for _ in ground_truth:
                gt_bboxes.append(_['bbox'])

            for _ in prediction:
                if _['score'] < score:
                    continue
                if _['category_id'] > 0:
                    vehicle_count += 1
                pred_bboxes.append(_['bbox'])


            gt_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in gt_bboxes])
            pred_bboxes = aitool.drop_invalid_bboxes([aitool.xywh2xyxy(_) for _ in pred_bboxes])

            if len(gt_bboxes) == 0:
                continue

            img = aitool.draw_confusion_matrix(img, gt_bboxes, pred_bboxes, with_gt_TP=False, line_width=1)

            if isinstance(img, list):
                continue

            output_file = os.path.join(save_dir, img_name + '.png')
            cv2.imwrite(output_file,img)
            #aitool.show_image(img, output_file=output_file, wait_time=10)