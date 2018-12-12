import os
import yaml
import random
from tqdm import tqdm
from shutil import copyfile
opj = os.path.join

SIXD = '/home/data/sixd/hinterstoisser/test/02'
DARKNET = '/home/projects/detection/darknet/data_linemod_gt/all'
LIST = '/home/projects/pose/yolopose/multi_obj_pose_estimation/cfg/train_occlusion.txt'
GT_RATIO = 0.1
WIDTH = 640
HEIGHT = 480

if __name__ == '__main__':
    SIXD_IMGS = opj(SIXD, 'rgb')
    DARKNET_IMGS = opj(DARKNET, 'images')
    SIXD_ANNO = opj(SIXD, 'gt.yml')
    with open(SIXD_ANNO) as f:
        gt_info = yaml.load(f)

    IMG_PATHS = []
    if os.path.exists(DARKNET_IMGS) == False:
        os.makedirs(DARKNET_IMGS, exist_ok=True)

    with open(LIST, 'r') as f:
        content = f.readlines()
    content = [x.strip().split('.jpg')[0].split('/')[-1][2:] for x in content]

    for imgname in tqdm(content, ascii=True):
        img = imgname + '.png'
        # image
        src_img = opj(SIXD_IMGS, img)
        dst_img = opj(DARKNET_IMGS, img)
        copyfile(src_img, dst_img)
        IMG_PATHS.append(dst_img)

        # anno
        dst_anno = dst_img.replace('.png', '.txt')
        annos = []
        img_name = src_img.split('.')[-2].split('/')[-1]
        gts = gt_info[int(img_name)]
        for gt in gts:
            obj_id = gt['obj_id'] - 1
            bbox = gt['obj_bb']
            bbox[0] = (bbox[0] + bbox[2] / 2) / WIDTH
            bbox[1] = (bbox[1] + bbox[3] / 2) / HEIGHT
            bbox[2] = bbox[2] / WIDTH
            bbox[3] = bbox[3] / HEIGHT
            annos.append("%d %f %f %f %f\n" % (obj_id, bbox[0], bbox[1], bbox[2], bbox[3]))
        with open(dst_anno, 'a') as f:
            for anno in annos:
                f.write(anno)

    DARKNET_TRAIN_LIST = opj(DARKNET, 'all.txt')
    DARKNET_DATA_CFG = opj(DARKNET, 'all.data')
    DARKNET_NAMES = opj(DARKNET, 'all.names')

    with open(DARKNET_TRAIN_LIST, 'a') as f:
        for img_path in IMG_PATHS:
            f.write(img_path + '\n')
    with open(DARKNET_DATA_CFG, 'a') as f:
        f.write('classes = 15\n')
        f.write('train = ' + opj('data_linemod_gt/all', 'all.txt') + '\n')
        f.write('val = ' + opj('data_linemod_gt/all', 'all.txt') + '\n')
        f.write('names = ' + opj('data_linemod_gt/all', 'all.names') + '\n')
        f.write('backup = ' + opj('backup_linemod_gt/all'))

