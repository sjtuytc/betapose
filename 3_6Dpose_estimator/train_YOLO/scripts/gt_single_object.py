import os
import yaml
import random
from tqdm import tqdm
from shutil import copyfile
opj = os.path.join

NUM_SEQS = 15
CLASS_NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup',
               'driller', 'duck', 'eggbo', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
SIXD = '/home/data/sixd/hinterstoisser/test'
DARKNET = '/home/projects/detection/darknet/data_linemod_gt'
SEQS = ['%02d' % (i+1) for i in range(NUM_SEQS)]
GT_RATIO = 0.1
WIDTH = 640
HEIGHT = 480


if __name__ == '__main__':
    tbar = tqdm(SEQS, ascii=True, ncols=80)
    for idx, seq in enumerate(tbar):
        CLASS_NAME = CLASS_NAMES[int(seq)-1]

        SIXD_IMGS = opj(SIXD, seq, 'rgb')
        DARKNET_IMGS = opj(DARKNET, seq, 'images')
        SIXD_ANNO = opj(SIXD, seq, 'gt.yml')
        with open(SIXD_ANNO) as f:
            gt_info = yaml.load(f)

        IMG_PATHS = []
        if os.path.exists(DARKNET_IMGS) == False:
            os.makedirs(DARKNET_IMGS, exist_ok=True)

        for img in tqdm(os.listdir(SIXD_IMGS), ascii=True, ncols=80):
            if random.random() < GT_RATIO:
                # image
                src_img = opj(SIXD_IMGS, img)
                dst_img = opj(DARKNET_IMGS, img)
                copyfile(src_img, dst_img)
                IMG_PATHS.append(dst_img)
                
                # anno
                img_name = src_img.split('.')[-2].split('/')[-1]
                bbox = gt_info[int(img_name)][0]['obj_bb']
                bbox[0] = (bbox[0] + bbox[2] / 2) / WIDTH
                bbox[1] = (bbox[1] + bbox[3] / 2) / HEIGHT
                bbox[2] = bbox[2] / WIDTH
                bbox[3] = bbox[3] / HEIGHT
                dst_anno = dst_img.replace('.png', '.txt')
                with open(dst_anno, 'a') as f:
                    f.write("0 %f %f %f %f\n" % (bbox[0], bbox[1], bbox[2], bbox[3]))


        DARKNET_TRAIN_LIST = opj(DARKNET, seq, 'all.txt')
        DARKNET_DATA_CFG = opj(DARKNET, seq, CLASS_NAME + '.data')
        DARKNET_NAMES = opj(DARKNET, seq, CLASS_NAME + '.names')

        with open(DARKNET_TRAIN_LIST, 'a') as f:
            for img_path in IMG_PATHS:
                f.write(img_path + '\n')
        with open(DARKNET_DATA_CFG, 'a') as f:
            f.write('classes = 1\n')
            f.write('train = ' + opj('data_linemod_gt', seq, 'all.txt') + '\n')
            f.write('val = ' + opj('data_linemod_gt', seq, 'all.txt') + '\n')
            f.write('names = ' + opj('data_linemod_gt', seq, CLASS_NAME + '.names') + '\n')
            f.write('backup = ' + opj('backup_linemod_gt', seq))
        with open(DARKNET_NAMES, 'a') as f:
            f.write(CLASS_NAME)
            