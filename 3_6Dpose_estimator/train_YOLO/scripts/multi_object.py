import os
import random
opj = os.path.join
from tqdm import tqdm
from shutil import copyfile
from xml.etree import ElementTree

NUM_SEQS = 15
seqs = ['%02d' % (i+1) for i in range(NUM_SEQS)]

CLASS_NAMES = ('ape', 'bvise', 'bowl', 'camera', 'can', 'cat', 'cup', 'driller', 'duck', 'eggbox', 'glue', 'holepuncher', 'iron', 'lamp', 'phone')
ROOT = '/media/data_2/COCO_SIXD/hinter'
ANNO_DIR = opj(ROOT, 'Annotations')
IMGS_DIR = opj(ROOT, 'JPEGImages')
DARKNET_ROOT = opj('/media/data_1/home/detection/darknet/data_linemod', 'all')
DARKNET_IMGS_DIR = opj(DARKNET_ROOT, 'images')
DARKNET_ANNO_DIR = opj(DARKNET_ROOT, 'images')
DARKNET_TRAIN_LIST = opj(DARKNET_ROOT, 'all.txt')
DARKNET_VAL_LIST = opj(DARKNET_ROOT, 'val.txt')
DARKNET_DATA_CFG = opj(DARKNET_ROOT, 'linemod.data')
DARKNET_NAMES = opj(DARKNET_ROOT, 'linemod.names')

os.makedirs(DARKNET_IMGS_DIR, exist_ok=True)
if os.path.exists(DARKNET_VAL_LIST):
    os.remove(DARKNET_VAL_LIST)
if os.path.exists(DARKNET_DATA_CFG):
    os.remove(DARKNET_DATA_CFG)
if os.path.exists(DARKNET_TRAIN_LIST):
    os.remove(DARKNET_TRAIN_LIST)
if os.path.exists(DARKNET_NAMES):
    os.remove(DARKNET_NAMES)

for anno in tqdm(os.listdir(ANNO_DIR), ascii=True):
    anno_path = opj(ANNO_DIR, anno)
    root = ElementTree.parse(anno_path).getroot()
    width = int(root.find('size').find('width').text)
    height = int(root.find('size').find('height').text)
    objects = root.findall('object')
    bboxes = []
    for o in objects:
        bndbox = o.find('bndbox')
        label = int(o.find('name').text) - 1
        x1 = int(bndbox[0].text)
        x2 = int(bndbox[1].text)
        y1 = int(bndbox[2].text)
        y2 = int(bndbox[3].text)
        xc = (x1 + x2) / 2
        yc = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        xc, yc, w, h = xc / width, yc / height, w / width, h / height
        bbox_str = str(label) + ' ' + str(xc) + ' ' + str(yc) + ' ' + str(w) + ' ' + str(h)
        bboxes.append(bbox_str + '\n')

    anno_path_new = anno_path.replace(ANNO_DIR, DARKNET_ANNO_DIR).replace('.xml', '.txt')
    if os.path.exists(anno_path_new):
        os.remove(anno_path_new)
    with open(anno_path_new, 'a') as f:
        for bbox in bboxes:
            f.write(bbox)
    img_path = anno_path.replace(ANNO_DIR, IMGS_DIR).replace('.xml', '.jpg')
    img_path_new = anno_path_new.replace('.txt', '.jpg')
    if not os.path.exists(img_path_new):
        copyfile(img_path, img_path_new)

    with open(DARKNET_TRAIN_LIST, 'a') as f:
        f.write(img_path_new + '\n')
    if random.random() < 0.1:
        with open(DARKNET_VAL_LIST, 'a') as f:
            f.write(img_path_new + '\n')

with open(DARKNET_DATA_CFG, 'a') as f:
    f.write('classes = %d\n' % len(CLASS_NAMES))
    f.write('train = data_linemod/all/all.txt\n')
    f.write('val = data_linemod/all/val.txt\n')
    f.write('names = data_linemod/all/linemod.names\n')
    f.write('backup = backup_linemod/all\n')

with open(DARKNET_NAMES, 'a') as f:
    for name in CLASS_NAMES:
        f.write(name + '\n')

print("\nDone!")
