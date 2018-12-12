import argparse
import torch

parser = argparse.ArgumentParser(description='Keypoint Annotation')

"----------------------------- General options -----------------------------"
parser.add_argument('--expID', default='default', type=str,
                    help='Experiment ID')
parser.add_argument('--obj_id', default=1, type=int,
                    help='Object ID to annotate.')
parser.add_argument('--total_kp_number', default=50, type=int,
                    help='Total number of keypoints to annotate.')
parser.add_argument('--train_split', default=180, type=int,
                    help='Select this number to train and others to test.')
parser.add_argument('--output_base', default = "/media/data_2/COCO_SIXD/supplementary/", type=str,
                    help='Output base address.')
parser.add_argument('--sixd_base', default = "/media/data_2/SIXD/hinterstoisser", type=str,
                    help='Base of LineMod dataset, including models and designated keypoints dataset.')
opt = parser.parse_args()