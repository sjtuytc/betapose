from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import cv2
from utils.model import *
from utils.metrics import *
import os
import sys
# from IPython import embed # This tool is for debugging
import yaml
from utils.utils import *
from matplotlib import pyplot as plt
import math
from PIL import Image
import glob
import random
from utils.sixd import *
import h5py
from tqdm import tqdm
from shutil import copyfile, rmtree
from opt import opt
args = opt

'''
	Global Parameters assignment*****************************************************
'''
OBJECT_CHOSEN = args.obj_id # the object we care 1 4 6 15
TOTAL_KP_NUMBER = args.total_kp_number # the final number of kp in one model
NUM_SELECTED = args.train_split # How many images to choose for training
# Methods include random, sifts, cluster and corners
# output base
kp_dataset_base = args.output_base +'{:02d}'.format(int(OBJECT_CHOSEN))
sixd_base = args.sixd_base # for model loading

''' 
	Load model and KP model*******************************************************
'''
class Benchmark:
	def __init__(self):
		self.cam = np.identity(3)
		self.models = {}
		self.kpmodels = {}

def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f)
        return content

def load_bench(base_path):
	print("Loading models and KP models...")
	bench = Benchmark()
	bench.scale_to_meters = 0.001 # Unit in model is mm
	# You need to give camera info manually. Here is the camera info in Linemod dataset.
	bench.cam = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

	#collect model info
	model_info = load_yaml(os.path.join(base_path, 'models', 'models_info.yml'))
	for key, val in model_info.items():
		name = '{:02d}'.format(int(key))
		bench.models[name] = Model3D()
		bench.models[name].diameter = val['diameter']

	# loading models, Linemod has 15 seqs, we use 13(except 3 and 7)
	for ID in range(1,16):
		name = 'obj_{:02d}'.format(ID)
		bench.models['{:02d}'.format(ID)].load(os.path.join(base_path, 'models/' + name + '.ply'), scale=bench.scale_to_meters)
	print("Loading models finished!")

	# loading and refine kp models
	ID = OBJECT_CHOSEN
	name = 'obj_{:02d}'.format(ID)
	bench.kpmodels['{:02d}'.format(ID)] = Model3D()
	bench.kpmodels['{:02d}'.format(ID)].load(os.path.join(base_path, 'kpmodels/' + name + '.ply'), scale=bench.scale_to_meters)
	bench.kpmodels['{:02d}'.format(ID)].refine(TOTAL_KP_NUMBER, save=True, save_path = str(ID)+'.ply') # delete too close points
	# embed()
	print("Load and refine KP models finished!")
	return bench

''' 
	Load ground truth *******************************************************
'''
class sinobj:
	name = 0
	gt_bbox = [] # [xmin, ymin, xmax, ymax], got from ground truth
	mask_bbox = [] # [xmin, ymin, xmax, ymax], got from posed model mask
	rot = []  # 1*9 array
	pose = np.eye(4) # trans unit: minidom
	all_3d = [] # collecting all 3d points
	kp_3d = [] # collecting kp 3d points
	kp_2d = [] # collecting kp 2d points in image coordinate system
	all_depth = np.empty([1, 1]) # depth img of 3d all points with pose applied
	kp_depth = np.empty([1, 1]) # depth img of 3d key points with pose applied
	real_all_depth = np.empty([1, 1]) 
	# depth img of 3d all points which are in original positions
	rgb = np.empty([1, 1]) # sinobj img cropped from original image
	kp_label = np.empty([1, 1]) # each pixel denotes a label for a key point
	wrapped_kp = [] # wrap kp for final output
	kpcounter = 0
	real_kp_label = []
	predict_pose = np.eye(4)

	def __init__(self, n, gt_bbox, pose):
		self.name = n
		self.gt_bbox = gt_bbox
		self.inprot = 0
		self.pose = pose
		self.newpose = pose

	def apply_pose(self, model_points, kpmodel_points = []):
		''' Apply pose to single model vertices and kpmodel vertices, and
		store them in self.all_3d and self.kp_3d.
			Args:
			model_points: original model vertices(x,y,z), N*3 list
			kpmodel_points: original kp model vertices(x,y,z), K*3 list
		'''		
		self.all_3d = trans_vertices_by_pose(model_points, self.newpose)
		if len(kpmodel_points) < 1: return
		self.kp_3d = trans_vertices_by_pose(kpmodel_points, self.newpose)
		# output_pointcloud(trans_vertices, "test.ply")

	def project_all(self, cam):
		''' Project all_3d to pixel space. This function is designed for
		testing project_kp and, more importantly, for the local-top algorithm.
		'''
		# Initialize mask with all zero
		all_depth = np.zeros((480, 640))
		# Project all_3d vertices
		for point in self.all_3d:
			px = point[0]
			py = point[1]
			pz = point[2]
			# Project 3D point to pixel space
			x = px * cam[0, 0] / pz + cam[0, 2]
			y = py * cam[1, 1] / pz + cam[1, 2]
			z = pz * 1000 # Using unit mm
			# Generate depth img
			if (int(y) > 0) and (int(y) < 480) and (int(x) > 0) and (int(x) < 640): 
				if all_depth[int(y)][int(x)] == 0 or z < all_depth[int(y)][int(x)]:
					# we only choose the top vertices
					all_depth[int(y)][int(x)] = z
		self.all_depth = all_depth # will be used in local-top
		# The following codes in this function are just for visualization
		# # Apply in-plane rotation angle
		# all_depth = rotate(all_depth, self.inprot)
		mask = generate_mask_img(all_depth)
		# Crop sin-obj depth image using bounding box from depth mask
		xmin, xmax, ymin, ymax = get_bbox_from_mask(mask)
		self.mask_bbox = [xmin, xmax, ymin, ymax]


	def project_kp(self, cam):
		''' Project kp_3d to pixel space, while we also want to maintain the label
		of every key point.
			More explanations:
			kp_raw_depth is got by directly projecting kp_3d.
			kp_depth is got by applying local-top algorithm, namely removing points
		that can't be seen in jpg.
			kp_label is the final annotation matrix we got, it's an array with shape
		(GLOO_WIDTH, GLOO_HEIGHT), each number indicating the label of kp, 0 for none.
		'''
		kp_raw_depth = np.zeros((480, 640))
		label_matrix = np.zeros((480, 640))
		self.kp_2d = []
		xmin = self.mask_bbox[0]
		xmax = self.mask_bbox[1]
		ymin = self.mask_bbox[2]
		ymax = self.mask_bbox[3]

		# Project kp_3d vertices
		for label, point in enumerate(self.kp_3d):
			px = point[0]
			py = point[1]
			pz = point[2]
			# Project 3D point to pixel space
			x = px * cam[0, 0] / pz + cam[0, 2]
			y = py * cam[1, 1] / pz + cam[1, 2]
			z = pz * 1000
			# Calculate corresponding 2D points' location
			x_ratio = (x - xmin) / (xmax - xmin)
			y_ratio = (y - ymin) / (ymax - ymin)
			# Storing in kp_2d
			self.kp_2d.append([x_ratio, y_ratio])

	def crop_img(self, whole_img):
		# don't use this
		self.jitter_bbox = jitter_bbox(self.gt_bbox, JITTER)
		cropped_img = whole_img.crop((self.jitter_bbox[0], self.jitter_bbox[1], self.jitter_bbox[2], self.jitter_bbox[3]))
		resized_cropped_img = cropped_img.resize((GLOO_WIDTH, GLOO_HEIGHT), Image.LANCZOS)
		self.rgb = np.array(resized_cropped_img)
		# visualize_img(self.rgb, filename = ("rgb_" + str(output_counter) + ".jpg"))

	def speak(self):
		print("name is ", self.name)
		print("gt_bbox is ", self.gt_bbox)
		print("rot is ", self.rot)
		print("pose is", self.pose)

	def wrap_kp(self):
		'''Wrap kp for output. The final form of kp will be a list [kp1, kp2, ..., kpN].
	Each kp will be a tuple (kpx, kpy, if_occluded). If kpi is not seen in self.rgb, it
	will be (-1, -1, -1).
		'''
		global LOG_FOUT
		wrapped_kp = []
		kp_label = self.kp_label[:, :, 0]
		# print_all_nonzero(kp_label)
		'''
		 The image coordinate system is cross-wise, which is different 
		from numpy common setting. So we will exchange x and y.
		'''
		kpcounter = 0
		for kp_index in range(1, TOTAL_KP_NUMBER + 1): 
			# Label range: 1, 2, ..., TOTAL_KP_NUMBER
			if len(np.argwhere(kp_label == kp_index)) != 0: # not occluded
				tmp_xy = np.mean(np.argwhere(kp_label == kp_index), 0)
				tmp_xy = tmp_xy[: : -1] # exchange x and y
				tmp_xy = tmp_xy.tolist()
				# tmp_xy.append(1)
				kpcounter += 1
				wrapped_kp.append(tmp_xy)
			elif len(np.argwhere(kp_label == kp_index + 66)) != 0 : # occluded
				tmp_xy = np.mean(np.argwhere(kp_label == kp_index + 66), 0)
				tmp_xy = tmp_xy[: : -1] # exchange x and y
				tmp_xy = tmp_xy.tolist()
				tmp_xy.append(0)
				wrapped_kp.append(tmp_xy)
			else: #not appear in current img at all
				wrapped_kp.append([-1, -1])
		self.wrapped_kp = wrapped_kp
		self.kpcounter = kpcounter
		# log_string(LOG_FOUT, str(kpcounter))

	def output(self, whole_img):
		''' Output rgb img and corresponding kp label file.
		'''
		global output_counter
		# # We don't need to save img because we just use linemod's original img.
		# img_file = os.path.join(output_img_dir, '%i.jpg' % (output_counter))     
		# img = np.array(whole_img) # In fact it's saved as B, G, R due to matplotlib
		# img = img[:, :, ::-1] # Convert B, G, R to R, G, B
		# cv2.imwrite(img_file, img, [int( cv2.IMWRITE_JPEG_QUALITY), 100])

		# #Just for testing...
		# kp_depth_file = os.path.join(output_kp_depth_dir, '%i.jpg' % (output_counter))
		# kp_depth = self.kp_depth
		# cv2.imwrite(kp_depth_file, kp_depth, [int( cv2.IMWRITE_JPEG_QUALITY), 100])
		
		# Saving bbox
		xmin = self.gt_bbox[0]
		ymin = self.gt_bbox[1]
		xmax = self.gt_bbox[2]
		ymax = self.gt_bbox[3]
		bbox_file = os.path.join(output_bbox_dir, '%i.npy' %(output_counter))
		bbox = self.gt_bbox
		np.save(bbox_file, bbox)

		# Change ratio into image coordinate system
		kp_2d = self.kp_2d
		kp_img_xy = [] # final output kp matrix
		for kp in kp_2d:
			x_ratio = kp[0]
			y_ratio = kp[1]
			x = x_ratio * (xmax - xmin) + xmin
			y = y_ratio * (ymax - ymin) + ymin
			kp_img_xy.append([x, y]) 
		# embed()
		kp_label_file = os.path.join(output_kp_label_dir, '%i.npy' % (output_counter))
		kp_img_xy = np.array(kp_img_xy)
		np.save(kp_label_file, kp_img_xy)
		output_counter += 1

	def pnp(self, cam_K, kp_model_vertices, correct_metric, model_points):
		# print(self.newpose)
		# points_2D = self.wrapped_kp
		points_3D = np.copy(kp_model_vertices)
		points_2D = np.array(self.kp_2d)
		# print "In add..."	
		del_idx = []
		for idx, ele in enumerate(points_2D):
			if (len(points_2D) - len(del_idx)) > KP_PER_IMG:
				del_idx.append(idx)
		points_2D = np.delete(points_2D, del_idx, 0)
		points_3D = np.delete(points_3D, del_idx, 0)
		# Add disturber
		for idx, ele in enumerate(points_2D):
			ele[0] = ele[0] + DISTURB_PIXEL * random.uniform(-1, 1)
			ele[1] = ele[1] + DISTURB_PIXEL * random.uniform(-1, 1)
		# Run PNP algorithm
		R, t = pnp(points_3D, points_2D, cam_K)

		# Form poses
		self.predict_pose = np.eye(4)
		self.predict_pose[:3, :3] = R
		self.predict_pose[:3, 3] = t[:,0]
		add = add_err(self.newpose, self.predict_pose, model_points)
		add *=1000 # changing unit
		# print (add)
		# embed()
		return add<correct_metric

''' 
	Annotate keypoints on all the files. *******************************************************
'''
def gene_all_files(frames, cam_K, models, kp_models):
	for idx,f in enumerate(frames):
		#f.gt is a tuple consists of all objs
		if idx % 100 == 0 :
			print(idx, "has finished! ")
		for obj_perf in f.gt:
			name = obj_perf[0]
			# embed()
			if name != OBJECT_CHOSEN: continue
			pose = obj_perf[1]
			bbox = obj_perf[2]
			bbox[2]+=bbox[0]
			bbox[3]+=bbox[1]
			
			frame_obj = sinobj(name, bbox, pose)
			frame_obj.apply_pose(models['{:02d}'.format(int(name))].vertices, \
					kpmodels['{:02d}'.format(int(name))].vertices)
			frame_obj.project_all(cam_K)
			frame_obj.project_kp(cam_K)
			# embed()
			whole_img = Image.fromarray(np.uint8(np.array(f.color)))
			frame_obj.output(whole_img)


if __name__ == '__main__':
	global input_img_dir
	global input_annotation_dir

	# log writer
	global LOG_FOUT
	LOG_FOUT = open('kp_dataset_log.txt', 'w')

	global output_counter
	global output_img_dir
	global output_kp_label_dir
	global output_bbox_dir

	output_counter = 0

	output_img_dir = os.path.join(kp_dataset_base, 'rgb')
	output_kp_label_dir = os.path.join(kp_dataset_base, 'kp_label')
	# output_kp_depth_dir = os.path.join(kp_dataset_base, 'kp_depth')
	output_bbox_dir = os.path.join(kp_dataset_base, 'bbox')

	if not os.path.exists(kp_dataset_base):
		os.makedirs(kp_dataset_base)
	if not os.path.exists(output_img_dir):
		os.makedirs(output_img_dir)
	if not os.path.exists(output_kp_label_dir):
		os.makedirs(output_kp_label_dir)	
	# if not os.path.exists(output_kp_depth_dir):
	# 	os.makedirs(output_kp_depth_dir)	
	if not os.path.exists(output_bbox_dir):
		os.makedirs(output_bbox_dir)	

	print ("Running keypoint dataset generator ...")
	bench = load_bench(sixd_base)
	cam_K = bench.cam
	models = bench.models
	kpmodels = bench.kpmodels
	print("Now generating", OBJECT_CHOSEN)
	print("Saving in", kp_dataset_base)
	print("Loading all sixd frames of current seq.")
	# new added validation dataset
	bench = load_sixd(sixd_base, seq=OBJECT_CHOSEN)	#Modified, take care!
	# bench = load_sixd(sixd_base, seq=2)
	print("Loading finished!")
	gene_all_files(bench.frames, cam_K, models, kpmodels)

	print("Now spliting images into training and eval.")
	LINEMOD_ROOT = os.path.join(sixd_base, 'test')
	opj = os.path.join
	all_imgs = opj(LINEMOD_ROOT, '{:02d}'.format(int(OBJECT_CHOSEN)), 'rgb')
	num_all_imgs = len(os.listdir(all_imgs))
	train_imgs = opj(kp_dataset_base, 'train')  # 186
	eval_imgs = opj(kp_dataset_base, 'eval')  # the rest
	if not os.path.exists(opj(kp_dataset_base)):
		print("target folder does not exist")
	if os.path.exists(train_imgs) or os.path.exists(eval_imgs):
		print("train val has been splited")
	if not os.path.exists(train_imgs):
		os.makedirs(train_imgs)
	if not os.path.exists(eval_imgs):
		os.makedirs(eval_imgs)	
	count = 0
	selected_ids = list(np.random.choice(num_all_imgs, NUM_SELECTED, replace=False))
	tbar = tqdm(os.listdir(all_imgs), ascii=True, ncols=80)
	for idx, img in enumerate(tbar):
		img_path = opj(all_imgs, img)
		img_idx = int(img_path.split('/')[-1].split('.')[0])  # 0001
		target_path = None
		if idx in selected_ids:
			count += 1
			target_path = opj(train_imgs, '%012d.png' % img_idx)
		else:
			target_path = opj(eval_imgs, '%012d.png' % img_idx)
		copyfile(img_path, target_path)
	assert count == NUM_SELECTED, "%d, %d" % (count, NUM_SELECTED)

	print("Now generating h5 files for annotations.")
	# h5 files are for training KPD
	for t in ('train', 'eval'):
		ANNO_PATH = opj(kp_dataset_base, 'annot_%s.h5' % t)
		img_names = []
		kp_labels = []
		bboxes = []
		for img in tqdm(os.listdir(opj(kp_dataset_base, t)), ascii=True):
			img_idx = str(int(img.split('.')[0]))
			bbox_path = opj(kp_dataset_base, 'bbox', img_idx + '.npy')
			kp_label_path = opj(kp_dataset_base, 'kp_label', img_idx + '.npy')

			name_chars = []
			for char in img:
				name_chars.append(ord(char))
			name_array = np.asarray(name_chars)

			img_names.append(name_array)
			if not os.path.exists(bbox_path): continue
			bboxes.append(np.load(bbox_path))
			kp_labels.append(np.load(kp_label_path))

		imgnames = np.vstack(img_names)
		bndboxes = np.vstack(bboxes).reshape(-1,1,4)
		parts = np.vstack(kp_labels).reshape(-1,TOTAL_KP_NUMBER,2)

		if os.path.exists(ANNO_PATH):
			os.remove(ANNO_PATH)
		with h5py.File(ANNO_PATH, "w") as f:
			f.create_dataset("bndbox", data=bndboxes)
			f.create_dataset("imgname", data=imgnames)
			f.create_dataset("part", data=parts)

	print("All done!")