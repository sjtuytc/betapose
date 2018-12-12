# load sixd groundtruth from xml file
from xml.dom.minidom import parse
import xml.dom.minidom
import numpy as np
import cv2
from model import Model3D
import os
import sys
from IPython import embed #debugging
import yaml
from utils import *
from matplotlib import pyplot as plt
import math
from PIL import Image

# Global Parameters *****************************************************
# The cropped img will be GLOO_WIDTH * GLOO_HEIGHT
GLOO_WIDTH = 100
GLOO_HEIGHT = 100
# Load model info *******************************************************
class Benchmark:
	def __init__(self):
		self.cam = np.identity(3)
		self.models = {}

def load_yaml(path):
    with open(path, 'r') as f:
        content = yaml.load(f)
        return content

def load_bench(base_path):
	
	bench = Benchmark()
	bench.scale_to_meters = 1 # don't care yet
	# You need to give camera info manually. Here is the camera info in Linemod dataset.
	bench.cam = np.array([[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]])

	#collect model info
	model_info = load_yaml(os.path.join(base_path, 'models', 'models_info.yml'))
	for key, val in model_info.items():
		name = '{:02d}'.format(int(key))
		bench.models[name] = Model3D()
		bench.models[name].diameter = val['diameter']

	# loading models, Linemod has 16 seqs
	for ID in range(1,16):
		name = 'obj_{:02d}'.format(ID)
		bench.models['{:02d}'.format(ID)].load(os.path.join(base_path, 'models/' + name + '.ply'), scale=bench.scale_to_meters)
	return bench

# Load ground truth *****************************************************************

class sinobj:
	name = 0
	bbox = [] # [xmin, ymin, xmax, ymax]
	rot = []  # 1*9 array
	pose = np.eye(4) # trans unit: minidom
	all_3d = [] # collecting all 3d points
	all_depth = np.empty([2, 2]) # depth img of 3d points applied pose
	rgb = np.empty([2, 2]) #sinobj img cropped from original image

	def __init__(self, n, inprot, bbox, rot):
		self.name = n
		self.bbox = bbox
		self.inprot = inprot
		self.rot = rot
		self.pose = np.eye(4)
		self.pose[:3, 3] = [0,0,500] #Linemod uses constant cam_t
		self.pose[0, 0] = rot[0]
		self.pose[0, 1] = rot[1]
		self.pose[0, 2] = rot[2]
		self.pose[1, 0] = rot[3]
		self.pose[1, 1] = rot[4]
		self.pose[1, 2] = rot[5]
		self.pose[2, 0] = rot[6]
		self.pose[2, 1] = rot[7]
		self.pose[2, 2] = rot[8]

	def apply_pose(self, sin_model_vertices):
		''' Apply pose to single model vertices and get all_3d.
			Args:
			sin_model_vertices: original vertices(x,y,z), N*3 list
		'''		
		# make points homogeneous, copy them to maintain the originals
		sin_model_vertices = np.array(sin_model_vertices)
		ext_model_points = np.ones((4,sin_model_vertices.shape[0]))
		ext_model_points[:3,:] = np.copy(sin_model_vertices.T)
		self.all_3d = np.dot(np.array(self.pose), ext_model_points)
		# Transfer points to original form
		self.all_3d = self.all_3d.T
		self.all_3d = np.copy(self.all_3d[:, :3])
		output_pointcloud(self.all_3d, "test.ply")
		return

	def project(self, cam):
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
			z = pz
			# Generate depth img
			if (int(y) > 0) and (int(y) < 480) and (int(x) > 0) and (int(x) < 640): 
				all_depth[int(y)][int(x)] = z
		all_depth = rotate(all_depth, self.inprot)
		mask = generate_mask_img(all_depth)
		# Crop image using bounding box from the all_depth
		xmin, xmax, ymin, ymax = get_bbox_from_mask(mask)
		# print(xmin, xmax, ymin, ymax)
		ori_all_depth = Image.fromarray(np.uint8(mask))
		cropped_all_depth = ori_all_depth.crop((xmin, ymin, xmax, ymax))
		# Resize the all_depth image due to the scale augmentation
		resized_cropped_all_depth = cropped_all_depth.resize((GLOO_WIDTH, GLOO_HEIGHT), Image.LANCZOS)
		final_all_depth = np.array(resized_cropped_all_depth)
		# visualize_img(final_all_depth)
		self.all_depth = final_all_depth
		return final_all_depth

	def crop_img(self, whole_img):
		cropped_img = whole_img.crop((self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]))
		resized_cropped_img = cropped_img.resize((GLOO_WIDTH, GLOO_HEIGHT), Image.LANCZOS)
		self.rgb = np.array(resized_cropped_img)
		# visualize_img(self.rgb)

	def speak(self):
		print("name is ", self.name)
		print("bbox is ", self.bbox)
		print("rot is ", self.rot)
		print("pose is", self.pose)

# Loading all the things***************************************************
def load_per_file(cam_K, models):

	whole_img = Image.open("2518.jpg")
	DOMTree = xml.dom.minidom.parse("2518.xml")
	collection = DOMTree.documentElement
	allobjs = [] # store all obj info in per file
	# collect all object entry
	objects = collection.getElementsByTagName("object") 
	for obj in objects:
		name_ele = obj.getElementsByTagName('name')[0]
		name = int(name_ele.childNodes[0].data)
		inprot_ele = obj.getElementsByTagName('inprot')[0]
		inprot = int(inprot_ele.childNodes[0].data)		
		bbox_ele = obj.getElementsByTagName('bndbox')[0]
		# get subelement of 'bndbox'
		xmin_ele = bbox_ele.getElementsByTagName('xmin')[0]
		xmax_ele = bbox_ele.getElementsByTagName('xmax')[0]
		ymin_ele = bbox_ele.getElementsByTagName('ymin')[0]
		ymax_ele = bbox_ele.getElementsByTagName('ymax')[0]
		xmin = int(xmin_ele.childNodes[0].data)
		xmax = int(xmax_ele.childNodes[0].data)
		ymin = int(ymin_ele.childNodes[0].data)
		ymax = int(ymax_ele.childNodes[0].data)
		bbox = [xmin, ymin, xmax, ymax]
		pose_ele = obj.getElementsByTagName('pose')[0]
		pose = (pose_ele.childNodes[0].data).split(',')
		pose = [float(i) for i in pose]
		# note here pose only means rotation
		allobj = sinobj(name, inprot, bbox, pose) 
		allobj.apply_pose(models['{:02d}'.format(int(name))].vertices)
		allobj.project(cam_K)
		allobj.crop_img(whole_img)
		allobjs.append(allobj)


if __name__ == '__main__':
	print ("Testing load_gt ...")
	sixd_base = "/media/data_2/SIXD/hinterstoisser/"
	nr_frames = 0
	sequence = 4
	print("Loading models...")
	bench = load_bench(sixd_base)
	cam_K = bench.cam
	models = bench.models

	# # test ground truth pose using obj1, 0000.png
	# testobj = sinobj(1,0,[],[0.09630630, 0.99404401, 0.05100790, 0.57332098, -0.01350810, -0.81922001, -0.81365103, 0.10814000, -0.57120699])
	# testobj.pose[:3,3] = [-105.35775150, -117.52119142, 1014.87701320]
	# testobj.apply_pose(models['{:02d}'.format(int(1))].vertices)
	# testobj.project(cam_K)
	# embed()

	load_per_file(cam_K, models)
	embed()