import os
from IPython import embed
my_path = "9_KP_11_1"
my_flag = 2
if (my_flag == 0):
	img_base = "/media/data_2/COCO_SIXD/" + my_path + "/01/rgb"
if my_flag == 1:
	img_base = "/media/data_2/COCO_SIXD/" + my_path + "/01/bbox"
if my_flag == 2:
	img_base = "/media/data_2/COCO_SIXD/" + my_path + "/01/kp_label"
img_files = os.listdir(img_base)
for idx, temp in enumerate(img_files):
	# print temp
	num = temp.rfind('.')
	if num != -1:	
		new_name = int(temp[: num])
		# embed()
		new_name = "%012d" % new_name
		if my_flag == 0:
			new_name = new_name + ".jpg"
		else:
			new_name = new_name + ".npy"
		if idx%1000 == 0: 
			print(idx, "finished!")
		os.rename(os.path.join(img_base, temp),os.path.join(img_base, new_name))
	# embed()
	# break
