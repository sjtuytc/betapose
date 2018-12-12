from __future__ import print_function
import mxnet as mx
import numpy as np
from timeit import default_timer as timer
from dataset.testdb import TestDB
from dataset.iterator import DetIter
from utils_recon.utils import *
from utils_icp.icp import icp
from utils_recon.metrics import add_err, rot_error, trans_error, iou
import random
from open3d import * # Using open3d icp algorithm
import copy
from utils_icp.global_icp_test import *

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def icp_refinement(sequence, src_points, des_points, dets_6d, image_dets, max_iterations = 30, tolerance=0.0001):
    '''
    Wrapper for icp algorithm.
    Input:
      src_points: det*X*3 list where det means number of dets in one img, X means number of points 
      (undetermined), 3 means XYZ cooridinates.
      des_points: det*vi*Y*3 list where det means number of dets in one img, vi means number of v,i
      pairs, Y means number of points (undetermined), 3 means XYZ cooridinates.
    Output:
      final_pools: a final pool of poses, det*vi*4*4 list where det means number of dets in one img, vi means number of v,i
      pairs, 4*4 means pose matrix.
      new_dets_6d: pose-refined dets_6d
      final_points: return finalpoints for visualization, det*vi*Y*3 list where det means number of dets 
      in one img, vi means number of v,i pairs, Y means number of points (undetermined), 3 means XYZ cooridinates.
    '''
    randsand = int(random.random() * 100000)
    threshold = 0.01
    new_dets_6d = dets_6d
    # print(np.array(src_points).shape)
    # print(np.array(des_points).shape)
    final_pose = []
    final_points = []
    for det, ele in enumerate(des_points):
        
        vi_pose_pool = []
        vi_group_points = []
        for vi, _ in enumerate(des_points[det]):
            current_pose = dets_6d[det][vi + 6]
            current_pose = np.array(current_pose)
            current_src_points = src_points[det][vi]
            current_des_points = des_points[det][vi]
            
            # make points homogeneous, copy them to maintain the originals
            ext_des_points = np.ones((4, np.array(current_des_points).shape[0]))
            ext_des_points[:3,:] = np.copy(np.array(current_des_points).T)
            # print(ext_des_points.shape)

            vi_points = ext_des_points
            # print("Handling current %d pose ......."%vi)
            # print(current_pose)

            # # ONLY NEEDED BY SIMPLE ICP
            # # Random sampling to make the dimensions of src_points and des_points correct for icp
            # if len(src_points[det][vi]) > len (des_points[det][vi]):
            #     current_src_points = random.sample(src_points[det][vi], len(des_points[det][vi]))
            #     # print("Changing dimension of src_points...") 
            # elif len(src_points[det][vi]) < len (des_points[det][vi]):
            #     current_des_points = random.sample(des_points[det][vi], len(src_points[det][vi]))
            #     # print("Changing dimension of des_points...") 
            
            # fix the error of lost of vi
            if current_des_points == []:
                print("error current_des_points")
                continue

            testcounter = 0
            # print("Current label:")
            # print(dets_6d[det][0])
            if (dets_6d[det][0]==sequence):# only refine current sequence for reducing computation
            # Running icp algorithm...
              
              # # #SIMPLE ICP
              # T, _, iterations, errordif = icp(np.array(current_des_points), np.array(current_src_points), init_pose=None, max_iterations = max_iterations, tolerance=tolerance) 
              # print("Current iterations and errordif are:", iterations, errordif)
                          
              # # Handling Error Frame
              # if T == [] :
              #   return [], [], []

              # USING OPEN3D
              output_pointcloud(current_src_points, "tmp/"+str(randsand) + str(sequence) + "tmp_current_src_points.ply") 
              output_pointcloud(current_des_points, "tmp/"+str(randsand) + str(sequence) + "tmp_current_des_points.ply")

              source = read_point_cloud("tmp/"+str(randsand) + str(sequence) + "tmp_current_des_points.ply")
              target = read_point_cloud("tmp/"+str(randsand) + str(sequence) + "tmp_current_src_points.ply")
              estimate_normals(source, search_param = KDTreeSearchParamHybrid(radius = 1, max_nn=200))#default:0.01, 30
              estimate_normals(target, search_param = KDTreeSearchParamHybrid(radius = 1, max_nn=200))
              threshold = 0.008
              # 0.02 works OK
              trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0, 0.0],
                                      [0.0, 0.0, 1.0, 0.0],
                                      [0.0, 0.0, 0.0, 1.0]])
              if_see_result = random.random()# We use this to reduce the number of display
              # if if_see_result > 0.9:
              #     draw_registration_result(source, target, trans_init)
              print("Initial alignment")
              evaluation = evaluate_registration(source, target,
                      threshold, trans_init)
              # print(evaluation)

              # print("Apply point-to-point ICP...")
              # reg = registration_icp(source, target, threshold, trans_init,
              #         TransformationEstimationPointToPoint())
              # print(reg)
              # print("Transformation is:")
              # print(reg.transformation)
              # print("")
              # draw_registration_result(source, target, reg.transformation)

              print("Apply point-to-plane ICP...")
              reg = registration_icp(source, target, threshold, trans_init,
                      TransformationEstimationPointToPlane())
              # print(reg)
              # print("Transformation is:")
              # print(reg.transformation)
              # print("")
              # if if_see_result > 0.9:
              #     draw_registration_result(source, target, reg.transformation)
              
              T = reg.transformation

              current_pose = np.dot(T, current_pose)
              # apply pose
              vi_points = np.dot(np.array(T), ext_des_points)

            # print("After refinement, pose is:")
            # print(current_pose)
            new_dets_6d[det][vi + 6] = current_pose
            vi_pose_pool.append(current_pose)

            # transfer points to original form
            vi_points = vi_points.T
            vi_points = np.copy(vi_points[:, :3])
            #print("vi points info: ", vi_points.shape)
            vi_points = vi_points.tolist()
            vi_group_points.append(vi_points)    
        final_pose.append(vi_pose_pool)
        final_points.append(vi_group_points)
    return final_pose, new_dets_6d, final_points
            

class Detector(object):
  """
  SSD detector which hold a detection network and wraps detection API

  Parameters:
  ----------
  symbol : mx.Symbol
      detection network Symbol
  model_prefix : str
      name prefix of trained model
  epoch : int
      load epoch of trained model
  data_shape : int
      input data resize shape
  mean_pixels : tuple of float
      (mean_r, mean_g, mean_b)
  batch_size : int
      run detection with batch size
  ctx : mx.ctx
      device to use, if None, use mx.cpu() as default context
  """

  def __init__(self, symbol, model_prefix, epoch, data_shape, mean_pixels,
               batch_size=1, ctx=None):
    self.ctx = ctx
    if self.ctx is None:
      self.ctx = mx.cpu()
    load_symbol, args, auxs = mx.model.load_checkpoint(model_prefix, epoch)
    if symbol is None:
      symbol = load_symbol
    self.mod = mx.mod.Module(symbol, label_names=None, context=ctx)
    self.data_shape = data_shape
    self.mod.bind(data_shapes=[('data', (batch_size, 3, data_shape, data_shape))])
    self.mod.set_params(args, auxs)
    self.data_shape = data_shape
    self.mean_pixels = mean_pixels

  def detect(self, det_iter, show_timer=False):
    """
    detect all images in iterator

    Parameters:
    ----------
    det_iter : DetIter
        iterator for all testing images
    show_timer : Boolean
        whether to print out detection exec time

    Returns:
    ----------
    list of detection results
    """
    num_images = det_iter._size
    result = []
    detections = []
    if not isinstance(det_iter, mx.io.PrefetchingIter):
      det_iter = mx.io.PrefetchingIter(det_iter)
    start = timer()
    for pred, _, _ in self.mod.iter_predict(det_iter):
      detections.append(pred[0].asnumpy())
    time_elapsed = timer() - start
    if show_timer:
      print("Detection time for {} images: {:.4f} sec".format(
          num_images, time_elapsed))
    for output in detections:
      for i in range(output.shape[0]):
        det = output[i, :, :]
        res = det[np.where(det[:, 0] >= 0)[0]]
        result.append(res)
    return result

  def im_detect(self, im_list, root_dir=None, extension=None, show_timer=False):
    """
    wrapper for detecting multiple images

    Parameters:
    ----------
    im_list : list of str
        image path or list of image paths
    root_dir : str
        directory of input images, optional if image path already
        has full directory information
    extension : str
        image extension, eg. ".jpg", optional

    Returns:
    ----------
    list of detection results in format [[det0, det1...]], det is in
    format np.array([id, score, xmin, ymin, xmax, ymax]...)
    Note that the result concludes one redundant dimension, for there is pnly one img.
    """
    test_db = TestDB(im_list, root_dir=root_dir, extension=extension)
    test_iter = DetIter(test_db, 1, self.data_shape, self.mean_pixels,
                        is_train=False)
    return self.detect(test_iter, show_timer)

  def visualize_detection(self, img, dets, classes=[], thresh=0.6):
    """
    visualize detections in one image

    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2, tpl_id]...])
        each row is one object
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    import matplotlib.pyplot as plt
    import random
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]
    colors = dict()
    for i in range(dets.shape[0]):
      cls_id = int(dets[i, 0])
      if cls_id >= 0:
        score = dets[i, 1]
        if score > thresh:
          if cls_id not in colors:
            colors[cls_id] = (random.random(), random.random(), random.random())
          xmin = int(dets[i, 2] * width)
          ymin = int(dets[i, 3] * height)
          xmax = int(dets[i, 4] * width)
          ymax = int(dets[i, 5] * height)
          rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                               ymax - ymin, fill=False,
                               edgecolor=colors[cls_id],
                               linewidth=3.5)
          plt.gca().add_patch(rect)
          class_name = str(cls_id)
          if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
          plt.gca().text(xmin, ymin - 2,
                         '{:s} {:.3f}'.format(class_name, score),
                         bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                         fontsize=12, color='white')
    plt.show()

  def save_detection(self, image, dets, filename, colors={}, classes=[], thresh=0.6):
    """
    visualize detections in one image

    Parameters:
    ----------
    img : numpy.array
        image, in bgr format
    dets : numpy.array
        ssd detections, numpy.array([[id, score, x1, y1, x2, y2]...])
        each row is one object
    filename : str
        path to save the result, include path and filename
    classes : tuple or list of str
        class names
    thresh : float
        score threshold
    """
    import matplotlib.pyplot as plt
    plt.ioff()
    plt.clf()
    img = image.copy()
    plt.imshow(img)
    height = img.shape[0]
    width = img.shape[1]

    for i in range(dets.shape[0]):
      cls_id = int(dets[i, 0])
      if cls_id >= 0:
        score = dets[i, 1]
        if score > thresh:
          if cls_id not in colors:
            print('Color not specified for class '+str(cls_id))
            colors[cls_id] = (random.random(), random.random(), random.random())
          xmin = int(dets[i, 2] * width)
          ymin = int(dets[i, 3] * height)
          xmax = int(dets[i, 4] * width)
          ymax = int(dets[i, 5] * height)
          rect = plt.Rectangle((xmin, ymin), xmax - xmin,
                               ymax - ymin, fill=False,
                               edgecolor=colors[cls_id],
                               linewidth=3.5)
          plt.gca().add_patch(rect)
          class_name = str(cls_id)
          if classes and len(classes) > cls_id:
            class_name = classes[cls_id]
          plt.gca().text(xmin, ymin - 2,
                         '{:s} {:.3f}'.format(class_name, score),
                         bbox=dict(facecolor=colors[cls_id], alpha=0.5),
                         fontsize=12, color='white')
    plt.savefig(filename)

  def detect_and_visualize(self, f, sequence, frame_counter, im_list, model_map, cam, color, depth,
                           root_dir=None, extension=None, img_size=(640, 480),
                           classes=[], thresh=0.1, show_timer=False, num_tpls=641, 
                           num_inplane=4, select_tpls=3, select_inplane=3, ifvisualize = True, refinement = False):
    """
    Wrapper for im_detect and visualize_detection

    Parameters:
    ----------
    frame_counter : indicate which frame is being handling
    im_list : list of str or str
        image path or list of image paths
    root_dir : str or None
        directory of input images, optional if image path already
        has full directory information
    extension : str or None
        image extension, eg. ".jpg", optional
    thresh : threshold to visualize
    ifvisualize: choose to output files to visualize(in demo_recon) or not(in evaluation). 
    """
    import cv2
    dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
    # print(dets)
    if not isinstance(im_list, list):
      im_list = [im_list]
    assert len(dets) == len(im_list)

    '''
    Note: des_points means points in model premultiplied by pose, src_points means points derived
      from depth img.
    The relation is pose * des_points = src_points.
    '''

    # Build 6D poses in pool and convert model to destination points
    dets_6d, des_points = build_6D_poses(dets, model_map, cam, img_size=img_size,
                             num_tpls=num_tpls, num_inplane=num_inplane,
                             select_tpls=select_tpls, select_inplane=select_inplane)
    
    # Initialize new_dets_6d with dets_6d
    new_dets_6d = dets_6d
    
    # Create mask using in create_src_pointcloud, only positive pixels are lifted
    mask = create_mask(des_points, cam, img_size = img_size)

    # Convert depth img to 3D pointcloud for pose refinement
    src_points = create_src_pointcloud(mask, color, depth, cam, model_map, img_size=img_size)

    # # Test src_points...
    # print("Shape of src_points:",np.array(src_points).shape)
    # for i, _ in enumerate (src_points[0]):
    #     print("size of element in src_points is: %d"%len(_))
    #     output_pointcloud(_, "src_points_"+str(frame_counter)+"["+str(i)+"].ply") #Output pointcloud for visualization
    
    # # Test unrefined des_points... 
    # print("Shape of des_points:",np.array(des_points).shape)
    # for i, _ in enumerate(des_points[0]):
    #     print("size of element in des_points is: %d"%len(_))
    #     output_pointcloud(_,  "des_points_"+str(frame_counter)+"["+str(i)+"].ply") #Output pointcloud for visualization

    if refinement==True:
      # Run icp to do pose refinement for each pose in pool
      print("Now doing ICP refinement ...")
      max_iterations = 80
      # 80, 1e-6
      tolerance=1e-8
      final_pose, new_dets_6d, final_points = icp_refinement(sequence, src_points, des_points, dets_6d, dets[0], max_iterations = max_iterations, tolerance=tolerance)
      if final_pose == []:
        return []
      print("ICP refinement finished!")

    # print("length of dets_6d now : ", len(dets_6d))
    # print("length of new_dets_6d:",len(new_dets_6d))
    # print("length of dets in new_dets_6d:", len(new_dets_6d[0]))

    # # Test final_points...
    # print("Shape of final_points:", np.array(final_points).shape)
    # for i, _ in enumerate(final_points[0]): # Note: final_points[0] means the first det
    #     print("size of element in final_points is: %d"%len(_))
    #     output_pointcloud(_, "test_final_"+str(frame_counter)+"_0_["+str(i)+"].ply") #Output pointcloud for visualization

    adds = []
    trans_errors_norm = []
    trans_errors_single = []
    rot_errors = []

    final = verify_6D_poses(new_dets_6d, model_map, cam, color)

    if ifvisualize:
      # Detection Result Visualization and Evaluation
      for k, det in enumerate(dets):
          img = cv2.imread(im_list[k])
          # cv2.namedWindow("Output Show")
          # img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
          # Pick for each detection the best pose from the 6D pose pool
          # self.visualize_detection(img, det, classes, thresh)
          # self.save_detection(img, det, 'output/'+im_list[k].split('/')[-1]+'2D.png',thresh=0.0)        
          out = draw_detections_3D(img / 255.0, final, cam, model_map, thresh)
          #cv2.imshow('6D pools', out)
          cv2.imwrite('./seq6_ref_80-6_visualize_n/'+im_list[k].split('/')[-1], out*255)
          # print("Number %d has been written." %det)
          print(im_list[k])
          # cv2.imshow('Final poses', draw_detections_3D(img/255.0, final, cam, model_map, thresh))
          cv2.waitKey()
    return final

  def detect_and_save(self, im_list, save_dir, colors, root_dir=None, extension=None,
                      classes=[], thresh=0.6, show_timer=False):
    """
    Wrapper for im_detect and save_detection

    Parameters:
    ----------
    im_list : list of str or str
        image path or list of image paths
    root_dir : str or None
        directory of input images, optional if image path already
        has full directory information
    extension : str or None
        image extension, eg. ".jpg", optional
    """
    import cv2
    dets = self.im_detect(im_list, root_dir, extension, show_timer=show_timer)
    if not isinstance(im_list, list):
      im_list = [im_list]
    assert len(dets) == len(im_list)

    if root_dir:
      for k, det in enumerate(dets):
        img = cv2.imread(root_dir + im_list[k])
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
        print(save_dir + im_list[k])
        self.save_detection(img, det, save_dir + im_list[k], colors, classes, thresh)
    else:
      for k, det in enumerate(dets):
        img = cv2.imread(im_list[k])
        img[:, :, (0, 1, 2)] = img[:, :, (2, 1, 0)]
        self.save_detection(img, det, save_dir + im_list[k], colors, classes, thresh)
