import math
import numpy as np
import cv2
from tqdm import tqdm
from scipy.linalg import expm, norm
from matplotlib import pyplot as plt
from IPython import embed #debugging
import renderer
import random
import copy

def pnp(points_3D, points_2D, cameraMatrix):
    try:
        distCoeffs = pnp.distCoeffs
    except:
        distCoeffs = np.zeros((8, 1), dtype='float32')

    assert points_3D.shape[0] == points_2D.shape[0], 'points 3D and points 2D must have same number of vertices'
    # embed()
    _, R_exp, t = cv2.solvePnP(points_3D,
                              # points_2D,
                              np.ascontiguousarray(points_2D[:,:2]).reshape((-1,1,2)),
                              cameraMatrix,
                              distCoeffs)
                              # , None, None, False, cv2.SOLVEPNP_UPNP)
                                
    # R_exp, t, _ = cv2.solvePnPRansac(points_3D,
    #                            points_2D,
    #                            cameraMatrix,
    #                            distCoeffs,
    #                            reprojectionError=12.0)
    # 

    R, _ = cv2.Rodrigues(R_exp)
    # Rt = np.c_[R, t]
    return R, t

def handle_occlusion(real_kp_depth, real_kp_label, real_all_depth):
    '''Handle occlusion by removing pixels that are previously marked with 1
     in real_all_depth.
    '''
    # embed()
    for i in range(len(real_kp_depth)):
        for j in range(len(real_kp_depth[i])):
            # loop for every key point as center point
            if (real_kp_depth[i][j][0] > 0):
                if (real_all_depth[i][j][0] == 1): #occluded
                    # print("Delete one kp due to occluding!")
                    real_kp_depth[i][j][:] = [0, 0, 0]
                    real_kp_label[i][j][:] = real_kp_label[i][j][:] + 66
                    # special mark for occluded kp
    return real_kp_depth, real_kp_label

def local_top(kp_raw_depth, all_depth, label_matrix, searching_radius = 5, threshold_ratio = 0.5):
    '''Remove vertices that are in the back of the model.
        Principal: prefer to save rather than kill.
    '''
    return kp_raw_depth, label_matrix
    # kp_depth = kp_raw_depth
    # min_kp_depth = 10000.000
    # max_kp_depth = np.max(kp_depth)
    # for i in range(len(kp_raw_depth)):
    #     for j in range(len(kp_raw_depth[i])):
    #         if kp_raw_depth[i][j]<min_kp_depth and kp_raw_depth[i][j] > 3:
    #             min_kp_depth = kp_raw_depth[i][j]
    # threshold = threshold_ratio * (max_kp_depth - min_kp_depth)
    # # print ("threshold_ratio is: ", threshold_ratio)
    # for i in range(len(kp_raw_depth)):
    #     for j in range(len(kp_raw_depth[i])):
    #         if kp_raw_depth[i][j] != 0:
    #         # loop for every key point as center point
    #             xmin = max(0, i - searching_radius)
    #             xmax = min(len(kp_raw_depth), i + searching_radius)
    #             ymin = max(0, j - searching_radius)
    #             ymax = min(len(kp_raw_depth[i]), j + searching_radius)
    #             # # For testing: print all ratios to choose the suitable one
    #             # if all_depth[i][j] > 1:
    #             #     current_ratio = (kp_raw_depth[i][j] - all_depth[i][j])\
    #             #     /(max_kp_depth - min_kp_depth)
    #             #     if current_ratio > 0 :
    #             #         print("Current ratio is:", current_ratio)
    #             for k in range(xmin, xmax):
    #                 if kp_depth[i][j] == 0:
    #                     break
    #                 for l in range(ymin, ymax):
    #                     if (kp_raw_depth[i][j] - all_depth[k][l]) > threshold:
    #                         '''
    #                          Means that the all_depth concludes a very near pixel,
    #                          so the corresponding pixel in kp_raw_depth must be in back.
    #                         '''
    #                         if all_depth[k][l] > 1:
    #                             current_ratio = (kp_raw_depth[i][j] - all_depth[k][l])\
    #                             /(max_kp_depth - min_kp_depth)
    #                             # print("current_ratio", current_ratio)
    #                             # print("Delete one kp in back!")
    #                             kp_depth[i][j] = 0
    #                             label_matrix[i][j] = 0
    #                             break
    # # embed()
    # return kp_depth, label_matrix

def trans_vertices_by_pose(ori_vertices, pose):
    # make points homogeneous, copy them to maintain the originals
    ori_vertices = np.array(ori_vertices, )
    ext_model_points = np.ones((4,ori_vertices.shape[0]))
    ext_model_points[:3,:] = np.copy(ori_vertices.T)
    trans_vertices = np.dot(np.array(pose), ext_model_points)
    # Transfer points to original form
    trans_vertices = trans_vertices.T
    trans_vertices = np.copy(trans_vertices[:, :3])
    return trans_vertices

def print_kp_result_distance(c):
    for x, y in enumerate(c):
        print(x,abs(y[0])+abs(y[1]))

def jitter_bbox(bbox, jitter):
    '''Jitter given bbox, a way of data augmentation.
        bbox: [xmin, ymin, xmax, ymax]
    '''
    newbbox = copy.copy(bbox)
    oh = bbox[3] - bbox[1];
    ow = bbox[2] - bbox[0];
    dw = (ow*jitter);
    dh = (oh*jitter);
    pleft  = int(random.uniform(-dw, dw));
    pright = int(random.uniform(-dw, dw));
    ptop   = int(random.uniform(-dh, dh));
    pbot   = int(random.uniform(-dh, dh));
    newbbox[0] = bbox[0] + pleft
    newbbox[1] = bbox[1] + ptop
    newbbox[2] = bbox[2] + pright
    newbbox[3] = bbox[3] + pbot
    return newbbox

def get_bbox_from_mask(mask, KP=False):
    '''Given a mask, this returns the bounding box annotations

    Args:
        mask(NumPy Array): Array with the mask
    Returns:
        tuple: Bounding box annotation (xmin, xmax, ymin, ymax)
    '''
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if len(np.where(rows)[0]) > 0:
        ymin, ymax = np.where(rows)[0][[0, -1]]
        xmin, xmax = np.where(cols)[0][[0, -1]]
        return xmin, xmax, ymin, ymax
    else:
        return -1, -1, -1, -1

def rotate(image, angle, center=None, scale=1.0, if_INTER_NEAREST = False):
    # grab the dimensions of the image
    (h, w) = image.shape[:2]

    # if the center is None, initialize it as the center of
    # the image
    if center is None:
        center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    if if_INTER_NEAREST:
        rotated = cv2.warpAffine(image, M, (w, h),flags=cv2.INTER_NEAREST)
    else:
        rotated = cv2.warpAffine(image, M, (w, h))
    # return the rotated image
    return rotated

def visualize_img(img, ifdepth = False, filename = "test.png"):
    """Visualize input img.
        Args: nparray with shape H * W * 3
    """
    output_img = np.zeros((480, 640, 3))
    if ifdepth:
        for i in range(len(img)):
            for j in range(len(img[i])):
                if img[i][j]!= 0:
                    output_img[i][j][0] = 255
                    output_img[i][j][1] = 255
                    output_img[i][j][2] = 255
    else:
        output_img = img
 #    plt.annotate(r'00',
 # xy=(1, 3), xycoords='data',
 # xytext=(+10, +10), textcoords='offset points', fontsize=16,
 # arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    cv2.imwrite(filename, img)
    plt.imshow(img)
    plt.show()
    # x,y is the kp coordinates, xytext is the location of annotation

    return img

def visualize_kp_in_img(img, keypoints):
    """Visualize input img.
        Args: nparray with shape H * W * 3
    """
    output_img = np.zeros((480, 640, 3))
    output_img = img
    # embed()
    plt.imshow(img)
    for idx, kp in enumerate(keypoints):
        x = kp[0]
        y = kp[1]
        show_idx = str(idx)
        # x,y is the kp coordinates, xytext is the location of annotation
        plt.annotate(show_idx,
         xy=(x, y), xycoords='data',
         xytext=(+10, +10), textcoords='offset points', fontsize=4,
         arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    # cv2.imwrite("test_kp_in_img", img)
    plt.show()

    return img

def generate_mask_img(depth, if_all_black = True):
    """Generate depth masked img.
        Args:
        depth: H*W np array(480*640)
    """
    img = np.zeros((480, 640, 3))
    for _h in range(0, 480):
        for _w in range(0, 640):
            if depth[_h][_w]!=0:
                if if_all_black:
                    img[_h][_w][0] = 255
                    img[_h][_w][1] = 255
                    img[_h][_w][2] = 255
                else:
                    img[_h][_w][0] = depth[_h][_w]
                    img[_h][_w][1] = depth[_h][_w]
                    img[_h][_w][2] = depth[_h][_w]
    return img


def print_all_nonzero(A):
    x, y = np.nonzero(A)
    print("Indexes are", x, y)
    print("Values are", A[x, y])

def draw_detections_2D(image, detections):
    """Draws detections onto resized image with name and confidence

        Parameters
        ----------
        image: Numpy array, normalized to [0-1]
        detections: A list of detections for this image, coming from SSD.detect() in the form
            [l, t, r, b, name, confidence, .....]

    """
    out = np.copy(image)
    for det in detections:
        lt = (int(det[0] * image.shape[1]), int(det[1] * image.shape[0]))
        rb = (int(det[2] * image.shape[1]), int(det[3] * image.shape[0]))
        text = '{}: {:.2f}'.format(det[4], det[5])
        cv2.rectangle(out, lt, rb, (0., 1., 0.), 2)
        cv2.putText(out, text, lt, 0, 0.8, (0., 1., 0.), 2)
    return out


def draw_detections_3D(image, detections, cam, model_map, thres):
    """Draws 6D detections onto resized image

        Parameters
        ----------
        image: Numpy array, normalized to [0-1]
        detections: A list of detections for this image, coming from SSD.detect() in the form
            [ name, confidence,l, t, r, b, 6D_pose0, ..., 6D_poseN]
        cam: Intrinsics for rendering
        model_map: Mapping of model name to Model3D instance {'obj': model3D}

    """
    if not detections:
        return np.copy(image)

    ren = Renderer((image.shape[1], image.shape[0]), cam)
    ren.clear()
    ren.set_cam(cam)
    out = np.copy(image)
    for det in detections:
        if det[1] < thres:
            break
        model = model_map['{:02d}'.format(int(det[0])+1)]
        for pose in det[6:]:
            #pose = [[0.32426249,0.94596714,0. ,0.02060331692],[0.45433376,-0.15573839,-0.87711253,0.0040045299],[-0.82971963,0.2844147,-0.48028493,0.72818325105],[0,0,0,1]]
            #print('ssss')
            #ren.draw_model(model, pose)
            ren.draw_boundingbox(model, pose)
    col, dep = ren.finish()
    # Copy the rendering over into the scene
    mask = np.dstack((dep, dep, dep)) > 0
    out[mask] = col[mask]
    return out


def compute_rotation_from_vertex(vertex):
    """Compute rotation matrix from viewpoint vertex """
    up = [0, 0, 1]
    if vertex[0] == 0 and vertex[1] == 0 and vertex[2] != 0:
        up = [-1, 0, 0]
    rot = np.zeros((3, 3))
    rot[:, 2] = -vertex / norm(vertex)  # View direction towards origin
    rot[:, 0] = np.cross(rot[:, 2], up)
    rot[:, 0] /= norm(rot[:, 0])
    rot[:, 1] = np.cross(rot[:, 0], -rot[:, 2])
    return rot.T


def create_pose(camR, scale=0, angle_deg=0):
    """Compute rotation matrix from viewpoint vertex and inplane rotation """
    rot = camR.reshape(3,3)
    transform = np.eye(4)
    rodriguez = np.asarray([0, 0, 1]) * (angle_deg * math.pi / 180.0)
    angle_axis = expm(np.cross(np.eye(3), rodriguez))
    transform[0:3, 0:3] = np.matmul(angle_axis, rot)
    transform[0:3, 3] = [0, 0, scale]
    return transform


def precompute_projections(camR, inplanes, cam, model3D, bbox_list):
    """Precomputes the projection information needed for 6D pose construction

    # Arguments
        camR: List of cam_R_m2c
        inplanes: List of inplane angles in degrees
        cam: Intrinsics to use for translation estimation
        model3D: Model3D instance

    # Returns
        data: a 3D list with precomputed entities with shape
            (views, inplanes, (4x4 pose matrix, 3) )
        3: norm_centroid_x, norm_centroid_y, lr
    """
    # w, h = 400.0, 400.0
    #ren = Renderer((w, h), cam)
    data = []
    if model3D.vertices is None:
        return data

    print(len(camR))
    for v in tqdm(range(len(camR))):
        data.append([])
        for i in inplanes:
            pose = create_pose(np.array(camR[v]), angle_deg=i)
            pose[:3, 3] = [0,0,0.5]  # zr = 0.65

            # Render object and extract tight 2D bbox and projected 2D centroid
            #ren.clear()
            #ren.draw_model(model3D, pose)
            #box = np.argwhere(ren.finish()[1])  # Deduct bbox from depth rendering
            bbox = np.array(bbox_list[v])
            box = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            centroid = np.matmul(pose[:3, :3], model3D.centroid) + pose[:3, 3]
            
            centroid_x = cam[0, 2] + centroid[0] * cam[0, 0] / centroid[2]
            centroid_y = cam[1, 2] + centroid[1] * cam[1, 1] / centroid[2]
            

            # Compute 2D centroid position in normalized, box-local reference frame
            box_w, box_h = (box[2] - box[0]), (box[3] - box[1])
            norm_centroid_x = (centroid_x - box[0]) / float(box_w)
            norm_centroid_y = (centroid_y - box[1]) / float(box_h)
            # Compute normalized diagonal box length
            lr = np.sqrt((box_w) ** 2 + (box_h) ** 2)
            
            data[-1].append((pose, [norm_centroid_x, norm_centroid_y, lr]))
    return data

def create_src_pointcloud(mask, color, depth, cam, model_map, img_size=(720, 540), num_tpls=641, num_inplane=4, select_tpls=3, select_inplane=1):
    '''Convert partial img in bbox to pointcloud
    # Arguments
        mask: Mask using to choose suitable outlines, size: D*VI*H*W
        color, depth: bench img in RGB/D, size: H*W*3/H*W
                     (in this function color is not used)
        cam: Intrinsics to use for backprojection

    # Returns
        points: D*vi*N*3 pointcloud, where D means number of dets, N depends on the 
        size of bbox,  3 means XYZ coordinates 
    '''
    width = img_size[0]
    height = img_size[1]
    points = []
        # print("Number of dets in one img is %d." %len(image_dets))
    for det in mask:
        det_points = []
        for vi in det:
            vi_points = []
            # Cover all the points in one img
            for _h in range(img_size[1]):
                for _w in range(img_size[0]):
                    if vi[_h][_w] >0: # If not masked
                        if depth[_h][_w] > 0.001: # Avoid storing (0.0, 0.0, 0.0)
                        # Backproject depth img to pointcloud
                            tmp_z = depth[_h][_w]  # Derive depth from depth img
                            tmp_x = tmp_z * (_w - cam[0, 2]) / cam[0, 0]
                            tmp_y = tmp_z * (_h - cam[1, 2]) / cam[1, 1]
                            tmp_xyz = [tmp_x, tmp_y, tmp_z] 
                            vi_points.append(tmp_xyz)
            det_points.append(vi_points)
        points.append(det_points)
    return points

def create_mask(des_points, cam, img_size = (720, 540)):
    ''' Create mask for creating src_points
    Arguments
        des_points: D * VI * N * 3 points, D is number of detections, VI is number of v,i
                    pair, default VI = 25.
        cam: Intrinsics to use for projection
    Return
        mask: D*VI*H*W, mask == 1/0 means the pixel should be/not be lifted
    '''
    mask = []
    for det in des_points:
        mask_det = []
        for vi in det:    
            # Initialize mask with all zero
            mask_vi = []
            for _h in range(img_size[1]):
                mask_vi.append([])                
                for _w in range(img_size[0]):
                    mask_vi[_h].append(0)

            for point in vi:
                px = point[0]
                py = point[1]
                pz = point[2]
                # Project 3D point to pixel space
                x = px * cam[0, 0] / pz + cam[0, 2]
                y = py * cam[1, 1] / pz + cam[1, 2]
                # Activate corresponding mask
                if (int(y) > 0) and (int(y) < img_size[1]) and (int(x) > 0) and (int(x) < img_size[0]): 
                    mask_vi[int(y)][int(x)] = 1
            mask_det.append(mask_vi)
        mask.append(mask_det)
    return mask

def output_pointcloud(points, output_filename, scale_back = 1000):
    """ Output pointcloud in ply format for visualization
    Arguments
        points: N*3 list
        output_filename: points will be saved in file named output_filename
    """
    with open(output_filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {0}\n".format(len(points)))
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n") 
        f.write("end_header\n")
        for point in points:
            f.write("{x} {y} {z} \n".format(
                x = point[0] * scale_back,
                y = point[1] * scale_back,
                z = point[2] * scale_back,
            ))
    print("Pointcloud has been saved in file %s" %output_filename)

def log_string(LOG_FOUT, out_str): 
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def build_6D_poses(detections, model_map, cam, img_size=(720, 540), num_tpls=641, num_inplane=4, select_tpls=3, select_inplane=1):
    """Processes the detections to build full 6D poses

    # Arguments
        detections: List of predictions for every image. Each prediction is:
                [label, confidence, xmin, ymin, xmax, ymax, 
                view0_conf, ..., viewN_conf, inplane0_conf, ... , inplaneM_conf]
        model_map: Mapping of model name to Model3D instance {'obj': model3D}
        cam: Intrinsics to use for backprojection

    # Returns
        new_detections: List of list of 6D predictions for every picture.
                Each prediction has the form:
                [label, confidence, xmin, ymin, xmax, ymax, 
                (pose00), ..., (poseNM)] where poseXX is a 4x4 matrix

        des_points: Model points changed by pose
    """
    new_detections = []
    des_points = []
    # print("One 6D detections contains %d images. "%len(detections))
    for image_dets in detections:
        new_image_dets = []
        image_des_points = []
        print("Number of 6D dets in one img is %d. " %len(image_dets))
        for det in image_dets:
            # print(det[0])
            det = det.tolist()
            new_det = det[:6]  # Copy over 2D bbox, label and confidence
            box_w, box_h = det[4] - det[2], det[5] - det[3]
            ls = np.sqrt((box_w*img_size[0]) ** 2 + (box_h*img_size[1]) ** 2)

            projected = model_map['{:02d}'.format(int(det[0])+1)].projections
            model_points = model_map['{:02d}'.format(int(det[0])+1)].vertices
            # Model_points: N*3 list

            # print("length of model_points is %d"%len(model_points))
            # print("length of point in model_points is %d"%len(model_points[4]))
            vicounter = 0

            # make points homogeneous, copy them to maintain the originals
            model_points = np.array(model_points)
            ext_model_points = np.ones((4,model_points.shape[0]))
            ext_model_points[:3,:] = np.copy(model_points.T)
            # print(ext_model_points.shape)

            vi_group_points = []
            for v in np.argsort(det[6:6+num_tpls])[-select_tpls:]:  
                # rank by confidence and choose select_tpls many views...
                for i in np.argsort(det[6+num_tpls:6+num_tpls+num_inplane])[-select_inplane:]:
                    if not projected:  # No pre-projections available for this model, skip...
                        new_det.append(np.eye(4))
                        continue
                    pose = projected[int(v)][i][0]
                    norm_centroid_x, norm_centroid_y, lr = projected[int(v)][i][1]
                    pose[2, 3] = 0.5 * lr / ls  # Compute depth from projective ratio
                    # print("Testing lr, ls and pose[2,3] info------------------------------------")
                    # print(lr, ls, pose[2, 3])
                    vicounter = vicounter + 1
                    # print(norm_centroid_x, norm_centroid_y, lr)
                    # Compute the new 2D centroid in pixel space
                    new_centroid_x = (det[2] + norm_centroid_x * box_w) * img_size[0]
                    new_centroid_y = (det[3] + norm_centroid_y * box_h) * img_size[1]
                    # print(new_centroid_x,new_centroid_y)
                    # Backproject into 3D metric space
                    pose[0, 3] = pose[2, 3] * (new_centroid_x - cam[0, 2]) / cam[0, 0]
                    pose[1, 3] = pose[2, 3] * (new_centroid_y - cam[1, 2]) / cam[1, 1]
                    new_det.append(pose)
                    # Apply pose
                    vi_points = np.dot(np.array(pose), ext_model_points)
                    # Transfer points to original form
                    vi_points = vi_points.T
                    vi_points = np.copy(vi_points[:, :3])
                    #print("vi points info: ", vi_points.shape)
                    vi_points = vi_points.tolist()
                    vi_group_points.append(vi_points)    
            # print("end %d v,i pairs-------------------------------------"%int(vicounter))
            new_image_dets.append(new_det)
            image_des_points.append(vi_group_points)
        new_detections.append(new_image_dets)
        des_points.append(image_des_points)

    return new_detections[0], des_points[0]


def verify_6D_poses(detections, model_map, cam, image):
    """For one image, select for each detection the best pose from the 6D pool

    # Arguments
        detections: List of predictions for one image. Each prediction is:
                [xmin, ymin, xmax, ymax, label, confidence,
                (pose00), ..., (poseNM)] where poseXX is a 4x4 matrix
        model_map: Mapping of model name to Model3D instance {'obj': model3D}
        cam: Intrinsics to use for backprojection
        image: The scene color image

    # Returns
        filtered: List of predictions for one image.
                Each prediction has the form:
                [label, confidence, xmin, ymin, xmax, ymax, pose] where pose is a 4x4 matrix

    """

    def compute_grads_and_mags(color):
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        grads = np.dstack((cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=5),
                           cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=5)))
        mags = np.sqrt(np.sum(grads**2, axis=2)) + 0.001  # To avoid div/0
        grads /= np.dstack((mags, mags))
        mask = mags < 5
        mags[mask] = 0
        grads[np.dstack((mask, mask))] = 0
        return grads, mags

    scene_grads, scene_mags = compute_grads_and_mags(image)
    scene_grads = np.reshape(scene_grads, (-1, 2))
    #cv2.imshow('mags', scene_mags)

    ren = Renderer((image.shape[1], image.shape[0]), cam)
    ren.set_cam(cam)
    filtered = []
    for det in detections:
        model = model_map['{:02d}'.format(int(det[0])+1)]
        scores = []
        for pose in det[6:]:
            ren.clear()
            ren.draw_model(model, pose)
            ren_grads, ren_mags = compute_grads_and_mags(ren.finish()[0])
            ren_grads = np.reshape(ren_grads, (-1, 2))
            dot = np.sum(np.abs(ren_grads[:, 0]*scene_grads[:, 0] + ren_grads[:, 1]*scene_grads[:, 1]))
            sum = np.sum(ren_mags>0)
            scores.append(dot / (sum+1))
        new_det = det[:6]
        # print(new_det)
        new_det.append(det[6 + np.argmax(np.asarray(scores))])  # Put best pose first
        filtered.append(new_det)

    return filtered
