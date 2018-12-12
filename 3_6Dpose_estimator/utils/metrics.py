import os
import sys
this_dir = os.path.dirname(__file__)
current_path = os.path.join(this_dir)
sys.path.append(current_path)
import numpy as np
from pyquaternion import Quaternion
from IPython import embed

def add_err(gt_pose, est_pose, model):
    def transform_points(points_3d, mat):
        rot = np.matmul(mat[:3, :3], points_3d.transpose())
        return rot.transpose() + mat[:3, 3]
    # print("Now showing model.vertices...")
    # print(model.vertices)
    v_A = transform_points(model, gt_pose)
    v_B = transform_points(model, est_pose)
    # print("Now showing transform_points...")
    # print(v_A)
    v_A = np.array([x for x in v_A])
    v_B = np.array([x for x in v_B])    
    return np.mean(np.linalg.norm(v_A - v_B, axis=1))
    # # Handling symmetric objects.
    # error = []
    # for idx_A, perv_A in enumerate(v_A):
    #     if idx_A > 4: break
    #     min_error_perv_A = 10000.0
    #     for idx_B, perv_B in enumerate(v_B):
    #         # if idx_B > 100: break
    #         if np.linalg.norm(perv_A - perv_B)<min_error_perv_A:
    #             min_error_perv_A = np.linalg.norm(perv_A - perv_B)
    #     error.append(min_error_perv_A)
    # return np.mean(error)

def rot_error(gt_pose, est_pose):
    def matrix2quaternion(m):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            S = np.sqrt(tr + 1.0) * 2
            qw = 0.25 * S
            qx = (m[2, 1] - m[1, 2]) / S
            qy = (m[0, 2] - m[2, 0]) / S
            qz = (m[1, 0] - m[0, 1]) / S
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            S = np.sqrt(1. + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            qw = (m[2, 1] - m[1, 2]) / S
            qx = 0.25 * S
            qy = (m[0, 1] + m[1, 0]) / S
            qz = (m[0, 2] + m[2, 0]) / S
        elif m[1, 1] > m[2, 2]:
            S = np.sqrt(1. + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            qw = (m[0, 2] - m[2, 0]) / S
            qx = (m[0, 1] + m[1, 0]) / S
            qy = 0.25 * S
            qz = (m[1, 2] + m[2, 1]) / S
        else:
            S = np.sqrt(1. + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            qw = (m[1, 0] - m[0, 1]) / S
            qx = (m[0, 2] + m[2, 0]) / S
            qy = (m[1, 2] + m[2, 1]) / S
            qz = 0.25 * S
        return np.array([qw, qx, qy, qz])

    gt_quat = Quaternion(matrix2quaternion(gt_pose[:3, :3]))
    est_quat = Quaternion(matrix2quaternion(est_pose[:3, :3]))

    return np.abs((gt_quat * est_quat.inverse).degrees)


def trans_error(gt_pose, est_pose):
    trans_err_norm = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    trans_err_single = np.abs(gt_pose[:3, 3] - est_pose[:3, 3])

    return trans_err_norm, trans_err_single


def iou(gt_box, est_box):
    xA = max(gt_box[0], est_box[0])
    yA = max(gt_box[1], est_box[1])
    xB = min(gt_box[2], est_box[2])
    yB = min(gt_box[3], est_box[3])

    if xB <= xA or yB <= yA:
        return 0.

    interArea = (xB - xA) * (yB - yA)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    boxBArea = (est_box[2] - est_box[0]) * (est_box[3] - est_box[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    return interArea / float(boxAArea + boxBArea - interArea)


def projection_error_2d(gt_pose, est_pose, model, cam):
    """Compute 2d projection error
    Args
    - gt_pose: (np.array) [4 x 4] pose matrix
    - est_pose: (np.array) [4 x 4] pose matrix
    - model: (np.array) [N x 3] model 3d vertices
    - cam: (np.array) [3 x 3] camera matrix
    """
    gt_pose = gt_pose[:3]
    est_pose = est_pose[:3]
    model = np.concatenate((model, np.ones((model.shape[0], 1))), axis=1)

    gt_2d = np.matmul(np.matmul(cam, gt_pose), model.T)
    est_2d = np.matmul(np.matmul(cam, est_pose), model.T)

    gt_2d /= gt_2d[2, :]
    est_2d /= est_2d[2, :]
    gt_2d = gt_2d[:2,:].T
    est_2d = est_2d[:2,:].T
    
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.scatter(gt_2d[:,0], gt_2d[:,1], s=1)
    # plt.savefig('gt.png')
    # plt.figure()
    # plt.scatter(est_2d[:,0], est_2d[:,1], s=1)
    # plt.savefig('pred.png')

    return np.mean(np.linalg.norm(gt_2d - est_2d, axis=1))