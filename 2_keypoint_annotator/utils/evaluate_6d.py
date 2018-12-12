import argparse
import tools.find_mxnet
import sys
#sys.path.insert(0, "/home/fred/git//flexiv_sw/3rdparty//mxnet/python")
import mxnet as mx
import os
import numpy as np
from detect_recon.detector import Detector
from symbol.symbol_factory import get_symbol
import yaml
import random
import pickle

from utils_recon.utils import precompute_projections, build_6D_poses, verify_6D_poses
from utils_recon.utils import draw_detections_2D, draw_detections_3D
from utils_recon.sixd import load_sixd
from utils_recon.metrics import *

def get_detector(net, prefix, epoch, data_shape, mean_pixels, ctx, num_class, num_tpls, num_inprots,
                 nms_thresh=0.5, force_nms=True, nms_topk=400):
    """
    wrapper for initialize a detector

    Parameters:
    ----------
    net : str
        test network name
    prefix : str
        load model prefix
    epoch : int
        load model epoch
    data_shape : int
        resize image shape
    mean_pixels : tuple (float, float, float)
        mean pixel values (R, G, B)
    ctx : mx.ctx
        running context, mx.cpu() or mx.gpu(?)
    num_class : int
        number of classes
    nms_thresh : float
        non-maximum suppression threshold
    force_nms : bool
        force suppress different categories
    """
    if net is not None:
        net = get_symbol(net, data_shape, num_classes=num_class, num_tpls = num_tpls, num_inprots = num_inprots, nms_thresh=nms_thresh,
            force_nms=force_nms, nms_topk=nms_topk)
    detector = Detector(net, prefix, epoch, data_shape, mean_pixels, ctx=ctx)
    return detector

def parse_args():
    parser = argparse.ArgumentParser(description='Single-shot detection 6D pose network demo')
    parser.add_argument('--network', dest='network', type=str, default='inceptionv3',
                        help='which network to use')
    parser.add_argument('--images', dest='images', type=str, default='./data/demo/dog.jpg',
                        help='run demo with images, use comma to seperate multiple images')
    parser.add_argument('--dir', dest='dir', nargs='?',
                        help='root dir of dataset, e.g., /data/Industry/t-less_v2/, optional', type=str)
    parser.add_argument('--list', dest='list', nargs='?',
                        help='demo list directory, optional', type=str)
    parser.add_argument('--ext', dest='extension', help='image extension, optional',
                        type=str, nargs='?')
    parser.add_argument('--epoch', dest='epoch', help='epoch of trained model',
                        default=0, type=int)
    parser.add_argument('--prefix', dest='prefix', help='trained model prefix',
                        default=os.path.join(os.getcwd(), 'model', 'ssd_'),
                        type=str)
    parser.add_argument('--seq', dest='seq', help='sequence number of the test set, e.g., 01, 02 or 20',
                        default=0,
                        type=int)
    parser.add_argument('--frames', dest='frames', help='number of frames to read, optional',
                        default=0,
                        type=int)
    parser.add_argument('--cpu', dest='cpu', help='(override GPU) use CPU to detect',
                        action='store_true', default=False)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to detect with')
    parser.add_argument('--data-shape', dest='data_shape', type=int, default=512,
                        help='set image shape')
    parser.add_argument('--mean-r', dest='mean_r', type=float, default=123,
                        help='red mean value')
    parser.add_argument('--mean-g', dest='mean_g', type=float, default=117,
                        help='green mean value')
    parser.add_argument('--mean-b', dest='mean_b', type=float, default=104,
                        help='blue mean value')
    parser.add_argument('--num-tpls', dest='num_tpls', type=int, default=642,
                        help='number of templates, default 642')
    parser.add_argument('--num-inprots', dest='num_inprots', type=int, default=4,
                        help='number of in-plane rotations, default 4')
    parser.add_argument('--thresh', dest='thresh', type=float, default=0.6,
                        help='object visualize score threshold, default 0.6')
    parser.add_argument('--nms', dest='nms_thresh', type=float, default=0.5,
                        help='non-maximum suppression threshold, default 0.5')
    parser.add_argument('--force', dest='force_nms', type=bool, default=True,
                        help='force non-maximum suppression on different class')
    parser.add_argument('--timer', dest='show_timer', type=bool, default=True,
                        help='show detection time')
    parser.add_argument('--deploy', dest='deploy_net', action='store_true', default=False,
                        help='Load network from json file, rather than from symbol')
    parser.add_argument('--class-names', dest='class_names', type=str,
                        default='aeroplane, bicycle, bird, boat, bottle, bus, \
                        car, cat, chair, cow, diningtable, dog, horse, motorbike, \
                        person, pottedplant, sheep, sofa, train, tvmonitor',
                        help='string of comma separated names, or text filename')
    parser.add_argument('--dump_dir', default='dump', help='dump folder path [dump]')
    parser.add_argument('--refinement', type=bool, default=True, help='choose if or not refine')
    parser.add_argument('--ifvisualize', type=bool, default=False, help='choose if or not visualize')
    
    #Note: there is a bug with bool variables: "--refinement False" will let refinement to be True.

    args = parser.parse_args()
    DUMP_DIR = args.dump_dir
    sequence = args.seq
    if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
    global LOG_FOUT 

    randi = random.randint(0,100)
    LOG_FOUT = open(os.path.join(DUMP_DIR, 'log_evaluate'+str(sequence)+'rand_'+str(randi)+'.txt'), 'w')
    
    return args

def parse_class_names(class_names):
    """ parse # classes and class_names if applicable """
    if len(class_names) > 0:
        if os.path.isfile(class_names):
            # try to open it to read class names
            with open(class_names, 'r') as f:
                class_names = [l.strip() for l in f.readlines()]
        else:
            class_names = [c.strip() for c in class_names.split(',')]
        for name in class_names:
            assert len(name) > 0
    else:
        raise RuntimeError("No valid class_name provided...")
    return class_names

def log_string(out_str):
    global LOG_FOUT 
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

if __name__ == '__main__':
    # print(mx.__path__)
    args = parse_args()
    if args.cpu:
        ctx = mx.cpu()
    else:
        ctx = mx.gpu(args.gpu_id)

    # parse image list
    #image_list = [i.strip() for i in args.images.split(',')]
    #image_list = [i.strip('\r').strip('\n') for i in open(args.list).readline()]
    #assert len(image_list) > 0, "No valid image specified to detect"

    # load model
    sixd_base = args.dir
    sequence = args.seq
    nr_frames = args.frames
    refinement = args.refinement
    show_timer = args.show_timer
    log_string("Exp918new_test...")
    log_string("If refinement:")
    refinement = True
    log_string(str(refinement))
    ifvisualize = args.ifvisualize
    ifvisualize = False
    print("If visualize:")    
    print(ifvisualize)
    bench = load_sixd(sixd_base, nr_frames=nr_frames, seq=sequence)
    # bench = pickle.dump(bench, open('bench', 'wb'))
    # bench = pickle.load(open('bench', 'rb'))
    class_names = parse_class_names(args.class_names)
    inplanes = [60, 55, 50, 45,40,35,30,25,20,15,10,5,0,-5,-10,-15,-20,-25,-30,-35,-40,-45,-50,-55,-60]
    viewpoints = []
    for i in range(0,10):
        for j in range(0,36):
            if i != 9:
                viewpoints.append(5*j+5*i*36*5)
            else:
                viewpoints.append(5*j + 5*8*36*5 + 4*36*5)
    #inplanes = [-180,-90,0,90]
    print('Inplanes:', len(inplanes))
    log_string('Precomputing projections for each used model...')

    # print(viewpoints)
    # print("viewpoints end .....................")
    tpl_camRs = {}
    tpl_bboxs = {}
    # tpl_cams = {}   ?
    for ID in range(len(class_names)):
        with open(os.path.join(sixd_base, 'train_render_cad/{:02d}/'.format(ID+1) + 'gt.yml'), 'r') as stream:
          a = yaml.load(stream)
          camR = []
          camT = []
          bbox = []
          # print("total viewpoints", len(viewpoints))
          # print("check obj_bb")
          # print(a[153][0]['obj_bb'])
          # print(a[255][0]['obj_bb'])
          for i in range(0,len(viewpoints)):
              camR.append(a[viewpoints[i]][0]['cam_R_m2c'])
              bbox.append(a[viewpoints[i]][0]['obj_bb'])
        tpl_camRs['{:02d}'.format(ID+1)] = camR
        tpl_bboxs['{:02d}'.format(ID+1)] = bbox

    model_map = bench.models  # Mapping from name to model3D instance
    for model_name in class_names:
        #m = model_map[model_name]
        m = model_map['{:02d}'.format(int(model_name))]
        m.projections = precompute_projections(tpl_camRs['{:02d}'.format(int(model_name))], inplanes, bench.cam, m, tpl_bboxs['{:02d}'.format(int(model_name))])
 

    network = None if args.deploy_net else args.network
    if args.prefix.endswith('_'):
        prefix = args.prefix + args.network + '_' + str(args.data_shape)
    else:
        prefix = args.prefix
    detector = get_detector(network, prefix, args.epoch,
                            args.data_shape,
                            (args.mean_r, args.mean_g, args.mean_b),
                            ctx, len(class_names), args.num_tpls, args.num_inprots, args.nms_thresh, args.force_nms)
    
    # run detection and evaluation
    adds = []
    trans_errors_norm = []
    trans_errors_single = []
    rot_errors = []
    ious = []
    TP = 0
    FN = 0
    TN = 0
    FP = 0
    # This list performs label transfer from groundtruth to prediction.
    gt_to_pre = [50, 0, 1, 50, 3, 4, 5, 50, 7, 8, 9, 10, 11, 12, 13, 14]

    # Process each frame separately
    frame_counter = -1
    for f in bench.frames:
        frame_counter = frame_counter + 1
        
        # # Testing: Jump to some given frame
        # if (frame_counter < 346):
        #     continue

        # Testing: Terminate at some certain frame
        # if frame_counter == 235 :
        #     break

        log_string("Now processing frame"+str(frame_counter))      
        final = detector.detect_and_visualize(f, gt_to_pre[sequence], frame_counter, f.path, model_map, f.cam, f.color, f.depth, ifvisualize = ifvisualize,  
                                  classes = class_names, thresh = args.thresh, show_timer = args.show_timer, img_size=(640, 480),
                                  num_tpls=args.num_tpls, num_inplane=args.num_inprots, select_tpls=5, select_inplane=5, refinement = refinement)
        
        if final == []:
            log_string("Error frame %d jumped!"%frame_counter)
            continue
        
        for gt_obj, gt_pose, gt_bbox in f.gt:
          # to bring both bboxes to the same format
          gt_bbox = [gt_bbox[0] / 640., gt_bbox[1] / 480.,
                     (gt_bbox[0] + gt_bbox[2]) / 640., (gt_bbox[1] + gt_bbox[3]) / 480.]
          # print ("ground truth label:", gt_obj)
          # # The following line is only used in seq 2 when testing simple LINEMOD!
          # if gt_obj!= 2:
          #       continue
          # print("gt_bbox:")
          # print(gt_bbox[0] * 640, gt_bbox[1] * 480, gt_bbox[2] * 640, gt_bbox[3] * 480)
          for fin in final:
              # print(fin)
                est_bbox = fin[2:6]
                est_pose = fin[6]
                est_obj = fin[0]
                est_confi = fin[1]
                est_obj = int(est_obj)
                est_obj = '{:02d}'.format(est_obj)
                # print(est_obj)
                # # print("est_bbox:")
                # # print(est_bbox[0] * 640, est_bbox[1] * 480, est_bbox[2] * 640, est_bbox[3] * 480)
                # print("Estimation confidence: ")
                # print(est_confi)
                # Calculate F1 scores
                if iou(gt_bbox, est_bbox) >= 0.5:
                    if est_obj == '{:02d}'.format(gt_to_pre[gt_obj]):
                        TP += 1
                    else:
                        FN += 1 
                else:
                    if est_obj == '{:02d}'.format(gt_to_pre[gt_obj]):
                        FP += 1
                    else:
                        TN += 1
                recall = float(TP) / (float(TP + FN) + 1e-5)
                precision = float(TP) / (float(TP + FP) + 1e-5)
                f1 = float(2 * recall * precision) / float(recall + precision + 1e-5)

                # Calculate ADD scores
                if est_obj == '{:02d}'.format(gt_to_pre[gt_obj]) :
                    ious.append(iou(gt_bbox, est_bbox) >= 0.5)
                    model = model_map['{:02d}'.format(gt_obj)]

                    if iou(gt_bbox, est_bbox) >= 0.5:
                        # log_string("Iou >= 0.5!, error and 0.1 diameter are: ")
                        log_string(str(add_err(gt_pose, est_pose, model)))
                        log_string(str((0.1 * model.diameter)))
                        print((add_err(gt_pose, est_pose, model)) < (0.1 * model.diameter)) 
                        adds.append((add_err(gt_pose, est_pose, model)) < (0.1 * model.diameter))
                        trans_errors = trans_error(gt_pose, est_pose)
                        trans_errors_norm.append(trans_errors[0])
                        trans_errors_single.append(trans_errors[1])
                        rot_errors.append(rot_error(gt_pose, est_pose))

                        # Show current detecting error
                        print("\tCurrent Trans Error: {:.3f}".format(trans_errors[0]))
                        log_string("\tCurrent Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(trans_errors[1][0],
                                                                                trans_errors[1][1],
                                                                                trans_errors[1][2]))
                        log_string("\tCurrent Rotation Error:: {:.3f}".format(rot_error(gt_pose, est_pose)))

        # show per frame result
        if ious and adds :
            mean_iou = np.mean(ious)
            mean_add = np.mean(adds)
            mean_trans_error_norm = np.mean(trans_errors_norm)
            mean_trans_error_single = np.mean(trans_errors_single, axis=0)
            mean_rot_error = np.mean(rot_errors)

            #store in log file
            log_string("Sequence %d, "%sequence + "Frame %d."%frame_counter)
            # log_string("\tMean IOU 0.5: {:.3f}".format(mean_iou))
            log_string("\tF1 score: {:.3f}".format(f1))            
            log_string("\tMean ADD: {:.3f}".format(mean_add))
            # log_string("\tMean Trans Error Norm: {:.3f}".format(mean_trans_error_norm))
            # log_string("\tMean Trans Errors: X: {:.3f}, Y: {:.3f}, Z: {:.3f}".format(mean_trans_error_single[0],
            #                                                                     mean_trans_error_single[1],
            #                                                                     mean_trans_error_single[2]))
            # log_string("\tMean Rotation Error: {:.3f}".format(mean_rot_error))
        
