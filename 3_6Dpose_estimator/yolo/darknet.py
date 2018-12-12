from __future__ import division

import torch 
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
import numpy as np
import cv2 
import matplotlib.pyplot as plt
try:
    from util import count_parameters as count
    from util import convert2cpu as cpu
    from util import predict_transform
except ImportError:
    from yolo.util import count_parameters as count
    from yolo.util import convert2cpu as cpu
    from yolo.util import predict_transform

from IPython import embed
from collections import defaultdict

class test_net(nn.Module):
    def __init__(self, num_layers, input_size):
        super(test_net, self).__init__()
        self.num_layers= num_layers
        self.linear_1 = nn.Linear(input_size, 5)
        self.middle = nn.ModuleList([nn.Linear(5,5) for x in range(num_layers)])
        self.output = nn.Linear(5,2)
    
    def forward(self, x):
        x = x.view(-1)
        fwd = nn.Sequential(self.linear_1, *self.middle, self.output)
        return fwd(x)
        
def get_test_input():
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (416,416)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    img_ = Variable(img_)
    return img_


def parse_cfg(cfgfile):
    """
    Takes a configuration file
    
    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    
    """
    file = open(cfgfile, 'r')
    lines = file.read().split('\n')     #store the lines in a list
    lines = [x for x in lines if len(x) > 0] #get read of the empty lines 
    lines = [x for x in lines if x[0] != '#']  
    lines = [x.rstrip().lstrip() for x in lines]

    
    block = {}
    blocks = []
    
    for line in lines:
        if line[0] == "[":               #This marks the start of a new block
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key,value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks
#    print('\n\n'.join([repr(x) for x in blocks]))

import pickle as pkl

class MaxPoolStride1(nn.Module):
    def __init__(self, kernel_size):
        super(MaxPoolStride1, self).__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size - 1

    def forward(self, x):
        padding = int(self.pad / 2)
        #padded_x = F.pad(x, (0,self.pad,0,self.pad), mode="replicate")
        #pooled_x = nn.MaxPool2d(self.kernel_size, self.pad)(padded_x)
        #padded_x = F.pad(x, (0, self.pad, 0, self.pad), mode="replicate")
        padded_x = F.pad(x, (padding, padding, padding, padding), mode="constant", value=0)
        pooled_x = nn.MaxPool2d(self.kernel_size, 1)(padded_x)
        return pooled_x


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()
        

# class DetectionLayer(nn.Module):
#     def __init__(self, anchors):
#         super(DetectionLayer, self).__init__()
#         self.anchors = anchors

#     def forward(self, x, inp_dim, num_classes, confidence):
#         x = x.data
#         global CUDA
#         prediction = x
#         prediction = predict_transform(prediction, inp_dim, self.anchors, num_classes, confidence, CUDA)
#         return prediction

class DetectionLayer(nn.Module):
  """Detection layer

  @Args
    anchors: (list) list of anchor box sizes tuple
    num_classes: (int) # classes
    reso: (int) original image resolution
    ignore_thresh: (float)
  """

  def __init__(self, anchors, num_classes, reso, ignore_thresh):
    super(DetectionLayer, self).__init__()
    self.anchors = anchors
    self.num_classes = num_classes
    self.reso = reso
    self.ignore_thresh = ignore_thresh

  def forward(self, x, y_true=None):
    """
    Transform feature map into 2-D tensor. Transformation includes
    1. Re-organize tensor to make each row correspond to a bbox
    2. Transform center coordinates
      bx = sigmoid(tx) + cx
      by = sigmoid(ty) + cy
    3. Transform width and height
      bw = pw * exp(tw)
      bh = ph * exp(th)
    4. Activation

    @Args
      x: (Tensor) feature map with size [bs, (5+nC)*nA, gs, gs]
        5 => [4 offsets (xc, yc, w, h), objectness]

    @Returns
      detections: (Tensor) feature map with size [bs, nA, gs, gs, 5+nC]
    """
    bs, _, gs, _ = x.size()
    stride = self.reso // gs
    num_attrs = 5 + self.num_classes
    nA = len(self.anchors)
    scaled_anchors = torch.Tensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors]).cuda()
    grid_x = torch.arange(gs).repeat(gs, 1).view([1, 1, gs, gs]).float().cuda()
    grid_y = torch.arange(gs).repeat(gs, 1).t().view([1, 1, gs, gs]).float().cuda()
    anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
    anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

    # Re-organize [bs, (5+nC)*nA, gs, gs] => [bs, nA, gs, gs, 5+nC]
    x = x.view(bs, nA, num_attrs, gs, gs).permute(0, 1, 3, 4, 2).contiguous()
    detections = torch.Tensor(bs, nA, gs, gs, num_attrs).cuda()
    detections[..., 0] = torch.sigmoid(x[..., 0]) + grid_x
    detections[..., 1] = torch.sigmoid(x[..., 1]) + grid_y
    detections[..., 2] = torch.exp(x[..., 2]) * anchor_w
    detections[..., 3] = torch.exp(x[..., 3]) * anchor_h
    detections[..., :4] *= stride
    detections[..., 4] = torch.sigmoid(x[..., 4])
    detections[..., 5:] = torch.sigmoid(x[..., 5:])

    return detections.view(bs, -1, num_attrs)
        
class Upsample(nn.Module):
    def __init__(self, stride=2):
        super(Upsample, self).__init__()
        self.stride = stride
        
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B = x.data.size(0)
        C = x.data.size(1)
        H = x.data.size(2)
        W = x.data.size(3)
        ws = stride
        hs = stride
        x = x.view(B, C, H, 1, W, 1).expand(B, C, H, stride, W, stride).contiguous().view(B, C, H*stride, W*stride)
        return x
#       
        
class ReOrgLayer(nn.Module):
    def __init__(self, stride = 2):
        super(ReOrgLayer, self).__init__()
        self.stride= stride
        
    def forward(self,x):
        assert(x.data.dim() == 4)
        B,C,H,W = x.data.shape
        hs = self.stride
        ws = self.stride
        assert(H % hs == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(H)
        assert(W % ws == 0),  "The stride " + str(self.stride) + " is not a proper divisor of height " + str(W)
        x = x.view(B,C, H // hs, hs, W // ws, ws).transpose(-2,-3).contiguous()
        x = x.view(B,C, H // hs * W // ws, hs, ws)
        x = x.view(B,C, H // hs * W // ws, hs*ws).transpose(-1,-2).contiguous()
        x = x.view(B, C, ws*hs, H // ws, W // ws).transpose(1,2).contiguous()
        x = x.view(B, C*ws*hs, H // ws, W // ws)
        return x


class Darknet(nn.Module):
  """YOLO v3 model

  @Args
    cfgfile: (str) path to yolo v3 config file  
    reso: (int) original image resolution  
  """

  def __init__(self, cfgfile, reso = 416):
    super(Darknet, self).__init__()
    self.blocks = parse_cfg(cfgfile)
    self.reso = reso
    self.net_info, self.module_list = self.build_model(self.blocks)

  def build_model(self, blocks):
    """
    @Args
      blocks: (list) list of building blocks description

    @Returns
      module_list: (nn.ModuleList) module list of neural network
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    in_channels = 3  # start from RGB 3 channels
    out_channels_list = []

    for idx, block in enumerate(blocks):
      module = nn.Sequential()

      # Convolutional layer
      if block['type'] == 'convolutional':
        activation = block['activation']
        try:
          batch_normalize = int(block['batch_normalize'])
          bias = False
        except:
          batch_normalize = 0
          bias = True
        out_channels = int(block['filters'])
        kernel_size = int(block['size'])
        padding = (kernel_size - 1) // 2 if block['pad'] else 0
        stride = int(block['stride'])
        conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        module.add_module("conv_{0}".format(idx), conv)

        if batch_normalize != 0:
          module.add_module("bn_{0}".format(idx), nn.BatchNorm2d(out_channels))

        if activation == "leaky":  # for yolo, it's either leaky ReLU or linear
          module.add_module("leaky_{0}".format(idx), nn.LeakyReLU(0.1, inplace=True))

      # Max pooling layer
      elif block['type'] == 'maxpool':
        stride = int(block["stride"])
        size = int(block["size"])
        if stride != 1:
          maxpool = nn.MaxPool2d(size, stride)
        else:
          maxpool = MaxPool1s(size)

        module.add_module("maxpool_{}".format(idx), maxpool)

      # Up sample layer
      elif block['type'] == 'upsample':
        stride = int(block["stride"])  # always to be 2 in yolo-v3
        upsample = nn.Upsample(scale_factor=stride, mode="nearest")
        module.add_module("upsample_{}".format(idx), upsample)

      # Shortcut layer
      elif block['type'] == 'shortcut':
        shortcut = EmptyLayer()
        module.add_module("shortcut_{}".format(idx), shortcut)

      # Routing layer
      elif block['type'] == 'route':
        route = EmptyLayer()
        module.add_module('route_{}'.format(idx), route)

        block['layers'] = block['layers'].split(',')
        if len(block['layers']) == 1:
          start = int(block['layers'][0])
          out_channels = out_channels_list[idx+start]
        elif len(block['layers']) == 2:
          start = int(block['layers'][0])
          end = int(block['layers'][1])
          out_channels = out_channels_list[idx+start] + out_channels_list[end]

      # Detection layer
      elif block['type'] == 'yolo':
        mask = block['mask'].split(',')
        mask = [int(x) for x in mask]

        anchors = block['anchors'].split(',')
        anchors = [int(a) for a in anchors]
        anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
        anchors = [anchors[i] for i in mask]

        num_classes = int(block['classes'])
        ignore_thresh = float(block['ignore_thresh'])

        detection = DetectionLayer(anchors, num_classes, self.reso, ignore_thresh)
        module.add_module('detection_{}'.format(idx), detection)

      module_list.append(module)
      in_channels = out_channels
      out_channels_list.append(out_channels)

    return (net_info, module_list)

  def forward(self, x, y_true=None):
    """
    @Args
      x: (Tensor) input Tensor, with size[batch_size, C, H, W]

    @Returns
      detections: (Tensor) detection result with size [num_bboxes, [batch idx, x1, y1, x2, y2, p0, conf, label]]
    """
    detections = torch.Tensor().cuda()  # detection results
    outputs = dict()  # output cache for route layer
    self.loss = defaultdict(float)

    for i, block in enumerate(self.blocks):
      # Convolutional, upsample, maxpooling layer
      if block['type'] == 'convolutional' or block['type'] == 'upsample' or block['type'] == 'maxpool':
        x = self.module_list[i](x)
        outputs[i] = x

      # Shortcut layer
      elif block['type'] == 'shortcut':
        x = outputs[i-1] + outputs[i+int(block['from'])]
        outputs[i] = x

      # Routing layer, length = 1 or 2
      elif block['type'] == 'route':
        layers = block['layers']
        layers = [int(a) for a in layers]

        if len(layers) == 1:  # layers = [-3]: output layer -3
          x = outputs[i + (layers[0])]

        elif len(layers) == 2:  # layers = [-1, 61]: cat layer -1 and No.61
          layers[1] = layers[1] - i
          map1 = outputs[i + layers[0]]
          map2 = outputs[i + layers[1]]
          x = torch.cat((map1, map2), 1)  # cat with depth

        outputs[i] = x

      elif block['type'] == 'yolo':
        x = self.module_list[i][0](x)
        detections = x if len(detections.size()) == 1 else torch.cat((detections, x), 1)
        outputs[i] = outputs[i-1]  # skip

    return detections

  def load_weights(self, path, cutoff=None):
    """Load darknet weights from disk.
    YOLOv3 is fully convolutional, so only conv layers' weights will be loaded
    Darknet's weights data are organized as
      1. (optinoal) bn_biases => bn_weights => bn_mean => bn_var
      1. (optional) conv_bias
      2. conv_weights

    @Args
      path: (str) path to .weights file
      cutoff: (optinoal, int) cutting layer
    """
    fp = open(path, 'rb')
    header = np.fromfile(fp, dtype=np.int32, count=4)
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    header = torch.from_numpy(header)
    # embed()
    ptr = 0
    
    for i, module in enumerate(self.module_list):
      block = self.blocks[i]

      if cutoff is not None and i == cutoff:
        print("Stop before", block['type'], "block (No.%d)" % (i+1))
        break
      # print(i)
      if block['type'] == "convolutional":
        batch_normalize = int(block['batch_normalize']) if 'batch_normalize' in block else 0
        conv = module[0]

        if batch_normalize > 0:
          bn = module[1]
          num_bn_biases = bn.bias.numel()

          bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_biases = bn_biases.view_as(bn.bias.data)
          bn.bias.data.copy_(bn_biases)
          ptr += num_bn_biases

          bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_weights = bn_weights.view_as(bn.weight.data)
          bn.weight.data.copy_(bn_weights)
          ptr += num_bn_biases

          bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_running_mean = bn_running_mean.view_as(bn.running_mean)
          bn.running_mean.copy_(bn_running_mean)
          ptr += num_bn_biases

          bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
          bn_running_var = bn_running_var.view_as(bn.running_var)
          bn.running_var.copy_(bn_running_var)
          ptr += num_bn_biases

        else:
          num_biases = conv.bias.numel()
          conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
          conv_biases = conv_biases.view_as(conv.bias.data)
          conv.bias.data.copy_(conv_biases)
          ptr = ptr + num_biases

        num_weights = conv.weight.numel()
        conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
        conv_weights = conv_weights.view_as(conv.weight.data)
        conv.weight.data.copy_(conv_weights)
        ptr = ptr + num_weights

# class Darknet(nn.Module):
#     def __init__(self, cfgfile):
#         super(Darknet, self).__init__()
#         self.blocks = parse_cfg(cfgfile)
#         self.net_info, self.module_list = create_modules(self.blocks)
#         self.header = torch.IntTensor([0,0,0,0])
#         self.seen = 0

        
        
#     def get_blocks(self):
#         return self.blocks
    
#     def get_module_list(self):
#         return self.module_list

                
#     def forward(self, x, CUDA):
#         detections = []
#         modules = self.blocks[1:]
#         outputs = {}   #We cache the outputs for the route layer
        
        
#         write = 0
#         for i in range(len(modules)):        
            
#             module_type = (modules[i]["type"])
#             if module_type == "convolutional" or module_type == "upsample" or module_type == "maxpool":
                
#                 x = self.module_list[i](x)
#                 outputs[i] = x

                
#             elif module_type == "route":
#                 layers = modules[i]["layers"]
#                 layers = [int(a) for a in layers]
                
#                 if (layers[0]) > 0:
#                     layers[0] = layers[0] - i

#                 if len(layers) == 1:
#                     x = outputs[i + (layers[0])]

#                 elif len(layers) == 2:
#                     if (layers[1]) > 0:
#                         layers[1] = layers[1] - i
                        
#                     map1 = outputs[i + layers[0]]
#                     map2 = outputs[i + layers[1]]

#                     x = torch.cat((map1, map2), 1)
#                 elif len(layers) == 4:  # SPP
#                     map1 = outputs[i + layers[0]]
#                     map2 = outputs[i + layers[1]]
#                     map3 = outputs[i + layers[2]]
#                     map4 = outputs[i + layers[3]]

#                     x = torch.cat((map1, map2, map3, map4), 1)
#                 outputs[i] = x
            
#             elif  module_type == "shortcut":
#                 from_ = int(modules[i]["from"])
#                 x = outputs[i-1] + outputs[i+from_]
#                 outputs[i] = x
                
            
            
#             elif module_type == 'yolo':        
                
#                 anchors = self.module_list[i][0].anchors
#                 #Get the input dimensions
#                 inp_dim = int (self.net_info["height"])
                
#                 #Get the number of classes
#                 num_classes = int (modules[i]["classes"])
                
#                 #Output the result
#                 x = x.data
#                 x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                
#                 if type(x) == int:
#                     continue

                
#                 if not write:
#                     detections = x
#                     write = 1
                
#                 else:
#                     detections = torch.cat((detections, x), 1)
                
#                 outputs[i] = outputs[i-1]
                
        
        
#         try:
#             return detections
#         except:
#             return 0

    # the following is the original        
    # def load_weights(self, weightfile):
        
    #     #Open the weights file
    #     fp = open(weightfile, "rb")

    #     #The first 4 values are header information 
    #     # 1. Major version number
    #     # 2. Minor Version Number
    #     # 3. Subversion number 
    #     # 4. IMages seen 
    #     header = np.fromfile(fp, dtype = np.int32, count = 5)
    #     self.header = torch.from_numpy(header)
    #     self.seen = self.header[3]
        
    #     #The rest of the values are the weights
    #     # Let's load them up
    #     weights = np.fromfile(fp, dtype = np.float32)
        
    #     ptr = 0
    #     for i in range(len(self.module_list)):
    #         module_type = self.blocks[i + 1]["type"]
            
    #         if module_type == "convolutional":
    #             model = self.module_list[i]
    #             try:
    #                 batch_normalize = int(self.blocks[i+1]["batch_normalize"])
    #             except:
    #                 batch_normalize = 0
                
    #             conv = model[0]
                
    #             if (batch_normalize):
    #                 bn = model[1]
                    
    #                 #Get the number of weights of Batch Norm Layer
    #                 num_bn_biases = bn.bias.numel()
                    
    #                 #Load the weights
    #                 bn_biases = torch.from_numpy(weights[ptr:ptr + num_bn_biases])
    #                 ptr += num_bn_biases
                    
    #                 bn_weights = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr  += num_bn_biases
                    
    #                 bn_running_mean = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr  += num_bn_biases
                    
    #                 bn_running_var = torch.from_numpy(weights[ptr: ptr + num_bn_biases])
    #                 ptr  += num_bn_biases
                    
    #                 #Cast the loaded weights into dims of model weights. 
    #                 bn_biases = bn_biases.view_as(bn.bias.data)
    #                 bn_weights = bn_weights.view_as(bn.weight.data)
    #                 bn_running_mean = bn_running_mean.view_as(bn.running_mean)
    #                 bn_running_var = bn_running_var.view_as(bn.running_var)

    #                 #Copy the data to model
    #                 bn.bias.data.copy_(bn_biases)
    #                 bn.weight.data.copy_(bn_weights)
    #                 bn.running_mean.copy_(bn_running_mean)
    #                 bn.running_var.copy_(bn_running_var)
                
    #             else:
    #                 #Number of biases
    #                 num_biases = conv.bias.numel()
                
    #                 #Load the weights
    #                 conv_biases = torch.from_numpy(weights[ptr: ptr + num_biases])
    #                 ptr = ptr + num_biases
                    
    #                 #reshape the loaded weights according to the dims of the model weights
    #                 conv_biases = conv_biases.view_as(conv.bias.data)
                    
    #                 #Finally copy the data
    #                 conv.bias.data.copy_(conv_biases)
                    
                    
    #             #Let us load the weights for the Convolutional layers
    #             num_weights = conv.weight.numel()
                
    #             #Do the same as above for weights
    #             conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
    #             ptr = ptr + num_weights

    #             conv_weights = conv_weights.view_as(conv.weight.data)
    #             conv.weight.data.copy_(conv_weights)
    
    # def load_weights(self, path, cutoff=None):
    #     """Load darknet weights from disk.
    #     YOLOv3 is fully convolutional, so only conv layers' weights will be loaded
    #     Darknet's weights data are organized as
    #       1. (optinoal) bn_biases => bn_weights => bn_mean => bn_var
    #       1. (optional) conv_bias
    #       2. conv_weights

    #     @Args
    #       path: (str) path to .weights file
    #       cutoff: (optinoal, int) cutting layer
    #     """
    #     fp = open(path, 'rb')
    #     header = np.fromfile(fp, dtype=np.int32, count=4)
    #     weights = np.fromfile(fp, dtype=np.float32)
    #     fp.close()

    #     header = torch.from_numpy(header)

    #     ptr = 0
        
    #     for i, module in enumerate(self.module_list):
    #       block = self.blocks[i]

    #       if cutoff is not None and i == cutoff:
    #         print("Stop before", block['type'], "block (No.%d)" % (i+1))
    #         break

    #       if block['type'] == "convolutional":
    #         batch_normalize = int(block['batch_normalize']) if 'batch_normalize' in block else 0
    #         conv = module[0]

    #         if batch_normalize > 0:
    #           bn = module[1]
    #           num_bn_biases = bn.bias.numel()

    #           bn_biases = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
    #           bn_biases = bn_biases.view_as(bn.bias.data)
    #           bn.bias.data.copy_(bn_biases)
    #           ptr += num_bn_biases

    #           bn_weights = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
    #           bn_weights = bn_weights.view_as(bn.weight.data)
    #           bn.weight.data.copy_(bn_weights)
    #           ptr += num_bn_biases

    #           bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
    #           bn_running_mean = bn_running_mean.view_as(bn.running_mean)
    #           bn.running_mean.copy_(bn_running_mean)
    #           ptr += num_bn_biases

    #           bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases])
    #           bn_running_var = bn_running_var.view_as(bn.running_var)
    #           bn.running_var.copy_(bn_running_var)
    #           ptr += num_bn_biases

    #         else:
    #           num_biases = conv.bias.numel()
    #           conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases])
    #           conv_biases = conv_biases.view_as(conv.bias.data)
    #           conv.bias.data.copy_(conv_biases)
    #           ptr = ptr + num_biases

    #         num_weights = conv.weight.numel()
    #         conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights])
    #         conv_weights = conv_weights.view_as(conv.weight.data)
    #         conv.weight.data.copy_(conv_weights)
    #         ptr = ptr + num_weights

    # def save_weights(self, savedfile, cutoff = 0):
            
    #     if cutoff <= 0:
    #         cutoff = len(self.blocks) - 1
        
    #     fp = open(savedfile, 'wb')
        
    #     # Attach the header at the top of the file
    #     self.header[3] = self.seen
    #     header = self.header

    #     header = header.numpy()
    #     header.tofile(fp)
        
    #     # Now, let us save the weights 
    #     for i in range(len(self.module_list)):
    #         module_type = self.blocks[i+1]["type"]
            
    #         if (module_type) == "convolutional":
    #             model = self.module_list[i]
    #             try:
    #                 batch_normalize = int(self.blocks[i+1]["batch_normalize"])
    #             except:
    #                 batch_normalize = 0
                    
    #             conv = model[0]

    #             if (batch_normalize):
    #                 bn = model[1]
                
    #                 #If the parameters are on GPU, convert them back to CPU
    #                 #We don't convert the parameter to GPU
    #                 #Instead. we copy the parameter and then convert it to CPU
    #                 #This is done as weight are need to be saved during training
    #                 cpu(bn.bias.data).numpy().tofile(fp)
    #                 cpu(bn.weight.data).numpy().tofile(fp)
    #                 cpu(bn.running_mean).numpy().tofile(fp)
    #                 cpu(bn.running_var).numpy().tofile(fp)
                
            
    #             else:
    #                 cpu(conv.bias.data).numpy().tofile(fp)
                
                
    #             #Let us save the weights for the Convolutional layers
    #             cpu(conv.weight.data).numpy().tofile(fp)
               




#
#dn = Darknet('cfg/yolov3.cfg')
#dn.load_weights("yolov3.weights")
#inp = get_test_input()
#a, interms = dn(inp)
#dn.eval()
#a_i, interms_i = dn(inp)
