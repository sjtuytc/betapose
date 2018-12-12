# Include class KPModel3D and Model3D, especially the loading function.
import os
import numpy as np
from scipy.spatial.distance import pdist
from plyfile import PlyData
import cv2
from vispy import gloo
from IPython import embed #debugging
from utils import output_pointcloud

class Model3D:
    def __init__(self, file_to_load=None):
        self.vertices = None
        self.centroid = None
        self.indices = None
        self.colors = None
        self.texcoord = None
        self.texture = None
        self.collated = None
        self.vertex_buffer = None
        self.index_buffer = None
        self.bb = None
        self.bb_vbuffer = None
        self.bb_ibuffer = None
        self.diameter = None
        if file_to_load:
            self.load(file_to_load)

    def refine(self, total_kp = 30, save = False, save_path = "test.ply"):
        vertices = self.vertices
        flag = True
        min_index = 0
        reduce_maximum = len(vertices) - total_kp
        for reduce_counter in range(reduce_maximum):
            # In every loop, the closest point will be deleted
            min_dist = 100.000
            for i, vi in enumerate(vertices):
                for j, vj in enumerate(vertices):
                    if i==j:
                        continue
                    current_dist = np.sqrt(np.sum(np.square(vi - vj)))
                    if (current_dist<min_dist):
                        min_index = i
                        min_dist = current_dist
            vertices = np.delete(vertices, min_index, 0)
        self.vertices = vertices
        if save:
            output_pointcloud(vertices, save_path)

    def _compute_bbox(self):

        self.bb = []
        minx, maxx = min(self.vertices[:, 0]), max(self.vertices[:, 0])
        miny, maxy = min(self.vertices[:, 1]), max(self.vertices[:, 1])
        minz, maxz = min(self.vertices[:, 2]), max(self.vertices[:, 2])
        self.bb.append([minx, miny, minz])
        self.bb.append([minx, maxy, minz])
        self.bb.append([minx, miny, maxz])
        self.bb.append([minx, maxy, maxz])
        self.bb.append([maxx, miny, minz])
        self.bb.append([maxx, maxy, minz])
        self.bb.append([maxx, miny, maxz])
        self.bb.append([maxx, maxy, maxz])
        self.bb = np.asarray(self.bb, dtype=np.float32)
        #self.diameter = max(pdist(self.bb, 'euclidean'))

        # Set up rendering data
        colors = [[1, 0, 0],[1, 1, 0], [0, 1, 0], [0, 1, 1],
                  [0, 0, 1], [0, 1, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
        indices = [0, 1, 0, 2, 3, 1, 3, 2,
                   4, 5, 4, 6, 7, 5, 7, 6,
                   0, 4, 1, 5, 2, 6, 3, 7]

        vertices_type = [('a_position', np.float32, 3), ('a_color', np.float32, 3)]
        collated = np.asarray(list(zip(self.bb, colors)), vertices_type)
        self.bb_vbuffer = gloo.VertexBuffer(collated)
        self.bb_ibuffer = gloo.IndexBuffer(indices)

    def load(self, path, demean=False, scale=1.0):
        data = PlyData.read(path)
        self.vertices = np.zeros((data['vertex'].count, 3))
        self.vertices[:, 0] = np.array(data['vertex']['x'])
        self.vertices[:, 1] = np.array(data['vertex']['y'])
        self.vertices[:, 2] = np.array(data['vertex']['z'])
        self.vertices *= scale