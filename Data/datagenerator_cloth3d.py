''' 
@DataGenerator wapper: Inspired from Frederik Kratzert code: https://github.com/kratzert/finetune_alexnet_with_tensorflow
'''

import re
import sys, os
from struct import unpack
from math import cos, sin

import tensorflow as tf
import numpy as np
import scipy.io as sio
import random
import cv2

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import convert_to_tensor

from datetime import datetime
import time

from Data.dataaugmentation import *

IMAGENET_MEAN = tf.constant([136.0368, 120.6123, 106.9652], dtype=tf.float32)
root = ''

VOC_path = 'Data/VOCdevkit/VOC2012/ImageSets/Main/person_trainval.txt'
VOC_root = 'Data/VOCdevkit/VOC2012/JPEGImages/'

bone_template = tf.constant(np.array([[0.0, 0.1148, 0.1127, 0.1056, 0.3656, 0.3725, 0.1277, 0.3860, 0.3862, 
                          0.0565, 0.1289, 0.1286, 0.2141, 0.1446, 0.1451, 0.0801, 0.0913, 0.0955, 
                          0.2530, 0.2472, 0.2382, 0.2454, 0.0813, 0.0814, 0.1461, 0.1390, 0.1461, 
                          0.1800, 0.1817, 0.1639, 0.1663, 0.1886, 0.1877, 0.0672, 0.0666, 0.0772, 
                          0.1203, 0.0927, 0.1423, 0.0912, 0.0814, 0.1187]]).T, dtype=tf.float32)

POSE = 'pose' # flag for pose relative rotation matrices
NORMALS = 'pose_feats' # flag for pose features, i.e. normalized bone vectors
SHAPE = 'shape' # flag for shape parameters
BONES = 'shape_feats' # flag for shape features, i.e. bone length
GENDER = 'gender' # flag for gender
POINTS2D = 'points2d' # flag for 2D joints and landmarks
POINTS3D = 'points3d' # flag for 3D joints and landmarks

class DataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """
    def __init__(self, txt_file, mode, batch_size, root, shuffle, buffer_size=1000, opts=None):
        """Create a new DataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has a path string to an image. All the data belonging 
        to that image have the same prefix name as the image.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'inference'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            root: The root path to the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            opts: A list of flags to create the data based on them.

        Raises:
            ValueError: If an invalid mode is passed.

        """            
        self.txt_file = txt_file
        self.mode = mode
        self.root = root
        self.opts = opts
        if mode is not 'training' and mode is not 'inference':
            raise ValueError("Invalid mode '%s'." % (mode))

        # retrieve the data from the text file
        self._read_txt_file()

    def to_tf(self):
        ''' Convert the data into tensorflow dataset '''
        # number of samples in the dataset
        if type(self.data) is dict:
            keys = self.data.keys()
            self.data_size = len(self.data[keys[0]])
        else:
            self.data_size = len(self.data)

        # initial shuffling of the file and label lists (together!)
        if self.shuffle:
            self._shuffle_lists()

        # convert lists to TF tensor
        self.to_tensor()

        # create dataset
        data = tf.data.Dataset.from_tensor_slices((self.data, self.labels))

        # distinguish between train/infer. when calling the parsing functions
        if self.mode == 'training':
            data = data.map(self._parse_function_train, num_parallel_calls=8)

        elif self.mode == 'inference':
            data = data.map(self._parse_function_inference, num_parallel_calls=8)        

        # shuffle the first `buffer_size` elements of the dataset
        if self.shuffle:
            data = data.shuffle(buffer_size=self.buffer_size)

        # create a new dataset with batches of images
        data = data.batch(self.batch_size)

        self.data = data

    def _read_txt_file(self):
        return
        
    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        permutation = np.random.permutation(self.data_size)
        if type(self.labels) is dict:
            labels = {}
            for key in self.labels.keys():
                labels[key] = self.labels[key]
                self.labels[key] = []
            for i in permutation:
                for key in labels.keys():
                    self.labels[key].append(labels[key][i])
        else:
            labels = self.labels
            self.labels = []
            for i in permutation:
                self.labels.append(labels[i])

        if type(self.data) is dict:
            data = {}
            for key in self.data.keys():
                data[key] = self.data[key]
                self.data[key] = []
            for i in permutation:
                for key in data.keys():
                    self.data[key].append(data[key][i])
        else:
            data = self.data
            self.data = []
            for i in permutation:
                self.data.append(data[i])
                    
    def to_tensor(self):
        ''' convert lists to TF tensor '''
        if type(self.data) is dict:
            for key in self.data.keys():
                self.data[key] = tf.stack(self.data[key])
        else:
            self.data = tf.stack(self.data)

        if type(self.labels) is dict:
            for key in self.labels.keys():
                self.labels[key] = tf.stack(self.labels[key])            
        else:
            self.labels = tf.stack(self.labels)
                    
    def _parse_function_train(self, data, label):
        """ data augmentation """
        return data, label
        
    def _parse_function_inference(self, data, label):
        return data, label
        
    
class ImageDataGenerator(object):
    """Wrapper class around the new Tensorflows dataset pipeline.

    Requires Tensorflow >= version 1.12rc0
    """

    def __init__(self, txt_file, mode, batch_size, root, shuffle, augmentation=0,
                 buffer_size=1000, validate=False, GT_crop=True):
        """Create a new ImageDataGenerator.

        Recieves a path string to a text file, which consists of many lines,
        where each line has first a path string to an image and seperated by
        a space an integer, referring to the class number. Using this data,
        this class will create TensrFlow datasets, that can be used to train
        e.g. a convolutional neural network.

        Args:
            txt_file: Path to the text file.
            mode: Either 'training' or 'inference'. Depending on this value,
                different parsing functions will be used.
            batch_size: Number of images per batch.
            root: The root path to the dataset.
            shuffle: Wether or not to shuffle the data in the dataset and the
                initial file list.
            augmentation: Augmentation type 
                (0: no aug., 1: standard aug., 2: synthetic occlusion or 3: all)
            buffer_size: Number of images used as buffer for TensorFlows
                shuffling of the dataset.
            validate: is True when there are ground truth data for validating 
                the predictions.
            GT_crop: is True when a crop needed on the image. Not implemented
                in this code.

        Raises:
            ValueError: If an invalid mode is passed.
            Exception: If GT_crop is True

        """            
        start = time.time()
        self.augmentation = augmentation
        self.txt_file = txt_file
        self.root = root
        self._validate = validate
        self._GT_crop = GT_crop

        # retrieve the data from the text file
        print("\t{}: READING TEXT FILE: ".format(datetime.now()) + self.txt_file)
        self._read_txt_file()
        if self.augmentation >= 2:
            self._read_voc_txt()

        # number of samples in the dataset
        self.data_size = len(self.img_paths)

        # initial shuffling of the file and label lists (together!)
        if shuffle:
            print("\t{}: SHUFFLING LIST...".format(datetime.now()))
            self._shuffle_lists()

        # convert lists to TF tensor
        print("\t{}: CONVERTING TO TENSOR...".format(datetime.now()))
        self.img_paths = convert_to_tensor(self.img_paths, dtype=dtypes.string)
        if self.augmentation >= 2:
            self.voc_paths = convert_to_tensor(self.voc_paths, dtype=dtypes.string)

        # create dataset
        print("\t{}: CREATING DATASET OBJECT...".format(datetime.now()))
        data = tf.data.Dataset.from_tensor_slices((self.img_paths))

        # distinguish between train/infer. when calling the parsing functions
        print("\t{}: MAPPING SAMPLES...".format(datetime.now()))
        if mode == 'training':
            data = data.map(lambda filename: tf.py_func(self._parse_function, [filename], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string]), num_parallel_calls=8)
            # Rotation pads image with zeros, normalizing after this padding messes with the pixel distribution
            if self.augmentation > 1:
                data = data.map(self._augmentation2, num_parallel_calls=8)
            data = data.map(self._normalize, num_parallel_calls=8)
            if self.augmentation % 2 == 1:
                data = data.map(self._augmentation1, num_parallel_calls=8)
        elif mode == 'inference':
            data = data.map(lambda filename: tf.py_func(self._parse_function, [filename], [tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.string]), num_parallel_calls=8)
            data = data.map(self._normalize, num_parallel_calls=8)
        else:
            raise ValueError("Invalid mode '%s'." % (mode))

        # shuffle the first `buffer_size` elements of the dataset
        if shuffle:
            print("\t{}: SHUFFLING DATA...".format(datetime.now()))
            data = data.shuffle(buffer_size=buffer_size)

        # create a new dataset with batches of images
        print("\t{}: CREATING DATASET BATCH...".format(datetime.now()))
        data = data.batch(batch_size)

        self.data = data
        
        print("\tTIME REQUIRED: " + str(time.time() - start))

    def _read_txt_file(self):
        # print("USE tf.py_func!!!")
        # print("tf.py_func: decode FILENAME from binary to string")
        # print("filename = filename.decode(sys.getdefaultencoding())")
        # print("OUTSIDE tf.py_func (in parse func): image.set_shape((None, None, 3)), label.set_shape((41, 3))")
        """Read the content of the text file and store it into lists."""
        self.img_paths = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n', '').replace('\r', '')
                self.img_paths.append(line)
            
    def _read_voc_txt(self):
        self.voc_paths = []
        with open(VOC_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.replace('\n','').split(' ')
                if int(line[-1]) != -1:
                    continue
                line = VOC_root + line[0] + '.jpg'
                if not os.path.isfile(line):
                    continue
                if not os.path.isfile(line.replace('JPEGImages', 'SegmentationObject').replace('.jpg', '.png')):
                    continue
                self.voc_paths.append(line)

    def _shuffle_lists(self):
        """Conjoined shuffling of the list of paths and labels."""
        path = self.img_paths
        permutation = np.random.permutation(self.data_size)
        self.img_paths = []
        for i in permutation:
            self.img_paths.append(path[i])

    def _parse_function(self, filename):
        filename = filename.decode("utf-8") # seq/frame.png
        
        base_name, frame = filename.rsplit('/',1)
        base_name = os.path.join(self.root, base_name)
        frame = int(frame.rsplit('.',1)[0])
        
        # Read global info
        info = sio.loadmat(os.path.join(base_name, 'info.mat'))
        
        # Points 3D
        if self._validate:
            # Read body info
            info_body = sio.loadmat(os.path.join(base_name, 'info_body.mat'))
            R = info_body['rotations'][frame, :, :, :]
            j3d = info_body['joints'][frame, :, :]
            l3d = info_body['lmarks'][frame, :, :]
            p3d = np.concatenate((j3d, l3d), axis=0)
            p3d = p3d[1:, :] - p3d[0, :].reshape((1, 3)) # Align at (0, 0, 0)
            
            zRot = self.zRotMatrix(info['zrot'])
            R[0] = zRot.dot(R[0])
            p3d = zRot.dot(p3d.T).T # or p3d.dot(zRot.T)
            
            R = R.reshape((-1))
            
            shape = np.concatenate([info[SHAPE], 2 * (info[GENDER] - 0.5)], -1).astype(np.float32).reshape((-1))
        else:
            p3d = np.zeros((41, 3), np.float32)
            R = np.zeros(216, np.float32)
            shape = np.zeros(11, np.float32)
        
        # Load image
        im_size = 256
        im = cv2.imread(os.path.join(self.root, filename), cv2.IMREAD_UNCHANGED)
        bg = np.expand_dims(im[:,:,3]/255.0, -1).astype(np.float32)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32)
        im = im[:,:,:3]
        ims = np.sum(bg)
        if ims==0 or np.any(np.isnan(im)) or np.any(np.isnan(bg)):
            print('empty image in: ' + filename)
            im = np.zeros((im_size, im_size, 3), dtype=np.float32)
        else:
            if self._GT_crop: # Crop the image
                H, W, _ = im.shape
                
                trans = zRot.dot(info['trans'][:, frame].reshape((-1,1)))
                c = self.proj(info['camLoc'].reshape((-1,1)))
                bbox = np.array([[0,0],[-1.25,1.25],[-1.25,1.25]], dtype=np.float32)
                corners = trans + bbox
                corners = np.concatenate((corners, np.ones((1,2))), 0)
                coords = np.matmul(c, corners)
                coords /= coords[2,:]
                xleft, xright = coords[0,:].astype(np.int32)
                ybottom, ytop = coords[1,:].astype(np.int32)
                l = np.maximum(xright-xleft, ybottom-ytop)
                
                imc = np.zeros((l,l,3), np.float32)
                xmargin = 0
                ymargin = 0
                if xleft<0:
                    xmargin = -1*xleft
                    xleft = 0
                xright = W if xright>=W else xright
                if ytop<0:
                    ymargin = -1*ytop
                    ytop = 0
                ybottom = H if ybottom>=H else ybottom
                imc[ymargin:ymargin+ybottom-ytop, xmargin:xmargin+xright-xleft, :] = im[ytop:ybottom, xleft:xright, :]
                im = cv2.resize(imc, (im_size,im_size))
                
                bgc = np.zeros((l,l,3), np.float32)
                bgc[ymargin:ymargin+ybottom-ytop, xmargin:xmargin+xright-xleft, :] = bg[ytop:ybottom, xleft:xright, :]
                bg = cv2.resize(bgc, (im_size,im_size))
            else:
                im = cv2.resize(im, (im_size,im_size))
                bg = cv2.resize(bg, (im_size,im_size))
        
        return im, bg, p3d, np.float32(R), np.float32(shape), filename
    
    def _normalize(self, img, mask, p3d, pose, shape, filename):
        img = tf.image.per_image_standardization(img)
        # mean_img, var_img = tf.nn.moments(img, axes=(0,1))
        # img = tf.divide(tf.subtract(img, mean_img), tf.sqrt(var_img))
        # img = tf.divide(tf.subtract(img, IMAGENET_MEAN), 255.)
        return img, mask, p3d, pose, shape, filename
    
    def _augmentation1(self, img, mask, p3d, pose, shape, filename):
        # img, p3d, pose = randomFlip(img, p3d, pose)
        img, p3d, pose = randomRotation(img, p3d, pose)
        return img, mask, p3d, pose, shape, filename
    
    def _augmentation2(self, img, mask, p3d, pose, shape, filename):
        img = randomBackground(img, mask, self.voc_paths)
        img = randomVOC(img, self.voc_paths)
        return img, mask, p3d, pose, shape, filename
    
    def _rodrigues(self, axis):
        _norm = np.linalg.norm(axis) + .000001
        _w = axis / _norm
        _W = np.array([[0, -_w[2], _w[1]], [_w[2], 0, -_w[0]], [-_w[1], _w[0], 0]], np.float32)
        return np.eye(3) + _W * np.sin(_norm) + np.matmul(_W, _W) * (1 - np.cos(_norm))
        
    def _read_pc2(self, filename, frame, float16=False):
        bytes = 2 if float16 else 4
        dtype = np.float16 if float16 else np.float32
        with open(filename,'rb') as f:
            # Num points
            f.seek(16)
            nPoints = unpack('<i', f.read(4))[0]#int.from_bytes(f.read(4), 'little')
            f.seek(28)
            # Number of samples
            nSamples = unpack('<i', f.read(4))[0]#int.from_bytes(f.read(4), 'little')
            if frame > nSamples:
                print("Frame index outside size")
                print("\tN. frame: " + str(frame))
                print("\tN. samples: " + str(nSamples))
                return
            # Read frame
            size = nPoints * 3 * bytes
            f.seek(size * frame, 1) # offset from current '1'
            T = np.frombuffer(f.read(size), dtype=dtype).astype(np.float32)
            return T.reshape(nPoints, 3)
            
    def zRotMatrix(self, zrot):
        c, s = cos(zrot), sin(zrot)
        return np.array([[c, -s, 0],
                         [s,  c, 0],
                         [0,  0, 1]], np.float32)
                 
    def intrinsic(self):
        RES_X = 640
        RES_Y = 480
        f_mm             = 50 # blender default
        sensor_w_mm      = 36 # blender default
        sensor_h_mm = sensor_w_mm * RES_Y / RES_X

        fx_px = f_mm * RES_X / sensor_w_mm;
        fy_px = f_mm * RES_Y / sensor_h_mm;

        u = RES_X / 2;
        v = RES_Y / 2;

        return np.array([[fx_px, 0,     u],
                         [0,     fy_px, v],
                         [0,     0,     1]], np.float32)

    def extrinsic(self, camLoc):
        R_w2bc = np.array([[0, 1, 0],
                           [0, 0, 1],
                           [1, 0, 0]], np.float32)
        
        T_w2bc = -1 * R_w2bc.dot(camLoc)
        
        R_bc2cv = np.array([[1,  0,  0],
                            [0, -1,  0],
                            [0,  0, -1]], np.float32)
        
        R_w2cv = R_bc2cv.dot(R_w2bc)
        T_w2cv = R_bc2cv.dot(T_w2bc)

        return np.concatenate((R_w2cv, T_w2cv), axis=1)
        
    def proj(self, camLoc):
        return self.intrinsic().dot(self.extrinsic(camLoc))

        
class SMPLRDataGenerator(DataGenerator):
    '''
    SMPL data reader class. Reads the whole dataset as numpy arrays and keep it in memory.
    '''
    def __init__(self, txt_file, mode, batch_size, root, shuffle, opts, buffer_size=1000):
        # Check options
        assert BONES in opts and SHAPE in opts and POINTS3D in opts and POSE in opts and NORMALS in opts and len(opts)==5
        
        DataGenerator.__init__(self, txt_file, mode, batch_size, root, shuffle, buffer_size, opts)
                         
    def _read_txt_file(self):
        """Read the content of the text file and store it into lists."""
        self.data = {}
        for opt in [BONES, NORMALS, POINTS3D, SHAPE, POSE, 'idx']:
            self.data[opt] = []
        self.labels = {}
        
        for opt in self.opts:
            self.labels[opt] = []
        
        # Read kinematic tree
        kintree = np.transpose(np.array(sio.loadmat(os.path.dirname(os.path.abspath(__file__))+'/../SMPL/kintree.mat')['kintree'], np.int32)) - 1
        self.kintree = tf.constant(kintree, name='kintree')
        
        # Set pre-computed noise for augmentation
        joint_gaussian_noise = np.array([0.0083, 0.0069, 0.0052, 0.0080, 0.0071, 0.0050, 0.0085, 0.0069, 0.0053,
                                         0.0345, 0.0222, 0.0196, 0.0343, 0.0231, 0.0191, 0.0149, 0.0089, 0.0098,
                                         0.0535, 0.0235, 0.0242, 0.0521, 0.0223, 0.0255, 0.0167, 0.0090, 0.0108,
                                         0.0567, 0.0243, 0.0253, 0.0561, 0.0226, 0.0247, 0.0265, 0.0106, 0.0144,
                                         0.0217, 0.0104, 0.0122, 0.0222, 0.0103, 0.0122, 0.0305, 0.0119, 0.0158,
                                         0.0262, 0.0164, 0.0174, 0.0270, 0.0159, 0.0166, 0.0441, 0.0258, 0.0281,
                                         0.0458, 0.0256, 0.0272, 0.0648, 0.0307, 0.0358, 0.0697, 0.0295, 0.0334,
                                         0.0757, 0.0336, 0.0406, 0.0817, 0.0325, 0.0375, 0.0206, 0.0146, 0.0150,
                                         0.0216, 0.0148, 0.0148, 0.0113, 0.0114, 0.0105, 0.0129, 0.0132, 0.0087,
                                         0.0132, 0.0129, 0.0081, 0.0177, 0.0159, 0.0144, 0.0186, 0.0153, 0.0141,
                                         0.0161, 0.0123, 0.0104, 0.0153, 0.0116, 0.0101, 0.0596, 0.0283, 0.0295,
                                         0.0604, 0.0263, 0.0287, 0.0851, 0.0410, 0.0486, 0.0304, 0.0172, 0.0191,
                                         0.0311, 0.0198, 0.0218, 0.0339, 0.0178, 0.0197, 0.0337, 0.0204, 0.0223,
                                         0.0912, 0.0416, 0.0459, 0.0392, 0.0216, 0.0291], dtype=np.float32) * 2.0
        self.joint_noise = tf.constant(joint_gaussian_noise, name='joint_noise')

        self._read_onthefly(kintree, tf.contrib.util.constant_value(bone_template))
        
        # Convert lists to numpy arrays
        self.data_size = len(self.data[BONES])
        for opt in [BONES, NORMALS, POINTS3D, SHAPE, POSE, 'idx']:
            self.data[opt] = np.array(self.data[opt], dtype=np.float32)
        for opt in self.opts:
            self.labels[opt] = np.array(self.labels[opt], dtype=np.float32)
    
    def _read_onthefly(self, kintree, bone_template):
        '''
        Reads the txt_file line by line and load data.
        '''
        def compute_relative_joint_norm(P, kintree):
            # Compute normalized relative joints including landmarks with respect to kinematic tree
            P_normalized = []
            for idx in range(1, len(kintree)):
                father_idx, joint_idx = kintree[idx]
                relative_P = P[joint_idx, :] - P[father_idx, :]
                P_normalized.append( relative_P / np.sqrt(np.sum(relative_P**2)) )
            return np.reshape(P_normalized, -1)

        def compute_bone(P, kintree, bone_template):
            # Compute bone length including landmarks with respect to kinematic tree
            bone = []
            for idx in range(1, len(kintree)):#
                father_idx, joint_idx = kintree[idx]
                bone.append( np.sqrt(np.sum((P[father_idx, :] - P[joint_idx, :])**2)) - bone_template[idx] )
            return np.reshape(bone, -1)
            
        def zRotMatrix(zrot):
            c, s = cos(zrot), sin(zrot)
            return np.array([[c, -s, 0],
                             [s,  c, 0],
                             [0,  0, 1]], np.float32)
        
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:

                line = line.replace('\n', '').replace('\r', '') # seq/frame.png
                
                base_name, frame = line.rsplit('/',1)
                base_name = os.path.join(self.root, base_name)
                frame = int(frame.rsplit('.',1)[0])
                
                # Read global info
                info = sio.loadmat(os.path.join(base_name, 'info.mat'))
                zRot = zRotMatrix(info['zrot'])
                
                # Read body info
                info_body = sio.loadmat(os.path.join(base_name, 'info_body.mat'))
                j3d = info_body['joints'][frame, :, :]
                l3d = info_body['lmarks'][frame, :, :]
                p3d = np.concatenate((j3d, l3d), axis=0)
                p3d -= p3d[0, :].reshape((1, 3)) # Align at (0, 0, 0)
                
                p3d = p3d.dot(zRot.T)
                p3d_norm = compute_relative_joint_norm(p3d, kintree)
                bone = compute_bone(p3d, kintree, bone_template)
                
                self.data[NORMALS].append(p3d_norm)
                self.data[POINTS3D].append(np.reshape(p3d[1:], -1))                    
                self.data[BONES].append(bone)
                
                self.labels[POINTS3D].append(np.reshape(p3d[1:], -1))
                self.labels[NORMALS].append(p3d_norm)
                self.labels[BONES].append(bone)

                # Read rotation matrices given pose parameters
                # Rotation matrices can be computed with Rodrigues formulation given axis-angles
                R = info_body['rotations'][frame, :, :, :]
                R[0] = zRot.dot(R[0])
                R = R.reshape((-1))
                self.labels[POSE].append(R)
                
                # Read shape parameters
                shape = np.concatenate([info[SHAPE], 2 * (info[GENDER] - 0.5)], -1).astype(np.float32).reshape((-1))
                self.labels[SHAPE].append(shape)
                self.data[SHAPE].append(shape)

    def rodriques(self, r):
        theta = np.sqrt(np.sum(r**2))
        theta = np.maximum(theta, np.finfo(np.float32).eps)
        
        r_hat = r / theta
        m = np.array([[0, -r_hat[2], r_hat[1]], [ r_hat[2], 0, -r_hat[0]], [ -r_hat[1], r_hat[0], 0]], np.float32)
        R = np.identity(3, np.float32) + m * np.sin(theta) + np.matmul(m, m) * (1 - np.cos(theta))

        return R

    def _parse_function_train(self, data, label):
        """ data augmentation for tensorflow """
        data[POINTS3D] += tf.random_uniform(data[POINTS3D].shape, -1., 1.) * self.joint_noise
        return data, label

    def _parse_function_inference(self, data, label):
        return data, label
