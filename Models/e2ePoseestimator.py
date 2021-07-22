import sys
from Models.common import *

import tensorflow as tf
import numpy as np
import pickle
from time import time as t
import scipy.io as sio
import os

# SMPL vertex indices of selected landmarks
landmarks = np.array([3042, # Breast left
            4428, # Breast right
            3503, # Belly
            4311, # Right hip
            822, # Left hip
            4361, # Right upper leg (front)
            877, # Left upper leg (front)
            4386, # Right upper leg (back)
            898, # Left upper leg (back)
            3222, # Left foot
            6620, # Right foot
            2544, # Left hand
            2820, # Left upper arm (biceps)
            1258, # Left upper arm (triceps)
            4116, # Right upper arm (biceps)
            5365, # Right upper arm (triceps)
            5989, # Right hand
            331], np.int32); # Top of head

# Grouping joints (out of 24 SMPL joints) for evaluation
# Left-Right
eval_joints = {
        'All': [1, 2, 3, 4, 5, 7, 8, 9, 12, 15, 16, 17, 18, 19, 20, 21],
        'Head': [12, 15], # Neck, Head
        'Torso': [3, 6, 9], # Back joints
        'Shoulder': [13, 14, 16, 17], # Inner, Outer 
        'Elbow': [18, 19],
        'Wrist': [20, 21],
        'Hip': [1, 2],
        'Knee': [4, 5],
        'Foot': [7, 8, 10, 11]} # Ankle, Foot

class SMPLREstimator:
    def __init__(self, joints, keep_prob, is_training, type='', shape=None, pose=None, kintree=None, bone_template=None):
        '''
        SMPL pose and shape estimator class, including DAE and SMPLR.
        Input:
            joints: flattened joints and landmarks (batch_size, 41x3)
            keep_prob: dropout keeping probability
            is_training: boolean flag for training/inference mode
            type: network type 
            shape: ground truth shape parameters if needed (batch_size, 11)
            pose: ground truth pose rotation matrices if needed (batch_size, 24, 3, 3)
            kintree: joints and landmarks kinematic tree
            bone_template: relative distances of joints and landmarks w.r.t. the kintree for the template body
        Output:
            joints_x: reconstructed joints in DAE (if applied)
            shape_x: estimated shape parameters (batch_size, 11)
            pose_x: estimated pose parameters (as flattened rotation matrices) (batch_size, 24x3x3)
            J: reconstructed SMPL joints (batch_size, 24, 3)
            T: reconstructed SMPL body vertices (batch_size, 6890, 3)
        '''
        self.joints = joints
        self.joints_x = joints
        self.shape_x = shape
        self.pose_x = pose
        self.KEEP_PROB = keep_prob
        self.is_training = is_training
        self.kintree = kintree
        self.bone_template = bone_template

        self.T = tf.zeros([])
        self.J = tf.zeros([])
        
        if type != 'dae' and type != 'shape' and type != 'pose' and type != 'shapepose' and type != 'all':
            print("'type' is not known. It must take a value from the list ['dae', 'shape', 'pose', 'shapepose', 'all'].")
            exit()
        
        if type == 'dae' or type == 'all':
            self.create_dae()
        
        if type != 'dae':
            # Build SMPL model
            self.smpl = SMPL()
            
            # Convert joints and landmarks to relative distances and normals
            self.get_features()
        
        if type == 'shape' or type == 'shapepose' or type == 'all':
            # Create shape estimator network
            self.create_shape()
        if type == 'pose' or type == 'shapepose' or type == 'all':
            # Create pose estimator network
            self.create_pose()
        if type != 'dae':
            if type == 'shape' or type == 'pose':
                # Apply SMPL operations only during validation
                self.T, self.J = tf.cond(self.is_training, true_fn=lambda: (np.zeros(0, np.float32), np.zeros(0, np.float32)), false_fn=self.run_smpl)
            else:
                self.T, self.J = self.run_smpl()

    def get_features(self):
        # compute relative normal joints from the output of joints reconstruction
        z = tf.constant(np.zeros((self.joints_x.get_shape()[0], 1, 3)), tf.float32)
        x1 = tf.concat([z, tf.reshape(self.joints_x, (-1, 41, 3))], 1)
    
        father_idx, joint_idx = np.transpose(self.kintree[1:])
        r1 = tf.subtract(tf.gather(x1, joint_idx, axis=1), tf.gather(x1, father_idx, axis=1))
        l1 = tf.sqrt(tf.reduce_sum(tf.square(r1), -1, keepdims=True))
        self.bones = tf.squeeze(tf.subtract(l1, self.bone_template[1:]), -1)
        self.normals = tf.contrib.layers.flatten(tf.divide(r1, l1))
        
    def run_smpl(self):
        return self.smpl.run(tf.reshape(self.pose_x, (-1, 24, 3, 3)), self.shape_x)
        
    def create_dae(self):
        '''DAE network. AutoEncoder model with: part dropout + bn + skip connection'''
        with tf.variable_scope('dae') as scope:
            drop = dropout(self.joints, self.KEEP_PROB, name='enc/drop_0')
            fc_e1 = fc(drop, 41*3, 1024, name='enc/fce1_0', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e2 = fc(fc_e1, 1024, 1024, name='enc/fce2_0', batch_norm=True, activation='elu', is_training=self.is_training)
            middle = fc(fc_e2, 1024, 512, name='enc/fce3_0', batch_norm=True, activation='elu', is_training=self.is_training)
            drop = dropout(middle, self.KEEP_PROB, name='enc/drop_1')
            fc_d2 = fc(drop, 512, 1024, name='dec/fcd2_0', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_d2 = tf.add_n([fc_d2, fc_e2], name='dec/skip0_0')
        
            fc_d1 = fc(fc_d2, 1024, 1024, name='dec/fcd1_0', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_d1 = tf.add_n([fc_d1, fc_e1], name='dec/skip1_0')
        
            self.joints_x = self.joints + fc(fc_d1, 1024, 41*3, name='dec/x_0', is_training=self.is_training)

    def create_shape(self):
        '''This is shape estimator model'''
        
        with tf.variable_scope('shape') as scope:
            drop = self.bones # dropout(self.bones, self.KEEP_PROB, name='enc/drop_0')# 
            
            fc_e1 = fc(drop, 41, 1024, name='enc/fce1', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e2 = fc(fc_e1, 1024, 1024, name='enc/fce2', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e3 = fc(fc_e2, 1024, 512, name='enc/fce3', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e4 = fc(fc_e3, 512, 512, name='enc/fce4', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e41 = fc(fc_e4, 512, 512, name='enc/fce41', batch_norm=True, activation='elu', is_training=self.is_training)
            
            fc_e42 = fc(fc_e41, 512, 256, name='enc/fce42', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e5 = fc(fc_e42, 256, 256, name='enc/fce5', batch_norm=True, activation='elu', is_training=self.is_training)
            self.shape_x = fc(fc_e5, 256, 11, name='enc/x', batch_norm=False, activation='', is_training=self.is_training)
            
            fc_e43 = fc(fc_e41, 512, 256, name='enc/fce43', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e44 = fc(fc_e43, 256, 11, name='enc/x_offset', batch_norm=False, activation='', is_training=self.is_training)
            self.shape_x = fc(tf.concat([self.shape_x, fc_e44], -1), 22, 11, name='enc/x2', is_training=self.is_training) # tf.nn.tanh(self.shape_x1 + fc_e44, name='enc/x2') * 5.0
        
    def create_pose(self):
        '''This is pose estimator model'''
        
        with tf.variable_scope('pose') as scope:
            drop = self.normals # dropout(self.normals, self.KEEP_PROB, name='enc/drop_0')
            
            fc_e1 = fc(drop, 41*3, 1024, name='enc/fce1', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e2 = fc(fc_e1, 1024, 1024, name='enc/fce2', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e3 = fc(fc_e2, 1024, 512, name='enc/fce3', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e4 = fc(fc_e3, 512, 512, name='enc/fce4', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e41 = fc(fc_e4, 512, 512, name='enc/fce41', batch_norm=True, activation='elu', is_training=self.is_training)
            
            fc_e42 = fc(fc_e41, 512, 256, name='enc/fce42', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e5 = fc(fc_e42, 256, 256, name='enc/fce5', batch_norm=True, activation='elu', is_training=self.is_training)
            self.pose_x = fc(fc_e5, 256, 24*9, name='enc/x', batch_norm=False, activation='', is_training=self.is_training)
            
            fc_e43 = fc(fc_e41, 512, 256, name='enc/fce43', batch_norm=True, activation='elu', is_training=self.is_training)
            fc_e44 = fc(fc_e43, 256, 24*9, name='enc/x_offset', batch_norm=False, activation='', is_training=self.is_training)
            self.pose_x = fc(tf.concat([self.pose_x, fc_e44], -1), 2*24*9, 24*9, name='enc/x2', activation='tanh', is_training=self.is_training) # tf.nn.tanh(self.pose_x1 + fc_e44, name='enc/x2')
            
            

class SMPL:

    def __init__(self):
        self.J_regressor = []
        self.v_template = []
        self.shapedirs = []
        self.posedirs = []
        self.weights = []
        # Load SMPL models
        with open('SMPL/model_f.pkl', 'rb') as f:
            if sys.version_info[0]<3:
                female = pickle.load(f)
            else:
                female = pickle.load(f, encoding="latin1")
            self.J_regressor.append(tf.constant(np.array(female['J_regressor'].todense(), dtype=np.float32)))
            self.v_template.append(tf.constant(female['v_template'], dtype=np.float32))
            self.posedirs.append(tf.constant(female['posedirs'], dtype=np.float32))
            self.shapedirs.append(tf.constant(female['shapedirs'], dtype=np.float32))
            self.weights.append(tf.constant(female['weights'], dtype=np.float32))
        with open('SMPL/model_m.pkl', 'rb') as f:
            if sys.version_info[0]<3:
                male = pickle.load(f)
            else:
                male = pickle.load(f, encoding="latin1")
            self.J_regressor.append(tf.constant(np.array(male['J_regressor'].todense(), dtype=np.float32)))
            self.v_template.append(tf.constant(male['v_template'], dtype=np.float32))
            self.posedirs.append(tf.constant(male['posedirs'], dtype=np.float32))
            self.shapedirs.append(tf.constant(male['shapedirs'], dtype=np.float32))
            self.weights.append(tf.constant(male['weights'], dtype=np.float32))
            kintree_table = male['kintree_table']
            id_to_col = {kintree_table[1, i]: i for i in range(kintree_table.shape[1])}
            self.parent = {i: id_to_col[kintree_table[0, i]] for i in range(1, kintree_table.shape[1])}
        self.J_regressor = tf.stack(self.J_regressor, axis=0)
        self.v_template = tf.stack(self.v_template, axis=0)
        self.shapedirs = tf.stack(self.shapedirs, axis=0)
        self.posedirs = tf.stack(self.posedirs, axis=0)
        self.weights = tf.stack(self.weights, axis=0)
    
    def rodrigues(self, r):
        theta = tf.norm(r, axis=(1, 2), keepdims=True)
        theta = tf.maximum(theta, np.finfo(np.float32).eps)
        r_hat = r / theta
        z_stick = tf.zeros(theta.get_shape().as_list()[0], dtype=tf.float32)
        m = tf.stack(
            (z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
            -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick), axis=1)
        m = tf.reshape(m, (-1, 3, 3))
        i_cube = tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0) + tf.zeros(
            (theta.get_shape().as_list()[0], 3, 3), dtype=tf.float32)
        #R = cos * i_cube + (1 - cos) * dot + tf.sin(theta) * m
        R = i_cube + m * tf.sin(theta) + tf.matmul(m, m) * (1 - tf.cos(theta))
        return R

    def with_zeros(self, x):
        return tf.concat((x, tf.constant([[0.0, 0.0, 0.0, 1.0]], dtype=tf.float32)), axis=0)

    def pack(self, x):
        return tf.concat((tf.zeros((x.get_shape().as_list()[0], 4, 3), dtype=tf.float32), x), axis=2)
        
    def run_rodrigues(self, pose):
        ''' Apply Rodrigues in the batch'''
        batch_size = pose.get_shape()[0]
        R_out = []
        for i in range(batch_size):
            R_out.append(self.rodrigues(tf.reshape(pose[i,:], (-1, 1, 3))))
        return tf.stack(R_out, axis=0)

    def run(self, R, betas):
        '''
        SMPL engine to shape and pose template body
        NOTE: THIS IMPLEMENTATION IS RECURSIVE AND NOT EFFICIENT
        Input:
            R: per joint rotation matrices (batch_size, 24, 3, 3)
            betas: shape parameters including gender (-1: female, 1: male) 
        Output:
            J: posed body joints
            T: posed body surface vertices
        '''
        batch_size = betas.get_shape()[0]
        
        # Update template body w.r.t. shape parameters
        v_shaped = []
        v_shaped.append(tf.tensordot(betas[:,:10], self.shapedirs[0], axes=[[1], [2]]) + tf.reshape(self.v_template[0], shape=(1,6890,3))) # female
        v_shaped.append(tf.tensordot(betas[:,:10], self.shapedirs[1], axes=[[1], [2]]) + tf.reshape(self.v_template[1], shape=(1,6890,3))) # male
        v_shaped = tf.stack(v_shaped, axis=-1)

        def single_run(x, J_regressor=self.J_regressor, posedirs=self.posedirs, weights=self.weights, parent=self.parent):
            ''' Skinning and joint regression for one single frame'''
            R = x[0]
            v_shaped = x[1]
            g = x[2]

            J = tf.matmul(J_regressor[g], v_shaped[:, :, g])
            R_cube_big = R
            R_cube = R_cube_big[1:]
            I_cube = tf.expand_dims(tf.eye(3, dtype=tf.float32), axis=0) + tf.zeros((R_cube.get_shape()[0], 3, 3), dtype=tf.float32)
            lrotmin = tf.squeeze(tf.reshape((R_cube - I_cube), (-1, 1)))
            v_posed = v_shaped[:, :, g] + tf.tensordot(posedirs[g], lrotmin, axes=[[2], [0]])
            results = []
            results.append(self.with_zeros(tf.concat((R_cube_big[0], tf.reshape(J[0, :], (3, 1))), axis=1)))
            for j in range(1, 24):
                results.append(tf.matmul(results[parent[j]], self.with_zeros(tf.concat((R_cube_big[j], tf.reshape(J[j, :] - J[parent[j], :], (3, 1))), axis=1))))
            stacked = tf.stack(results, axis=0)
            results = stacked - self.pack(tf.matmul(stacked, tf.reshape(tf.concat((J, tf.zeros((24, 1), dtype=tf.float32)), axis=1), (24, 4, 1))))
            
            J = tf.matmul(results, tf.expand_dims(tf.concat([J, tf.ones([24, 1], tf.float32)], axis=1), axis=1), transpose_b=True)
            J = tf.gather(J, tf.constant([0, 1, 2], tf.int32), axis=1)
            
            T = tf.tensordot(weights[g], results, axes=((1), (0)))
            rest_shape_h = tf.concat((v_posed, tf.ones((v_posed.get_shape().as_list()[0], 1), dtype=tf.float32)), axis=1)
            v = tf.matmul(T, tf.reshape(rest_shape_h, (-1, 4, 1)))
            v = tf.reshape(v, (-1, 4))[:, :3]

            return v, tf.squeeze(J)

        gender = tf.cast(tf.greater_equal(betas[:, 10] , 0), tf.int32)
        
        # Parallelize SMPL over samples in the batch
        T_out, J_out = tf.map_fn(single_run, (R, v_shaped, gender), (tf.float32, tf.float32))
        
        # Make sure the root joint has [0,0,0] coordinate
        J0 = tf.reshape(J_out[:,0,:], (-1, 1, 3))
        return tf.subtract(T_out, J0), tf.subtract(J_out, J0)


def compute_bone(P):
    # Compute bone length including landmarks with respect to kinematic tree
    bones = []
    for p in P:
        p = np.concatenate([np.zeros((1,3)), p])
        bone = []
        for idx in range(1, len(kintree)):#
            father_idx, joint_idx = kintree[idx]
            bone.append( np.sqrt(np.sum((p[father_idx, :] - p[joint_idx, :])**2)) - bone_template[idx] )

        bones.append(np.reshape(bone, -1))
    return np.array(bones)