'''
Created by Hugo Bertiche in Oct. 2018
'''


import sys
import os
import time
from datetime import datetime

import scipy.io as sio
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf
import numpy as np

# PROCRUSTES
#from util.procrustes import procrustes

# Import datagenerator
from Data.datagenerator_cloth3d import *

# Import hourglass model
from Models.StackedHourglass.hourglassModelVoxelsBones import HourglassModel

# Import loss
from Losses.losses import VoxelLoss

# SMPL
#from SMPL.smpl import *

'''################### set arguments ###################'''
# Args
if len(sys.argv) < 4:
    print("Missing args")
    print("1st arg: model name (string)")
    print("2nd arg: GPU (integer)")
    print("3rd arg: execution mode ('train','validate','test')")
    print("4th arg (optional): augmentation type (0:no aug, 1 (default): standard aug, 2: synthetic occlusion aug, 3: 1+2 aug)")
    print("5th arg (optional): initial learning rate (float, default=0.01)")
    print("6th arg (optional): model name for finetuning (string)")
    sys.exit()

# Model name for save file
name = sys.argv[1]

# GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

# Is test
mode = sys.argv[3]
test = mode == 'test'
validate = mode == 'validate'

# Augmntation
augmentation = 1
if len(sys.argv) > 4:
    augmentation = float(sys.argv[4])

# Set initial learning rate
learning_rate = 0.01
if len(sys.argv) > 5:
    learning_rate = float(sys.argv[5])

# Finetune
finetune = ''
if len(sys.argv) > 6:
    finetune = sys.argv[6]
if validate and not finetune:
    print("CANNOT VALIDATE A MODEL IF NAME IS NOT PROVIDED")
    sys.exit()

# Learning and Network parameters
num_stacks = 5
batch_size = 6
num_epochs = 200
num_points = 41
num_bones = 4
voxel_depth = 16
if validate:
    learning_rate = 0
    num_epochs = 1
    batch_size = 32
    augmentation = 0
if test:
    learning_rate = 0
    num_epochs = 1
    batch_size = 1
    augmentation = 0
    print("TEST MODE")

# Verbose
print("NETWORK PARAMETERS")
print("\tNumber of points: " + str(num_points))
print("\tVoxel depth: " + str(voxel_depth))
print("\tNumber of stacks: " + str(num_stacks))
print("\tLearning rate: " + str(learning_rate))
if not learning_rate:
    print("\t\033[93mNO LEARNING!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\033[0m")
print("\tBatch size: " + str(batch_size))
print("")
print("\tDATA AUGMENTATION: " + str(augmentation))
print("")

# PATHS FOR CHECKPOINTS AND RESULTS
root = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(root, "tmp", "checkpoints")
results_path = os.path.join(root, "tmp", "results", finetune)
print("Checkpoint shall be saved at: " + checkpoint_path)
if not os.path.isdir(checkpoint_path):
    print("CREATING CHECKPOINT PATH...")
    os.makedirs(checkpoint_path)
if validate and not os.path.isdir(results_path):
    print("CREATING RESULTS PATH...")
    os.makedirs(results_path)

# Path to the textfiles for the trainings and validation set
print("SETTING TRAIN/VAL PATHS...")
if not test and not validate:
    train_file = os.path.join(root, 'Data', 'train.txt')
    val_file = os.path.join(root, 'Data', 'val.txt')
elif validate:
    train_file = ''
    val_file = ''
else:
    train_file = ''
    val_file = ''

'''################### Build data reader ###################'''
# DATA
print("Loading training data...")
with tf.device('/cpu:0'):
    tr_data = ImageDataGenerator(train_file,
                                mode='training',
                                batch_size=batch_size,
                                root=root+'/Data/frames',
                                shuffle=True,
                                augmentation=augmentation,
                                validate=not test,
                                GT_crop=True)
    print("Loading validation data...")
    val_data = ImageDataGenerator(val_file,
                                mode='inference',
                                batch_size=batch_size,
                                root=root+'/Data/frames',
                                shuffle=False,
                                validate=not test,
                                GT_crop=True)

iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
next_batch = iterator.get_next()

tr_init_op = iterator.make_initializer(tr_data.data)
val_init_op = iterator.make_initializer(val_data.data)

# Number of batches
train_batches = int(np.floor(tr_data.data_size / batch_size))
val_batches = int(np.floor(val_data.data_size / batch_size))

'''################### build model ###################'''
# TF placeholders
print("Creating placeholders...")
x = tf.placeholder(tf.float32, [batch_size, 256, 256, 3]) # Input image
y = tf.placeholder(tf.float32, [batch_size, num_points, 3]) # 3D Points

is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)

# Initialize model
print("Initializing models...")
model = HourglassModel(x, outDim=num_points, outBones=num_bones, nbStacks=num_stacks, voxelDepth=voxel_depth, training=is_training)
kintree = np.array(sio.loadmat(os.path.join(root, "SMPL", "kintree.mat"))['kintree'][:, 1:] - 1, np.int32).tolist()

# Link variable to model output
print("Linking model output...")
scores = model.out

'''################### build loss ###################'''
# OP for the loss
print("Setting losses...")
with tf.name_scope("loss"):
    voxelLoss = VoxelLoss(depth=voxel_depth)
    loss_stack = voxelLoss.run_with_bones(scores, y, kintree)
    loss = tf.reduce_sum(loss_stack)
    
    # Joints 3D coords extraction
    final = scores[num_stacks - 1] # Nx64x64x16x(41 + 4)
    voxels = []
    for i in range(num_points):
        heatmap = final[:, :, :, :, i]
        heatmap = tf.reshape(heatmap, (batch_size, -1, 1))
        
        heatmap = tf.nn.softmax(heatmap, axis=1)
        xxyyzz = tf.reduce_sum(tf.multiply(heatmap, voxelLoss._xxyyzz), axis=1)

        voxels.append(xxyyzz)
    voxels = tf.cast(tf.stack(voxels, axis=1), tf.float32) # N x num_points x 3
    p3d = voxelLoss._voxels_to_points(voxels) # N x num_points x 3
    err = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(p3d, y)), axis=2)), axis=0) # num_points
    
    # Bones (n_bones = num_points)
    # Score bones
    ext_zeros = tf.constant(np.zeros((batch_size, 1, 3), np.float32))
    ext_p3d = tf.concat((ext_zeros, p3d), axis=1) 
    bone_heads = tf.gather(ext_p3d, kintree[0], axis=1) # N x n_bones x 3
    bone_ends = tf.gather(ext_p3d, kintree[1], axis=1) # N x n_bones x 3
    bones = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(bone_heads, bone_ends)), axis=2)) # N x n_bones
    # GT bones
    ext_y = tf.concat((ext_zeros, y), axis=1)
    y_bone_heads = tf.gather(ext_y, kintree[0], axis=1) # N x n_bones x 3
    y_bone_ends = tf.gather(ext_y, kintree[1], axis=1) # N x n_bones x 3
    y_bones = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(y_bone_heads, y_bone_ends)), axis=2)) # N x n_bones
    # Bone error
    bone_err = tf.sqrt(tf.square(tf.subtract(bones, y_bones)))[:,:num_points] # N x n_bones
    bone_err_mean, bone_err_var = tf.nn.moments(bone_err, axes=[0])

'''################### build training OP ###################'''
# Train OP
print("Training OP...")
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)

# Saver/Loader
saver = tf.train.Saver(var_list=tf.global_variables())

'''################### start session ###################'''
# Start TF session
with tf.Session() as sess:
    # Initialize variable
    sess.run(tf.global_variables_initializer())
    
    if finetune and os.path.exists(os.path.join(checkpoint_path, finetune + '.ckpt.index')):
        print("Restoring model: " + finetune)
        saver.restore(sess, os.path.join(checkpoint_path, finetune + '.ckpt'))
    else:
        if validate:
            print("NO MODEL TO VALIDATE")
            sys.exit()
        print("Training from scratch")
    
    print("Start training...")
    
    best = float('inf')
    
    # Loop over epochs
    for epoch in range(num_epochs):
        start = time.time()
        print("")
        print("")
        
        if not validate and not test:
            print("EPOCH NUMBER: " + str(epoch))
            print("-------------------------------------------")
            
            # Initialize iterator
            print("")
            print("TRAINING")
            sess.run(tr_init_op)
            
            loss_ = 0
            loss_stack_ = np.zeros(num_stacks)
            err_ = np.zeros(num_points)
            count = 0
            for step in range(np.minimum(500*(epoch+1), train_batches)):
                img_batch, _, label_batch, pose_batch, shape_batch, _ = sess.run(next_batch)

                ops = [train_op, loss, loss_stack, err]
                feed_dict = {x: img_batch,
                            y: label_batch[:, :num_points],
                            is_training: True,
                            lr: learning_rate*(1.0-1.0/(1+np.exp(10-epoch/12.5)))+0.0001}
                            
                _, _loss, _loss_stack, _err = sess.run(ops, feed_dict=feed_dict)
            
                # Compute cumulative sum
                loss_ += _loss
                loss_stack_ += _loss_stack
                err_ += _err
                count += 1
                if step%1000 == 0:
                    sys.stdout.write("\rStep: " + str(step + 1) + "/" + str(train_batches) + "... Loss: " + str(_loss) + " - Error: " + str(_err.mean()))
                    sys.stdout.flush()
                #if step == 25000:
                #    break

            loss_ /= count
            loss_stack_ /= count
            err_ /= count
            print("")
            print("Loss: " + str(loss_))
            print(loss_stack_)
            print("Error: " + str(np.mean(err_)))
            if num_points > 23:
                print("Error Joints: " + str(np.mean(err_[:23])))
                print("Error landmarks: " + str(np.mean(err_[23:])))
            # print(err_)
            # print("Error Bone: " + str(np.mean(bone_err_mean_)))
            # print(bone_err_mean_)
            # print("Error Bone Variance: " + str(np.mean(bone_err_var_)))
            # print(bone_err_var_)
        
        if test: 
            print("test mode has not been implemented.")
            continue
        
        print("")
        print("VALIDATION")
        sess.run(val_init_op)
        
        val_loss_ = 0
        val_loss_stack_ = np.zeros(num_stacks)
        val_err_ = np.zeros(num_points)
        val_bone_err_ = np.zeros(num_points)
        val_count = 0
        for val_step in range(val_batches):
            img_batch, _, label_batch, pose_batch, shape_batch, path_batch = sess.run(next_batch)

            feed_dict = {x: img_batch,
                        y: label_batch[:, :num_points],
                        is_training: False,
                        lr: 0}
                        
            opts = [loss, loss_stack, err, bone_err]
            _loss, _loss_stack, _err, _bone_err = sess.run(opts, feed_dict=feed_dict)
            
            # if validate:
                # pred = sess.run(p3d, feed_dict=feed_dict)
                # sio.savemat(results_path + '/val/batch' + str(val_step), {'pred':pred,
                                                                    # 'label': label_batch[:, :num_points],
                                                                    # 'pose': pose_batch,
                                                                    # 'shape': shape_batch,
                                                                    # 'path': path_batch})
            # Compute cumulative sum
            val_loss_ += _loss
            val_loss_stack_ += _loss_stack
            val_err_ += _err
            val_bone_err_ += np.mean(_bone_err, 0)
            val_count += 1
            if val_step%1000 == 0:
                sys.stdout.write("\rStep: " + str(val_step + 1) + "/" + str(val_batches) + ' Error: ' + str(val_err_.mean() / val_count))
                sys.stdout.flush()
        
        val_loss_ /= val_count
        val_loss_stack_ /= val_count
        val_err_ /= val_count
        val_bone_err_ /= val_count
        print("")
        print("Loss: " + str(val_loss_))
        print(val_loss_stack_)
        print("Error: " + str(np.mean(val_err_)))
        if num_points > 23:
            print("Error Joints: " + str(np.mean(val_err_[:23])))
            print("Error Landmarks: " + str(np.mean(val_err_[23:])))
            print("Error Bones: " + str(np.mean(val_bone_err_)))
        print("")
        
        if validate:
            sys.exit()
            
        if val_err_.mean() < best:
            best = val_err_.mean()
            print("Saving model checkpoint...")
                
            checkpoint_name = os.path.join(checkpoint_path, name + '.ckpt')
            save_path = saver.save(sess, checkpoint_name)
            print("Model saved at: " + checkpoint_name)

        print("")
        print("ELAPSED: {:.2f}".format(time.time() - start))
        print("BEST: " + str(best))
