import sys
import os
import time
from datetime import datetime
import scipy.io as sio
import numpy as np
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import tensorflow as tf

# Import datagenerator
from Data.datagenerator_cloth3d import *

# Import ShapeEstimator model
from Models.e2ePoseestimator import *

# Import Euclidean distance function
from Models.common import get_euc_err

'''################### set arguments ###################'''
# Args
if len(sys.argv) < 4:
    print("Missing args")
    print("1st arg: model name (string)")
    print("2nd arg: GPU (integer)")
    print("3rd arg: model type ('dae', 'shape', 'pose', 'shapepose')")
    print("4th arg (optional): is there augmentation? (True (default), False)")
    print("5th arg (optional): initial learning rate (float, default=0.01)")
    print("6th arg (optional): model name for finetuning (string)")
    print("7th arg (optional): 2nd model name for finetuning (string)")
    sys.exit()

# Model name for save file
name = sys.argv[1]
# Is test
test = name == 'test'

# GPU device
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[2]

# Model type to select the network
model_type = sys.argv[3]
if model_type != 'dae' and model_type != 'shape' and model_type != 'pose' and model_type != 'shapepose':
    print("'model_type' is not known. It must take a value from the list ['dae', 'shape', 'pose', 'shapepose'].")
    exit()

# Define whether training with augmentation or not
is_augmentation = True
if len(sys.argv) > 4:
    if sys.argv[4] != 'True' and sys.argv[4] != 'False':
        print("'augmentation' value does not exist. It must take a value from list ['True', 'False'].")
        print("We set 'is_augmentation = True'")
        is_augmentation = True
    else:
        is_augmentation = sys.argv[4] == 'True'

# Set initial learning rate
learning_rate = 0.01
if len(sys.argv) > 5:
    learning_rate = float(sys.argv[5])

# Finetune
finetune = ''
if len(sys.argv) > 6:
    finetune = sys.argv[6]
finetune_pose = ''
if model_type == 'shapepose' and len(sys.argv) > 7:
    finetune_pose = sys.argv[7]

# Learning and Network parameters
num_epochs = 200
if finetune and model_type == 'shapepose':
    num_epochs = 30
batch_size = 256
num_points = 41
dropout = 0.8
if test:
    batch_size = 10
is_save = True

# Verbose
print("NETWORK PARAMETERS")
print("\tLearning rate: " + str(learning_rate))
print("\tDropout: " + str(dropout))
print("\tBatch size: " + str(batch_size))
if test:
    print("TEST MODE")
# PATHS FOR CHECPOINTS AND RESULTS
root = os.path.dirname(os.path.abspath(__file__))
checkpoint_path = os.path.join(root, "tmp", "checkpoints")
results_path = os.path.join(root, "tmp", "results", finetune)
print("Checkpoint shall be saved at: " + checkpoint_path)
if not os.path.isdir(checkpoint_path):
    print("CREATING CHECKPOINT PATH...")
    os.makedirs(checkpoint_path)

# Path to the textfiles for the trainings and validation set
print("SETTING TRAIN/VAL PATHS...")
if not test:
    train_file = os.path.join(root, 'Data', 'train.txt')
    val_file = os.path.join(root, 'Data', 'val.txt')
else:
    train_file = ''
    val_file = ''


'''################### Build data reader ###################'''
# DATA
print("Loading training data...")
opts = [SHAPE, BONES, POSE, POINTS3D, NORMALS]
with tf.device('/cpu:0'):
    tr_data = SMPLRDataGenerator(train_file,
                                mode='training',
                                batch_size=batch_size,
                                root=root+'/Data/frames',
                                shuffle=False,
                                opts=opts)
    print("Loading validation data...")
    val_data = SMPLRDataGenerator(val_file,
                                mode='training',
                                batch_size=batch_size,
                                root=root+'/Data/frames',
                                shuffle=False,
                                opts=opts)

# Number of batches
train_batches = np.minimum(400, int(np.floor(tr_data.data_size / batch_size)))
val_batches = int(np.floor(val_data.data_size / batch_size))

'''################### build model ###################'''
# TF placeholders
print("Creating placeholders...")
joints = tf.placeholder(tf.float32, [batch_size, num_points*3]) # Input 3D joints and landmarks
y_joints = tf.placeholder(tf.float32, [batch_size, num_points*3]) # gt 3D joints and landmarks
y_pose = tf.placeholder(tf.float32, [batch_size, 24*9]) # gt rotation matrices
y_shape = tf.placeholder(tf.float32, [batch_size, 11]) # gt shape parameters

keep_prob = tf.placeholder(tf.float32)
is_training = tf.placeholder(tf.bool)
lr = tf.placeholder(tf.float32)

kintree = np.transpose(np.array(sio.loadmat('SMPL/kintree.mat')['kintree'], np.int32)) - 1

# Initialize model
print("Initializing models...")
joints_noise = tf.random_uniform(joints.shape, -0.5, 0.5) * tr_data.joint_noise
model = SMPLREstimator(joints + joints_noise if is_augmentation else joints, keep_prob=keep_prob, is_training=is_training, type=model_type, shape=y_shape, pose=y_pose, kintree=kintree, bone_template=bone_template)

# Link variable to model output
print("Linking model output...")
joints_x = model.joints_x
shape_x = model.shape_x
pose_x = model.pose_x
smpl_joints = model.J
smpl_mesh = model.T


'''################### build loss ###################'''
# OP for the loss
print("Setting losses...")
with tf.name_scope("loss"):
    if model_type == 'dae':
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_joints, joints_x)), name='dae')
    if model_type == 'shape':
        loss_shape_reg = tf.reduce_mean(tf.abs(shape_x), name='shape-reg')
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_shape, shape_x)), name='shape') + 0.01 * loss_shape_reg
    if model_type == 'pose':
        loss = tf.reduce_mean(tf.abs(tf.subtract(y_pose, pose_x)), name='pose')
    if model_type == 'shapepose':
        loss_shape = tf.reduce_mean(tf.abs(tf.subtract(y_shape, shape_x)), name='shape')
        loss_pose = tf.reduce_mean(tf.abs(tf.subtract(y_pose, pose_x)), name='pose')
        loss_joints = tf.reduce_mean(tf.abs(tf.subtract(y_joints[:,:23*3], tf.reshape(smpl_joints[:, 1:, :], (-1,23*3)))), name='joints')
        loss_landmarks = tf.reduce_mean(tf.abs(tf.subtract(y_joints[:,23*3:], tf.reshape(tf.gather(smpl_mesh, landmarks, axis=1), [-1,18*3]))), name='landmarks')
        loss = loss_pose + loss_shape + loss_joints + loss_landmarks


'''################### build training OP ###################'''
# Train OP
print("Training OP...")
with tf.name_scope("train"):
    # Create optimizer and apply gradient descent to the trainable variables
    optimizer = tf.train.AdamOptimizer(lr)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss)
        
# Saver/Loader
saver = tf.train.Saver(var_list=tf.global_variables())
if finetune_pose:
    loader_shape = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='shape'))
    loader_pose = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pose'))
else:
    loader = tf.train.Saver(var_list=tf.global_variables())

np.set_printoptions(precision=3)

'''################### start session ###################'''
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
# Start TF session
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:    
    # Initialize variable
    sess.run(tf.global_variables_initializer())

    if finetune_pose and os.path.exists(os.path.join(checkpoint_path, finetune + '.ckpt.index')) and os.path.exists(os.path.join(checkpoint_path, finetune_pose + '.ckpt.index')):
        print("Restoring models: " + finetune + " and " + finetune_pose)
        loader_shape.restore(sess, os.path.join(checkpoint_path, finetune + '.ckpt'))
        loader_pose.restore(sess, os.path.join(checkpoint_path, finetune_pose + '.ckpt'))
    elif finetune and os.path.exists(os.path.join(checkpoint_path, finetune + '.ckpt.index')):
        print("Restoring model: " + finetune)
        loader.restore(sess, os.path.join(checkpoint_path, finetune + '.ckpt'))
    else:
        print("Training from scratch")

    print("Start training...")
    
    best = float('inf')
    
    # Loop over epochs
    for epoch in range(num_epochs):
        start = time.time()
        print("")
        print("")
        print("EPOCH NUMBER: " + str(epoch))
        print("-------------------------------------------")
        
        perm = np.random.permutation(tr_data.data_size) # 
        
        train_loss = 0
        dae_joints_error = 0
        shape_loss = 0
        pose_loss = 0
        smpl_points_error = 0
        train_count = 0
        for step in range(train_batches):
            r = perm[step*batch_size : (step+1)*batch_size]
            
            # Run training op
            feed_dict={joints: tr_data.data[POINTS3D][r], 
              y_shape: tr_data.labels[SHAPE][r],
              y_joints: tr_data.labels[POINTS3D][r], 
              y_pose: tr_data.labels[POSE][r], 
              keep_prob: dropout, 
              is_training: True,
              lr: learning_rate*(1.0-1.0/(1+np.exp(4-epoch/12.5)))+0.0001}
            _, _loss, _joints_x, _shape_x, _pose_x, _smpl_joints, _smpl_mesh = sess.run([train_op, loss, joints_x, shape_x, pose_x, smpl_joints, smpl_mesh], feed_dict=feed_dict)

            # Compute avg loss
            train_loss += _loss
            dae_joints_error += get_euc_err(_joints_x, tr_data.labels[POINTS3D][r])
            shape_loss += np.mean(np.abs(_shape_x - tr_data.labels[SHAPE][r]))
            pose_loss += np.mean(np.abs(_pose_x - tr_data.labels[POSE][r]))
            if model_type == 'shapepose':
                _smpl_points = np.concatenate([_smpl_joints[:, 1:, :], _smpl_mesh[:, landmarks, :]], 1)
                smpl_points_error += get_euc_err(_smpl_points, tr_data.labels[POINTS3D][r])
            else:
                smpl_points_error = 0
            train_count += 1
        
        train_loss /= train_count
        dae_joints_error /= train_count
        shape_loss /= train_count
        pose_loss /= train_count
        smpl_points_error /= train_count
        print("")
        print("Loss: %2.3f, dae: %2.3f, shape: %2.3f, pose: %2.3f, smpl: %2.3f" % 
             (train_loss, dae_joints_error, shape_loss, pose_loss, smpl_points_error))
        sys.stdout.flush()
        
        # Validation
        print("")
        print("VALIDATION")
        
        val_loss = 0
        dae_joints_error = 0
        shape_loss = 0
        pose_loss = 0
        smpl_points_error = 0
        val_count = 0
        for step in range(val_batches):
            r = np.arange(step*batch_size, (step+1)*batch_size)
            
            feed_dict={joints: val_data.data[POINTS3D][r], 
              y_shape: val_data.labels[SHAPE][r],
              y_joints: val_data.labels[POINTS3D][r], 
              y_pose: val_data.labels[POSE][r], 
              keep_prob: 1.0,
              is_training: False,
              lr: 0}
            _loss, _joints_x, _shape_x, _pose_x, _smpl_joints, _smpl_mesh = sess.run([loss, joints_x, shape_x, pose_x, smpl_joints, smpl_mesh], feed_dict=feed_dict)

            # Compute avg loss
            val_loss += _loss
            dae_joints_error += get_euc_err(_joints_x, val_data.labels[POINTS3D][r])
            shape_loss += np.mean(np.abs(_shape_x - val_data.labels[SHAPE][r]))
            pose_loss += np.mean(np.abs(_pose_x - val_data.labels[POSE][r]))
            if model_type != 'dae':
                _smpl_points = np.concatenate([_smpl_joints[:, 1:, :], _smpl_mesh[:, landmarks, :]], 1)
                smpl_points_error += get_euc_err(_smpl_points, val_data.labels[POINTS3D][r])
            else:
                smpl_points_error = 0
            val_count += 1
        
        val_loss /= val_count
        dae_joints_error /= val_count
        shape_loss /= val_count
        pose_loss /= val_count
        smpl_points_error /= val_count
        print("")
        print("Loss: %2.3f, dae: %2.3f, shape: %2.3f, pose: %2.3f, smpl: %2.3f" % 
             (val_loss, dae_joints_error, shape_loss, pose_loss, smpl_points_error))
        
        if val_loss < best:
            best = val_loss

            print("Best model detected...")
            checkpoint_name = os.path.join(checkpoint_path, name + '.ckpt')
            if is_save:
                save_path = saver.save(sess, checkpoint_name)
                print("Model saved at: " + checkpoint_name)
        
        print("ELAPSED: {:.2f}".format(time.time() - start))
        sys.stdout.flush()
