'''
Created by Hugo Bertiche in Oct. 2018
'''


import tensorflow as tf

DEFAULT_PROB1 = .25
DEFAULT_PROB2 = .35

def randomHUE(img, prob=DEFAULT_PROB1):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: tf.image.random_hue(img, .5)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def randomBrightness(img, prob=DEFAULT_PROB1):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: tf.image.random_brightness(img, max_delta=.05)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def randomSaturation(img, prob=DEFAULT_PROB1):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: tf.image.random_saturation(img, lower=.05, upper=0.15)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def randomContrast(img, prob=DEFAULT_PROB1):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: tf.image.random_contrast(img, lower=.05, upper=0.15)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)
	
def randomNoise(img, prob=DEFAULT_PROB1):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: img + tf.random_normal(shape=tf.shape(img), mean=0, stddev=.05, dtype=tf.float32)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def randomBackground(img, mask, objects, size, prob=DEFAULT_PROB2):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: _change_background(img, mask, objects, size)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def _change_background(img, mask, objects):
	# Read and resize mask
	mask = tf.read_file(mask)
	mask = tf.image.decode_png(mask, channels=1)
	mask = tf.cast(mask, tf.float32)
	mask = tf.clip_by_value(mask, 0, 1)
	mask = tf.image.resize_images(mask, (256, 256))
	mask = tf.cast(mask, tf.float32)
	# Read VOC sample
	object = tf.gather(objects, tf.random_uniform(shape=(), minval=0, maxval=objects.get_shape()[0] - 1, dtype=tf.int32))
	object = tf.read_file(object)
	object = tf.image.decode_jpeg(object, channels=3)
	object = tf.cast(object, tf.float32)
	# mean_o, var_o = tf.nn.moments(object, axes=[0, 1])
	# object = tf.divide(tf.subtract(object, mean_o), tf.sqrt(var_o))
	object = tf.image.resize_images(object, (256, 256))
	# Change backogrund
	img = tf.multiply(img, mask) + tf.multiply(object, 1 - mask)
	return img
	
def randomVOC(img, objects, prob=DEFAULT_PROB2):
	if prob <= 0:
		return img
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: _put_voc_object(img, objects)
	fn2 = lambda: img
	return tf.cond(pred, fn1, fn2)

def _put_voc_object(img, objects):
	# Pick random VOC sample
	object = tf.gather(objects, tf.random_uniform(shape=(), minval=0, maxval=objects.get_shape()[0] - 1, dtype=tf.int32))
	# Read object mask
	mask = tf.regex_replace(tf.regex_replace(object, 'JPEGImages', 'SegmentationObject'), '.jpg', '.png')
	mask = tf.read_file(mask)
	mask = tf.image.decode_png(mask, channels=1)
	# Pick one object (multiple objects per mask)
	# Get list of idx
	idx = tf.cast(tf.unique(tf.reshape(mask, [-1]))[0], tf.int32)
	# Remove '0' (no object)
	idx = tf.contrib.framework.sort(idx)
	idx = idx[1:]
	idx = tf.concat((idx[0:1], idx), axis=0)
	# Remove '220' (objects contour)
	idx = tf.contrib.framework.sort(idx, direction='DESCENDING')
	idx = idx[1:]
	idx = tf.unique(idx)[0]
	idx = tf.cast(tf.random_shuffle(idx)[0], tf.uint8)
	# Mask object
	mask = tf.cast(tf.equal(mask, idx), tf.float32)
	# Read and normalize VOC image
	object = tf.read_file(object)
	object = tf.image.decode_jpeg(object, channels=3)
	object = tf.cast(object, tf.float32)
	# mean_o, var_o = tf.nn.moments(object, axes=[0, 1])
	# object = tf.divide(tf.subtract(object, mean_o), tf.sqrt(var_o))
	# Resizing object and mask
	object = tf.image.resize_images(object, (128, 128))
	mask = tf.image.resize_images(mask, (128, 128))
	offset = tf.random_uniform([2], minval=0, maxval=127, dtype=tf.int32)
	object = tf.image.pad_to_bounding_box(object, offset[0], offset[1], 256, 256)
	mask = tf.image.pad_to_bounding_box(mask, offset[0], offset[1], 256, 256)
	# Mix image and object
	img = tf.multiply(img, 1 - mask) + tf.multiply(object, mask)
	return img

def randomFlip(image, points, pose, prob=DEFAULT_PROB1):
	if prob <= 0:
		return image
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: _flip(image, points, pose)
	fn2 = lambda: _no_flip(image, points, pose)
	return tf.cond(pred, fn1, fn2)

flipper = tf.reshape(tf.constant([1, 1, -1], tf.float32), (1, 3))
flip_mat = tf.constant([[[-1, 0, 0],[0, 1, 0],[0, 0, 1]]], tf.float32)
def _flip(image, label, pose):
	# Flip image
	image = tf.image.flip_left_right(image)
	# Flip joints
	label = tf.multiply(flipper, label)
	# Flip pose
	mat = tf.concat((flip_mat, eyes), axis=0)

	pose = tf.reshape(pose, (24, 3, 3))
	pose = tf.matmul(mat, pose)
	pose = tf.reshape(pose, [-1])
	return image, label, pose
	
def _no_flip(image, label, pose):
	return image, label, pose

def randomRotation(image, points, pose, prob=DEFAULT_PROB1):
	if prob <= 0:
		return image
	pred = tf.random_uniform([]) < prob
	fn1 = lambda: _rotate(image, points, pose)
	fn2 = lambda: _no_rotate(image, points, pose)
	return tf.cond(pred, fn1, fn2)

eyes = tf.tile(tf.expand_dims(tf.eye(3), 0), [23, 1, 1])	
def _rotate(image, points, pose, max=0.15):
	degree = tf.random_uniform([], maxval=max) * tf.sign(tf.random_uniform([]) - 0.5)
	c = tf.cos(degree)
	s = tf.sin(degree)
	# Rotation matrix in 3D for joints/landmarks
	rot3d = tf.stack([[1, 0, 0], [0, c, s], [0, -s, c]])
	rot3d2 = tf.stack([[c, -s, 0], [s, c, 0], [0, 0, 1]])
	# Rotate image and points
	image.set_shape((256, 256, 3))
	image = tf.contrib.image.rotate(image, degree)
	points = tf.transpose(tf.matmul(rot3d, tf.transpose(points)))
	# Rotate pose
	pose = tf.reshape(pose, (24, 3, 3))
	rot3d2 = tf.concat((tf.expand_dims(rot3d2, 0), eyes), axis=0) # 24x3x3
	pose = tf.matmul(rot3d2, pose)
	pose = tf.reshape(pose, [-1])
	return image, points, pose

def _no_rotate(image, points, pose):
	return image, points, pose
