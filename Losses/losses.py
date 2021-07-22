'''
Created by Hugo Bertiche in Oct. 2018
'''


import tensorflow as tf
import numpy as np
import pickle as pkl

class SMPLoss:
	
	def __init__(self, score, y, smpl, gender=0):
		# Compute mesh and joints from params (score / labels)
		score_mesh, score_joints = smpl.run(score, gender)
		y_mesh, y_joints = smpl.run(y, gender)
		
		# Compute squared errors
		self.jointsSE = tf.reduce_sum(tf.square(tf.subtract(score_joints, y_joints)), axis=2) # N x 24
		self.meshSE = tf.reduce_sum(tf.square(tf.subtract(score_mesh, y_mesh)), axis=2) # N x 6890
		
	def joint(self):
		return tf.reduce_mean(self.jointsSE, axis=0)
		
	def joint_err(self):
		return 1000 * tf.reduce_mean(tf.sqrt(self.jointsSE), axis=0)
		
	def mesh(self):
		return tf.reduce_mean(self.meshSE)
		
	def mesh_err(self):
		return 1000 * tf.reduce_mean(tf.sqrt(self.meshSE))
		
class GMMLoss:

	def __init__(self, gmm_path):
		with open(gmm_path, 'rb') as f:
			gmm = pkl.load(f)
		
		self.means = gmm.means_
		
		self.weight_log = gmm._estimate_log_weights()
		self.k, self.n_features, _ = gmm.covariances_.shape
		
		self.first = - .5 * self.n_features * np.log(2 * 3.14159265358979323846)

		self.cholesky = tf.constant(gmm.precisions_cholesky_, tf.float32)
		self.second = np.sum(np.log(gmm.precisions_cholesky_.reshape(self.k, -1)[:, ::self.n_features + 1]), 1)
		
	def run(self, score):
		third = []
		for i in range(self.k):
			dev = tf.subtract(score, self.means[i, :])
			third.append(tf.square(tf.norm(tf.matmul(tf.transpose(self.cholesky[i]), tf.transpose(dev)), axis=0)))
		third = -.5 * tf.transpose(tf.stack(third))
		logprob = self.first + self.second + third
		return tf.reduce_logsumexp(self.weight_log + logprob, axis=1)
	
def heatmapLoss(score, y):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=y))

class HeatmapLoss:

	def __init__(self, size=64, cov=[1., 1.], original_size=200.):
		self._size = size
		self._cov = cov
		self._scale = size / original_size
		self._voxel_size = [.1, .1, .1]
	
		x = np.linspace(0, size - 1, size)
		y = np.linspace(0, size - 1, size)
		xx, yy = np.meshgrid(x, y)
		self._xxyy = tf.reshape(tf.constant(np.c_[xx.ravel(), yy.ravel()], tf.float32), (1, size*size, 2))
	
	def run(self, score, points):
		N = points.get_shape()[1]
		_heatmaps = []
		for i in range(N):
			_heatmaps.append(self._generate_heatmap(tf.gather(points, i, axis=1)))
		_heatmaps = tf.stack(_heatmaps, axis=-1)
		return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score, labels=_heatmaps))
	
	def get_coords(self, heatmap):
		heatmap = tf.nn.softmax(tf.reshape(heatmap, (-1, self._size * self._size, 1)))
		return tf.reduce_sum(tf.multiply(heatmap, self._xxyy), axis=1)	

	def _generate_heatmap(self, points):
		points = tf.reshape(points * self._scale + self._size / 2., (-1, 1, 2))
		mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=points, scale_diag=self._cov)
		Z = mvn.prob(self._xxyy)
		Z = Z / tf.reshape(tf.reduce_max(Z, axis=1) + 1e-16, (-1, 1))
		return tf.reshape(Z, (-1, self._size, self._size))

class VoxelLoss:

	# SURREAL VALUE RANGES (TRAINING ONLY)
	# Z: -1.2099651 - 1.1857777 <----- DEPTH (Camera axis)
	# Y: -1.2889311 - 1.1836728
	# X: -1.2055532 - 1.2032807

	def __init__(self, scale=2.5, size=64, depth=16, cov=[1., 1., 1.]):
		self._size = size
		self._depth = depth
		self._cov = cov
		self._scale = scale
		self._voxel_size = [scale / size,
							scale / size, 
							scale / depth]
		# Limb bone kintree
		self._limbs = [[16, 18, 20, 22], # Right arm
					[15, 17, 19, 21], # left arm
					[4, 7, 10], # Right leg
					[3, 6, 9], # Left leg
					[2, 5, 8, 11, 14]] # Rest (back-neck-head)
						
		x = np.linspace(0, size - 1, size)
		y = np.linspace(0, size - 1, size)
		z = np.linspace(0, depth - 1, depth)

		xx, yy, zz = np.meshgrid(x, y, z)
		self._xxyyzz = tf.reshape(tf.constant(np.c_[xx.ravel(), yy.ravel(), zz.ravel()], tf.float32), (1, size*size*depth, 3))
	
	def run(self, score, points):
		num_stacks = score.get_shape()[0]
		N = points.get_shape()[1]
		_heatmaps = []
		for i in range(N):
			_heatmaps.append(self._generate_heatmap(tf.gather(points, i, axis=1)))
		_heatmaps = tf.stack(_heatmaps, axis=-1)
		
		_loss = []
		for i in range(num_stacks):
			_loss.append( tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score[i], labels=_heatmaps)) )
		return tf.stack(_loss, axis=0)
	
	def run_bones(self, score, points, kintree):
		# Add root joint (0, 0, 0)
		_points = tf.concat((tf.zeros(shape=(points.get_shape()[0], 1, 3), dtype=tf.float32), points), axis=1)
		_limbs = []
		for _ in range(score.get_shape()[-1]):
			limb = self._limbs[_]
			_limb = []
			for bone in limb:
				bone_heads = tf.gather(_points, kintree[0][bone], axis=1)
				bone_ends = tf.gather(_points, kintree[1][bone], axis=1)
				_limb.append(self._generate_bonemap(bone_heads, bone_ends))
			_limb = tf.stack(_limb, axis=0)
			_limb = tf.reduce_max(_limb, axis=0)
			_limbs.append(_limb)
		_limbs = tf.stack(_limbs, axis=-1)
		
		num_stacks = score.get_shape()[0]
		_loss = []
		for i in range(num_stacks):
			_loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score[i], labels=_limbs)))
		return tf.stack(_loss, axis=0)
	
	def run_with_bones(self, score, points, kintree):
		num_stacks = score.get_shape()[0]
		# Joint/Landmark Heatmaps
		N = points.get_shape()[1]
		_heatmaps = []
		for i in range(N):
			_heatmaps.append(self._generate_heatmap(tf.gather(points, i, axis=1)))
		_heatmaps = tf.stack(_heatmaps, axis=-1) # 64x64x16xN
		
		# Limb Heatmaps
		# Add root joint (0, 0, 0)
		_points = tf.concat((tf.zeros(shape=(points.get_shape()[0], 1, 3), dtype=tf.float32), points), axis=1)
		_limbs = []
		for i in range(4):
			limb = self._limbs[i]
			_limb = []
			for bone in limb:
				bone_heads = tf.gather(_points, kintree[0][bone], axis=1)
				bone_ends = tf.gather(_points, kintree[1][bone], axis=1)
				_limb.append(self._generate_bonemap(bone_heads, bone_ends))
			_limb = tf.stack(_limb, axis=0)
			_limb = tf.reduce_max(_limb, axis=0) # 64x64x16 should be
			_limbs.append(_limb)
		_limbs = tf.stack(_limbs, axis=-1) # 64x64x16xL
		
		_y = tf.concat((_heatmaps, _limbs), axis=-1) # 64x64x16x(N + L)
		
		_loss = []
		for i in range(num_stacks):
			_loss.append(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=score[i], labels=_y)))
		return tf.stack(_loss, axis=0)
			
	def get_coords(self, heatmap):
		heatmap = tf.nn.softmax(tf.reshape(heatmap, (-1, self._size * self._size * self._depth, 1)))
		return tf.reduce_sum(tf.multiply(heatmap, self._xxyyzz), axis=1)	
		
	def _generate_heatmap(self, points):
		points = tf.reshape(self._points_to_voxels(points), (-1, 1, 3))
		mvn = tf.contrib.distributions.MultivariateNormalDiag(loc=points, scale_diag=self._cov)
		Z = mvn.prob(self._xxyyzz)
		# Z = Z / tf.reshape(tf.reduce_max(Z, axis=1), (-1, 1))
		return tf.reshape(Z, (-1, self._size, self._size, self._depth))
	
	def _generate_bonemap(self, v, w):
		N = v.get_shape()[0]
		v = self._points_to_voxels(v) # Nx3
		w = self._points_to_voxels(w) # Nx3
		b = w - v # Nx3
		bones = tf.reduce_sum(tf.square(b), axis=1) + .00001 # N
		_Z = []
		for i in range(N):
			a = tf.squeeze(self._xxyyzz) - tf.expand_dims(v[i], 0) # Kx3
			c = tf.reduce_sum(tf.multiply(a, tf.expand_dims(b[i], 0)), axis=1) # K
			t = tf.divide(tf.expand_dims(c, 1), tf.expand_dims(bones[i], 0)) # K
			t = tf.clip_by_value(t, 0, 1)
			proj = tf.expand_dims(v[i], 0) + tf.multiply(t, tf.expand_dims(b[i], 0)) # Kx3
			Z = tf.reduce_sum(tf.square(tf.subtract(proj, tf.squeeze(self._xxyyzz))), axis=1)
			Z = tf.divide(1, Z + 1)
			Z = tf.nn.softmax(Z)
			_Z.append(tf.reshape(Z, (self._size, self._size, self._depth)))
		return tf.stack(_Z, axis=0)
	
	def _points_to_voxels(self, points):
		_points = points / self._scale + .5
		_voxel_X = _points[:, 0] * self._size
		_voxel_Y = _points[:, 1] * self._size
		_voxel_Z = _points[:, 2] * self._depth	
		return tf.stack((_voxel_X, _voxel_Y, _voxel_Z), axis=-1)
	
	def _voxels_to_points(self, voxels):
		N = voxels.get_shape()[0]
		_points_X = (voxels[:, :, 0] / self._size - .5) * self._scale
		_points_Y = (voxels[:, :, 1] / self._size - .5) * self._scale
		_points_Z = (voxels[:, :, 2] / self._depth - .5) * self._scale
		
		_points = tf.stack((_points_X, _points_Y, _points_Z), axis=-1)
		return _points
		
	def _extract_points(self, voxelMaps, numLimbs=4):
		# VoxelMaps Nx64x64x16xNUM_POINTS
		batch_size = voxelMaps.get_shape()[0]
		num_points = voxelMaps.get_shape()[-1] - numLimbs #
		voxels = []
		for i in range(num_points):
			_soft = tf.nn.softmax(tf.reshape(voxelMaps[:, :, :, :, i], (batch_size, -1, 1)), axis=1)
			xxyyzz = tf.reduce_sum(tf.multiply(_soft, self._xxyyzz), axis=1) #Nx1x3
			voxels.append(xxyyzz)

		voxels = tf.stack(voxels, axis=1)
		return self._voxels_to_points(voxels)