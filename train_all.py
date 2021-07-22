import sys
import subprocess as sp

gpu = sys.argv[1]

def _print(msg):
	print("")
	print("")
	print("")
	print("")
	print("----------------------------------------------------------------------")
	print(msg)
	print("")


_print("TRAIN SHN")
_ = sp.call(['python', 'train_shn.py', 'shn', gpu, 'train', '3'])

_print("TRAIN DAE")
_ = sp.call(['python', 'train_smplr.py', 'dae', gpu, 'dae', 'True'])

_print("TRAIN SHAPE")
_ = sp.call(['python', 'train_smplr.py', 'shape', gpu, 'shape', 'True'])

_print("TRAIN POSE")
_ = sp.call(['python', 'train_smplr.py', 'pose', gpu, 'pose', 'True'])

_print("FINETUNE SHAPE AND POSE TOGETHER")
_ = sp.call(['python', 'train_smplr.py', 'smplr', gpu, 'shapepose', 'True', '0.0001', 'shape', 'pose'])