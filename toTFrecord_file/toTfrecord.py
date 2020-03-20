
import tensorflow as tf
import numpy
import cv2
import os
import hashlib
import sys
import argparse
import io
import json
import numpy as np

import PIL.Image

from utils import dataset_util

def toTfrecord(f, pathTofile):
	height = None # Image height
	width = None # Image width
	filename = None # Filename of the image. Empty if image is not from file
	encoded_image_data = None # Encoded image bytes
	image_format = b'jpeg' # b'jpeg' or b'png'

	xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
	xmaxs = [] # List of normalized right x coordinates in bounding box (1 per box)
	ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
	ymaxs = [] # List of normalized bottom y coordinates in bounding box (1 per box)
	classes_text = [] # List of string class name of bounding box (1 per box)
	classes = [] # List of integer class id of bounding box (1 per box)
	poses = []
	truncated = []
	difficult_obj = []
	filename = f.readline().rstrip()
	print(filename)
	full_path=os.path.join(pathTofile, filename)
	print(full_path)
	with tf.io.gfile.GFile(full_path, 'rb') as fid:
		encoded_jpg = fid.read()
	
	encoded_jpg_io = io.BytesIO(encoded_jpg)
	image = PIL.Image.open(encoded_jpg_io)
	image_raw = cv2.imread(full_path)
	key = hashlib.sha256(encoded_jpg).hexdigest()
	height, width, channel = image_raw.shape
	print("height is %d, width is %d, channel is %d" % (height, width, channel))

	face_num = int(f.readline().rstrip())
	valid_face_num = 0
	print("face_num:>>", face_num)
	for i in range(face_num):
		annot = f.readline().rstrip().split()
		# WIDER FACE DATASET CONTAINS SOME ANNOTATIONS WHAT EXCEEDS THE IMAGE BOUNDARY

		if(float(annot[2]) > 25.0):
			if(float(annot[3]) > 30.0):
				xmins.append( max(0.005, (float(annot[0]) / width) ) )
				ymins.append( max(0.005, (float(annot[1]) / height) ) )
				xmaxs.append( min(0.995, ((float(annot[0]) + float(annot[2])) / width) ) )
				ymaxs.append( min(0.995, ((float(annot[1]) + float(annot[3])) / height) ) )
				classes_text.append("face".encode('utf8'))
				classes.append(0)
				print(xmins[-1], ymins[-1], xmaxs[-1], ymaxs[-1], classes_text[-1], classes[-1])
				valid_face_num += 1;
				
	feature_dict = {
		'image/height':
		dataset_util.int64_feature(int(height)),
		'image/width':
		dataset_util.int64_feature(int(width)),
		'image/filename':
		dataset_util.bytes_feature(filename.encode('utf8')),
		'image/source_id':
		dataset_util.bytes_feature(filename.encode('utf8')),
		'image/key/sha256':
		dataset_util.bytes_feature(key.encode('utf8')),
		'image/encoded':
		dataset_util.bytes_feature(encoded_jpg),
		'image/format':
		dataset_util.bytes_feature('jpeg'.encode('utf8')),
		'image/object/bbox/xmin':
		dataset_util.float_list_feature(xmins),
		'image/object/bbox/xmax':
		dataset_util.float_list_feature(xmaxs),
		'image/object/bbox/ymin':
		dataset_util.float_list_feature(ymins),
		'image/object/bbox/ymax':
		dataset_util.float_list_feature(ymaxs),
		'image/object/class/text':dataset_util.bytes_list_feature(value=classes_text),
		'image/object/class/label': dataset_util.int64_list_feature(classes),
	}
	print("xxxxx", xmins)
	example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
	return example

def countImagefiles(path):
	count=0
	for name in os.listdir(path):
		for file in os.listdir(path+"/"+name):
			if os.path.isfile(path+"/"+name+"/"+file):
				count+=1
	return count

def main():
	parser = argparse.ArgumentParser()
	# Required arguments.
	parser.add_argument(
		"--path",
		default="WIDER_train",
		help="directory to annotaint txt file")
	# Optional arguments.
	parser.add_argument(
		"--annotation",
		default="wider_face_train_annot.txt",
		help="The name of annotaion file")
	parser.add_argument("--output",
		default ='tfrecord',
		help="Output path"
		)
	parser.add_argument("--filename",
		default ='train',
		help="File name"
		)

	args = parser.parse_args()
	numOfFiles=(countImagefiles(args.path+"/images"))
	f=open(args.path+"/"+args.annotation)
	output = tf.io.TFRecordWriter(args.output+"/"+args.filename+".tfrecord")
	for img in range(numOfFiles):
		example =toTfrecord(f, args.path+"/images")
		
		output.write(example.SerializeToString())
	output.close()




		
if __name__ == '__main__':
	main()