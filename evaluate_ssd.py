import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from keras import backend as K
from keras.models import load_model
from keras.optimizers import Adam
from scipy.misc import imread
import numpy as np

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization
from data_generator.object_detection_2d_data_generator import DataGenerator
from eval_utils.average_precision_evaluator import Evaluator

import tensorflow as tf
from tensorflow.python.lib.io import file_io

def main():

	# Set a few configuration parameters.
	img_height = 300
	img_width = 300
	n_classes = 20
	model_mode = 'training'

	# Set the path to the `.h5` file of the model to be loaded.
	model_file = file_io.FileIO('gs://deeplearningteam11/vgg19BNmodel.h5', mode='rb')

	# Store model locally on instance
	model_path = 'model.h5'
	with open(model_path, 'wb') as f:
		f.write(model_file.read())
	model_file.close()

	data_dir = "gs://deeplearningteam11/data"
	os.system("gsutil -m cp -r " + data_dir + "  " + os.path.dirname(__file__) + " > /dev/null 2>&1 " )
	# We need to create an SSDLoss object in order to pass that to the model loader.
	ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

	K.clear_session() # Clear previous models from memory.

	model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
	                                               'L2Normalization': L2Normalization,
	                                               'DecodeDetections': DecodeDetections,
	                                               'compute_loss': ssd_loss.compute_loss})

	model.summary()

	te_dataset = DataGenerator(load_images_into_memory=True)
	tr_dataset = DataGenerator(load_images_into_memory=True)

	# TODO: Set the paths to the dataset here.
	tr_Pascal_VOC_dataset_images_dir = os.path.dirname(__file__) + "/" + "data/data/VOC2007/train/JPEGImages/"
	tr_Pascal_VOC_dataset_annotations_dir = os.path.dirname(__file__) + "/" + "data/data/VOC2007/train/Annotations/"
	tr_Pascal_VOC_dataset_image_set_filename = os.path.dirname(__file__) + "/" + "data/data/VOC2007/train/ImageSets/Main/trainval.txt"

	te_Pascal_VOC_dataset_images_dir = os.path.dirname(__file__) + "/" + "data/data/VOC2007/test/JPEGImages/"
	te_Pascal_VOC_dataset_annotations_dir = os.path.dirname(__file__) + "/" + "data/data/VOC2007/test/Annotations/"
	te_Pascal_VOC_dataset_image_set_filename = os.path.dirname(__file__) + "/" + "data/data/VOC2007/test/ImageSets/Main/test.txt"



	# The XML parser needs to now what object class names to look for and in which order to map them to integers.
	classes = ['background',
	           'aeroplane', 'bicycle', 'bird', 'boat',
	           'bottle', 'bus', 'car', 'cat',
	           'chair', 'cow', 'diningtable', 'dog',
	           'horse', 'motorbike', 'person', 'pottedplant',
	           'sheep', 'sofa', 'train', 'tvmonitor']


	with tf.device('/device:GPU:0'):
		# Testing results
		te_dataset.parse_xml(images_dirs=[te_Pascal_VOC_dataset_images_dir],
		                  image_set_filenames=[te_Pascal_VOC_dataset_image_set_filename],
		                  annotations_dirs=[te_Pascal_VOC_dataset_annotations_dir],
		                  classes=classes,
		                  include_classes='all',
		                  exclude_truncated=False,
		                  exclude_difficult=True,
		                  ret=False,
		                  verbose=False)

		te_evaluator = Evaluator(model=model,
		                      n_classes=n_classes,
		                      data_generator=te_dataset,
		                      model_mode=model_mode)

		te_results = te_evaluator(img_height=img_height,
		                    img_width=img_width,
		                    batch_size=64,
		                    data_generator_mode='resize',
		                    round_confidences=False,
		                    matching_iou_threshold=0.5,
		                    border_pixels='include',
		                    sorting_algorithm='quicksort',
		                    average_precision_mode='sample',
		                    num_recall_points=11,
		                    ignore_neutral_boxes=True,
		                    return_precisions=True,
		                    return_recalls=True,
		                    return_average_precisions=True,
		                    verbose=False)

		mean_average_precision, average_precisions, precisions, recalls = te_results

		for i in range(1, len(average_precisions)):
		    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
		print()
		print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

		print('TRAIN')
		tr_dataset.parse_xml(images_dirs=[tr_Pascal_VOC_dataset_images_dir],
		                  image_set_filenames=[tr_Pascal_VOC_dataset_image_set_filename],
		                  annotations_dirs=[tr_Pascal_VOC_dataset_annotations_dir],
		                  classes=classes,
		                  include_classes='all',
		                  exclude_truncated=False,
		                  exclude_difficult=True,
		                  ret=False,
		                  verbose=False)

		# Training results
		tr_evaluator = Evaluator(model=model,
		                      n_classes=n_classes,
		                      data_generator=tr_dataset,
		                      model_mode=model_mode)

		tr_results = tr_evaluator(img_height=img_height,
		                    img_width=img_width,
		                    batch_size=64,
		                    data_generator_mode='resize',
		                    round_confidences=False,
		                    matching_iou_threshold=0.5,
		                    border_pixels='include',
		                    sorting_algorithm='quicksort',
		                    average_precision_mode='sample',
		                    num_recall_points=11,
		                    ignore_neutral_boxes=True,
		                    return_precisions=True,
		                    return_recalls=True,
		                    return_average_precisions=True,
		                    verbose=False)

		mean_average_precision, average_precisions, precisions, recalls = tr_results

		for i in range(1, len(average_precisions)):
		    print("{:<14}{:<6}{}".format(classes[i], 'AP', round(average_precisions[i], 3)))
		print()
		print("{:<14}{:<6}{}".format('','mAP', round(mean_average_precision, 3)))

if __name__ == "__main__":
	main()
