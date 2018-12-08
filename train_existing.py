from tensorflow.python.lib.io import file_io
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
import tensorflow as tf
from keras.models import load_model
from keras.utils import plot_model
from math import ceil
import numpy as np
#from matplotlib import pyplot as plt
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_input_encoder import SSDInputEncoder
from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.data_augmentation_chain_original_ssd import SSDDataAugmentation
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms
from tensorflow.python.lib.io import file_io
import argparse

from tensorflow.python.client import device_lib
print("CHECK GPU USAGE!")
print(device_lib.list_local_devices())
K.tensorflow_backend._get_available_gpus()

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
img_channels = 3 # Number of color channels of the model input images
mean_color = [123, 117, 104] # The per-channel mean of the images in the dataset. Do not change this value if you're using any of the pre-trained weights.
swap_channels = [2, 1, 0] # The color channel order in the original SSD is BGR, so we'll have the model reverse the color channel order of the input images.
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
scales_pascal = [0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05] # The anchor box scaling factors used in the original SSD300 for the Pascal VOC datasets
scales_coco = [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05] # The anchor box scaling factors used in the original SSD300 for the MS COCO datasets
scales = scales_pascal
aspect_ratios = [[1.0, 2.0, 0.5, 3.0, 1.0/3.0, 4.0, 0.25],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0, 4.0, 0.25],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                 [1.0, 2.0, 0.5, 3.0, 1.0/3.0]] # The anchor box aspect ratios used in the original SSD300; the order matters
two_boxes_for_ar1 = True
steps = [8, 16, 32, 64, 100, 300] # The space between two adjacent anchor box center points for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5] # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the step size for each predictor layer.
clip_boxes = False # Whether or not to clip the anchor boxes to lie entirely within the image boundaries
variances = [0.1, 0.1, 0.2, 0.2] # The variances by which the encoded target coordinates are divided as in the original implementation
normalize_coords = True

def main(job_dir, **args):
    ##Setting up the path for saving logs
    logs_dir = job_dir + 'logs/'
    data_dir = "gs://deeplearningteam11/data"

    print("Current Directory: " + os.path.dirname(__file__))
    print("Lets copy the data to: " + os.path.dirname(__file__))
    os.system("gsutil -m cp -r " + data_dir + "  " + os.path.dirname(__file__) + " > /dev/null 2>&1 " )
    #exit(0)

    with tf.device('/device:GPU:0'):
        # 1: Build the Keras model.
        K.clear_session() # Clear previous models from memory.
        model = ssd_300(image_size=(img_height, img_width, img_channels),
                        n_classes=n_classes,
                        mode='training',
                        l2_regularization=0.0005,
                        scales=scales,
                        aspect_ratios_per_layer=aspect_ratios,
                        two_boxes_for_ar1=two_boxes_for_ar1,
                        steps=steps,
                        offsets=offsets,
                        clip_boxes=clip_boxes,
                        variances=variances,
                        normalize_coords=normalize_coords,
                        subtract_mean=mean_color,
                        swap_channels=swap_channels)

        # Set the path to the `.h5` file of the model to be loaded.
        model_file = file_io.FileIO('gs://deeplearningteam11/vgg19BNmodel.h5', mode='rb')

        # Store model locally on instance
        model_path = 'model.h5'
        with open(model_path, 'wb') as f:
          f.write(model_file.read())
        model_file.close()

        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                                 'L2Normalization': L2Normalization,
                                                 'DecodeDetections': DecodeDetections,
                                                 'compute_loss': ssd_loss.compute_loss})

        for layer in model.layers:
          layer.trainable = True

        model.summary()

        # 1: Instantiate two `DataGenerator` objects: One for training, one for validation.
        train_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)
        val_dataset = DataGenerator(load_images_into_memory=True, hdf5_dataset_path=None)

        # 2: Parse the image and label lists for the training and validation datasets. This can take a while.
        #  VOC 2007
        #  The directories that contain the images.
        VOC_2007_train_images_dir      =  'data/data/VOC2007/train/JPEGImages/'
        VOC_2007_test_images_dir      = 'data/data/VOC2007/test/JPEGImages/'

        VOC_2007_train_anns_dir      =  'data/data/VOC2007/train/Annotations/'
        VOC_2007_test_anns_dir      = 'data/data/VOC2007/test/Annotations/'

        # The paths to the image sets.
        VOC_2007_trainval_image_set_dir = 'data/data/VOC2007/train/ImageSets/Main/'
        VOC_2007_test_image_set_dir = 'data/data/VOC2007/test/ImageSets/Main/'

        VOC_2007_train_images_dir = os.path.dirname(__file__) + "/" + VOC_2007_train_images_dir
        VOC_2007_test_images_dir = os.path.dirname(__file__) + "/" + VOC_2007_test_images_dir

        VOC_2007_train_anns_dir = os.path.dirname(__file__) + "/" + VOC_2007_train_anns_dir
        VOC_2007_test_anns_dir = os.path.dirname(__file__) + "/" + VOC_2007_test_anns_dir

        VOC_2007_trainval_image_set_dir = os.path.dirname(__file__) + "/" + VOC_2007_trainval_image_set_dir
        VOC_2007_test_image_set_dir = os.path.dirname(__file__) + "/" + VOC_2007_test_image_set_dir

        VOC_2007_trainval_image_set_filename = VOC_2007_trainval_image_set_dir + '/trainval.txt'
        VOC_2007_test_image_set_filename = VOC_2007_test_image_set_dir + '/test.txt'

        # The XML parser needs to now what object class names to look for and in which order to map them to integers.
        classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

        print("Parsing Training Data ...")
        train_dataset.parse_xml(images_dirs=[VOC_2007_train_images_dir],
                        image_set_filenames=[VOC_2007_trainval_image_set_filename],
                        annotations_dirs=[VOC_2007_train_anns_dir],
                        classes=classes,
                        include_classes='all',
                        exclude_truncated=False,
                        exclude_difficult=False,
                        ret=False,
                        verbose=False)
        print("Done")
        print("================================================================")

        print("Parsing Test Data ...")
        val_dataset.parse_xml(images_dirs=[VOC_2007_test_images_dir],
                      image_set_filenames=[VOC_2007_test_image_set_filename],
                      annotations_dirs=[VOC_2007_test_anns_dir],
                      classes=classes,
                      include_classes='all',
                      exclude_truncated=False,
                      exclude_difficult=True,
                      ret=False,
                      verbose=False)
        print("Done")
        print("================================================================")

        # 3: Set the batch size.
        batch_size = 32 # Change the batch size if you like, or if you run into GPU memory issues.

        #  4: Set the image transformations for pre-processing and data augmentation options.

        # For the training generator:
        ssd_data_augmentation = SSDDataAugmentation(img_height=img_height,
                                                    img_width=img_width,
                                                    background=mean_color)

        # For the validation generator:
        convert_to_3_channels = ConvertTo3Channels()
        resize = Resize(height=img_height, width=img_width)

        # 5: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function.

        # The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
        predictor_sizes = [model.get_layer('conv4_4_norm_mbox_conf').output_shape[1:3],
                           model.get_layer('fc7_mbox_conf').output_shape[1:3],
                           model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv9_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv10_2_mbox_conf').output_shape[1:3],
                           model.get_layer('conv11_2_mbox_conf').output_shape[1:3]]

        ssd_input_encoder = SSDInputEncoder(img_height=img_height,
                                            img_width=img_width,
                                            n_classes=n_classes,
                                            predictor_sizes=predictor_sizes,
                                            scales=scales,
                                            aspect_ratios_per_layer=aspect_ratios,
                                            two_boxes_for_ar1=two_boxes_for_ar1,
                                            steps=steps,
                                            offsets=offsets,
                                            clip_boxes=clip_boxes,
                                            variances=variances,
                                            matching_type='multi',
                                            pos_iou_threshold=0.5,
                                            neg_iou_limit=0.5,
                                            normalize_coords=normalize_coords)

        # 6: Create the generator handles that will be passed to Keras' `fit_generator()` function.

        train_generator = train_dataset.generate(batch_size=batch_size,
                                                 shuffle=True,
                                                 transformations=[ssd_data_augmentation],
                                                 label_encoder=ssd_input_encoder,
                                                 returns={'processed_images',
                                                          'encoded_labels'},
                                                 keep_images_without_gt=False)

        val_generator = val_dataset.generate(batch_size=batch_size,
                                             shuffle=False,
                                             transformations=[convert_to_3_channels,
                                                              resize],
                                             label_encoder=ssd_input_encoder,
                                             returns={'processed_images',
                                                      'encoded_labels'},
                                             keep_images_without_gt=False)

        # Get the number of samples in the training and validations datasets.
        train_dataset_size = train_dataset.get_dataset_size()
        val_dataset_size   = val_dataset.get_dataset_size()

        print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
        print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

        # Define a learning rate schedule.

        def lr_schedule(epoch):
            return 1e-6
            # if epoch < 80:
            #     return 0.001
            # elif epoch < 100:
            #     return 0.0001
            # else:
            #     return 0.00001

        learning_rate_scheduler = LearningRateScheduler(schedule=lr_schedule,
                                                        verbose=1)

        terminate_on_nan = TerminateOnNaN()

        callbacks = [learning_rate_scheduler,
                     terminate_on_nan]


        # If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
        initial_epoch   = 120
        final_epoch     = 200
        steps_per_epoch = 500

        history = model.fit_generator(generator=train_generator,
                                      steps_per_epoch=steps_per_epoch,
                                      epochs=final_epoch,
                                      callbacks=callbacks,
                                      validation_data=val_generator,
                                      validation_steps=ceil(val_dataset_size/batch_size),
                                      initial_epoch=initial_epoch)

        model_name = "vgg19BNmodel_cont.h5"
        model.save(model_name)
        with file_io.FileIO(model_name, mode='rb') as input_f:
            with file_io.FileIO("gs://deeplearningteam11/" + model_name, mode='w+') as output_f:
                output_f.write(input_f.read())
                


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )

    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
