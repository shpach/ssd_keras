from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TerminateOnNaN, CSVLogger
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from math import ceil
import numpy as np
from matplotlib import pyplot as plt

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

img_height = 300 # Height of the model input images
img_width = 300 # Width of the model input images
n_classes = 20 # Number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO
normalize_coords = True

def main():
    K.clear_session() # Clear previous models from memory.

    train_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)
    val_dataset = DataGenerator(load_images_into_memory=False, hdf5_dataset_path=None)

    # VOC 2007
    # The directories that contain the images.
    VOC_2007_train_images_dir      = './data/VOC2007/train/JPEGImages/'
    VOC_2007_test_images_dir      = './data/VOC2007/test/JPEGImages/'

    VOC_2007_train_anns_dir      = './data/VOC2007/train/Annotations/'
    VOC_2007_test_anns_dir      = './data/VOC2007/test/Annotations/'

    # The paths to the image sets.
    VOC_2007_train_image_set_filename    = './data/VOC2007/train/ImageSets/Main/train.txt'
    VOC_2007_val_image_set_filename      = './data/VOC2007/train/ImageSets/Main/val.txt'
    VOC_2007_trainval_image_set_filename = './data/VOC2007/train/ImageSets/Main/trainval.txt'
    VOC_2007_test_image_set_filename     = './data/VOC2007/test/ImageSets/Main/test.txt'

    # The XML parser needs to now what object class names to look for and in which order to map them to integers.
    classes = ['background',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat',
               'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']

    train_dataset.parse_xml(images_dirs=[VOC_2007_train_images_dir],
                                       #VOC_2012_images_dir],
                          image_set_filenames=[VOC_2007_trainval_image_set_filename],
                                               #VOC_2012_trainval_image_set_filename],
                          annotations_dirs=[VOC_2007_train_anns_dir],
                                            #VOC_2012_annotations_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=True,
                          exclude_difficult=True,
                          ret=False)

    val_dataset.parse_xml(images_dirs=[VOC_2007_test_images_dir],
                          image_set_filenames=[VOC_2007_test_image_set_filename],
                          annotations_dirs=[VOC_2007_test_anns_dir],
                          classes=classes,
                          include_classes='all',
                          exclude_truncated=True,
                          exclude_difficult=True,
                          ret=False)

    convert_to_3_channels = ConvertTo3Channels()
    resize = Resize(height=img_height, width=img_width)
        
    predict_generator = train_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[convert_to_3_channels,
                                                          resize],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'filenames',
                                                  'inverse_transform',
                                                  'original_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

    # 2: Generate samples.

    batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

    i = 0 # Which batch item to look at

    print("Image:", batch_filenames[i])
    print()
    print("Ground truth boxes:\n")
    print(np.array(batch_original_labels[i]))


    ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

    # 3: Make predictions.
    model = model = load_model("vgg19BNmodel.h5", custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})
    y_pred = model.predict(batch_images)

    # 4: Decode the raw predictions in `y_pred`.

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=0.5,
                                       iou_threshold=0.4,
                                       top_k=200,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    # 5: Convert the predictions for the original image.

    y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

    np.set_printoptions(precision=2, suppress=True, linewidth=90)
    print("Predicted boxes:\n")
    print('   class   conf xmin   ymin   xmax   ymax')
    print(y_pred_decoded_inv[i])

    # 5: Draw the predicted boxes onto the image

    # Set the colors for the bounding boxes
    colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()

    plt.figure(figsize=(20,12))
    for i in range(10):
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)

        i = 0 # Which batch item to look at

        print("Image:", batch_filenames[i])
        print()
        print("Ground truth boxes:\n")
        print(np.array(batch_original_labels[i]))


        ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

        # 3: Make predictions.
        model = model = load_model("vgg19BNmodel.h5", custom_objects={'AnchorBoxes': AnchorBoxes,
                                                   'L2Normalization': L2Normalization,
                                                   'DecodeDetections': DecodeDetections,
                                                   'compute_loss': ssd_loss.compute_loss})
        y_pred = model.predict(batch_images)

        # 4: Decode the raw predictions in `y_pred`.

        y_pred_decoded = decode_detections(y_pred,
                                           confidence_thresh=0.5,
                                           iou_threshold=0.5,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)

        # 5: Convert the predictions for the original image.

        y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded_inv[i])

        plt.imshow(batch_original_images[i])

        current_axis = plt.gca()

        #for box in batch_original_labels[i]:
        #    xmin = box[1]
        #    ymin = box[2]
        #    xmax = box[3]
        #    ymax = box[4]
        #    label = '{}'.format(classes[int(box[0])])
        #    current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))  
        #    current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})

        for box in y_pred_decoded_inv[i]:
            xmin = box[2]
            ymin = box[3]
            xmax = box[4]
            ymax = box[5]
            color = colors[int(box[0])]
            label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='red', fill=False, linewidth=2))  
            current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'red', 'alpha':1.0})

        plt.show()

if __name__ == "__main__":
    main()
