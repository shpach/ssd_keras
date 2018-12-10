## SSD: Single-Shot MultiBox Detector implementation in Keras
Original attempt: https://github.com/shpach/SSD-Experiments
---
### Contents

### Overview

This is a Keras port of the SSD model architecture introduced by Wei Liu et al. in the paper [SSD: Single Shot MultiBox Detector](https://arxiv.org/abs/1512.02325).

#### Training details

To train the original SSD300 model on Pascal VOC:

1. Download the datasets:
  ```c
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
  wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
  ```
2. Download the weights for the convolutionalized VGG-16 or for one of the trained original models provided below.
3. Set the file paths for the datasets and model weights accordingly in [`ssd300_training.ipynb`](ssd300_training.ipynb) and execute the cells.

#### Directory Structure

```bash
.
├── LICENSE.txt
├── README.md
├── __init__.py
├── bounding_box_utils
│   ├── __init__.py
│   └── bounding_box_utils.py
├── data
│   ├── VGG_ILSVRC_16_layers_fc_reduced.h5
│   ├── VOC2007
│   │   ├── test
│   │   │   ├── Annotations
│   │   │   ├── ImageSets
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   └── SegmentationObject
│   │   └── train
│   │       ├── Annotations
│   │       ├── ImageSets
│   │       ├── JPEGImages
│   │       ├── SegmentationClass
│   │       └── SegmentationObject
│   ├── VOCtrainval_06-Nov-2007.tar
│   └── VOCtrainval_11-May-2012.tar
├── data_generator
│   ├── __init__.py
│   ├── data_augmentation_chain_constant_input_size.py
│   ├── data_augmentation_chain_original_ssd.py
│   ├── data_augmentation_chain_satellite.py
│   ├── data_augmentation_chain_variable_input_size.py
│   ├── object_detection_2d_data_generator.py
│   ├── object_detection_2d_geometric_ops.py
│   ├── object_detection_2d_image_boxes_validation_utils.py
│   ├── object_detection_2d_misc_utils.py
│   ├── object_detection_2d_patch_sampling_ops.py
│   └── object_detection_2d_photometric_ops.py
├── eval_utils
│   ├── __init__.py
│   ├── average_precision_evaluator.py
│   └── coco_utils.py
├── keras_layers
│   ├── __init__.py
│   ├── keras_layer_AnchorBoxes.py
│   ├── keras_layer_DecodeDetections.py
│   ├── keras_layer_DecodeDetectionsFast.py
│   └── keras_layer_L2Normalization.py
├── keras_loss_function
│   ├── __init__.py
│   └── keras_ssd_loss.py
├── misc_utils
│   ├── __init__.py
│   └── tensor_sampling_utils.py
├── model.h5
├── models
│   ├── __init__.py
│   └── keras_ssd300.py
├── ssd300_evaluation.ipynb
├── ssd300_evaluation_COCO.ipynb
├── ssd300_inference.ipynb
├── ssd300_pascal_07+12_training_log.csv
├── ssd300_training.ipynb
├── ssd_encoder_decoder
│   ├── __init__.py
│   ├── matching_utils.py
│   ├── ssd_input_encoder.py
│   └── ssd_output_decoder.py
├── train_ssd.py
└── training_summaries
    ├── ssd300_pascal_07+12_loss_history.png
    └── ssd300_pascal_07+12_training_summary.md
```

### Performance

Here are the mAP evaluation results of the ported weights and below that the evaluation results of a model trained from scratch using this implementation. All models were evaluated using the official Pascal VOC test server (for 2012 `test`) or the official Pascal VOC Matlab evaluation script (for 2007 `test`). In all cases the results match (or slightly surpass) those of the original Caffe models. Download links to all ported weights are available further below.

<table width="70%">
  <tr>
    <td></td>
    <td colspan=1 align=center>Mean Average Precision</td>
    <td colspan=1 align=center>Training Loss</td>
  </tr>
  <tr>
    <td>Evaluated on</td>
    <td colspan=2 align=center>VOC2007 test</td>
  </tr>
  <tr>
    <td><b>Original SSD</td>
    <td align=center><b>0.738</td>
    <td align=center><b>3.9</td>
  </tr>
  <tr>
    <td><b>Modernized SSD</td>
    <td align=center><b>0.52</td>
    <td align=center><b>4.72</td>
  </tr>
</table>



#### Encoding and decoding boxes

The [`ssd_encoder_decoder`](ssd_encoder_decoder) sub-package contains all functions and classes related to encoding and decoding boxes. Encoding boxes means converting ground truth labels into the target format that the loss function needs during training. It is this encoding process in which the matching of ground truth boxes to anchor boxes (the paper calls them default boxes and in the original C++ code they are called priors - all the same thing) happens. Decoding boxes means converting raw model output back to the input label format, which entails various conversion and filtering processes such as non-maximum suppression (NMS).

In order to train the model, you need to create an instance of `SSDInputEncoder` that needs to be passed to the data generator. The data generator does the rest, so you don't usually need to call any of `SSDInputEncoder`'s methods manually.

Models can be created in 'training' or 'inference' mode. In 'training' mode, the model outputs the raw prediction tensor that still needs to be post-processed with coordinate conversion, confidence thresholding, non-maximum suppression, etc. The functions `decode_detections()` and `decode_detections_fast()` are responsible for that. The former follows the original Caffe implementation, which entails performing NMS per object class, while the latter performs NMS globally across all object classes and is thus more efficient, but also behaves slightly differently. Read the documentation for details about both functions. If a model is created in 'inference' mode, its last layer is the `DecodeDetections` layer, which performs all the post-processing that `decode_detections()` does, but in TensorFlow. That means the output of the model is already the post-processed output. In order to be trainable, a model must be created in 'training' mode. The trained weights can then later be loaded into a model that was created in 'inference' mode.

A note on the anchor box offset coordinates used internally by the model: This may or may not be obvious to you, but it is important to understand that it is not possible for the model to predict absolute coordinates for the predicted bounding boxes. In order to be able to predict absolute box coordinates, the convolutional layers responsible for localization would need to produce different output values for the same object instance at different locations within the input image. This isn't possible of course: For a given input to the filter of a convolutional layer, the filter will produce the same output regardless of the spatial position within the image because of the shared weights. This is the reason why the model predicts offsets to anchor boxes instead of absolute coordinates, and why during training, absolute ground truth coordinates are converted to anchor box offsets in the encoding process. The fact that the model predicts offsets to anchor box coordinates is in turn the reason why the model contains anchor box layers that do nothing but output the anchor box coordinates so that the model's output tensor can include those. If the model's output tensor did not contain the anchor box coordinates, the information to convert the predicted offsets back to absolute coordinates would be missing in the model output.

### Training on Google Cloud
If you want to train on Google Cloud, you will need to port over this entire codebase to a Google Cloud bucket, as well as the training and validation data. Then, edit the training scripts to point to where your datasets exist. To actually run the job, you will have to provide the [`train_ssd.py`](train_ssd.py) script to the ML Engine. You also need to specify the path to your GPU config (we have given ours in this repository). Specifically, if you were in the parent directory to this repository. You could run:
```bash
gcloud ml-engine jobs submit training model1_job --module-name=ssd_keras.train_ssd --package-path=./ssd_keras --job-dir=gs://deeplearningteam11/logs --config=./ssd_keras/cloudml-gpu.yaml --python-version 3.5 --runtime-version 1.10 --region us-east1
```

If you wanted to resume training an existing model, you will need to change the range of epochs in which you will train the model in [`train_existing.py`](train_existing.py). Then, from the parent directory to this repository, you could run:
```bash
gcloud ml-engine jobs submit training resume_model1_job --module-name=ssd_keras.train_existing --package-path=./ssd_keras --job-dir=gs://deeplearningteam11/logs --config=./ssd_keras/cloudml-gpu.yaml --python-version 3.5 --runtime-version 1.10 --region us-east1
```

### Examples

Below are some prediction examples of the fully trained original SSD300 "07+12" model (i.e. trained on Pascal VOC2007 `trainval` and VOC2012 `trainval`). The predictions were made on Pascal VOC2007 `test`.

| | |
|---|---|
| ![img01](./examples/tvmonitor.jpeg) | ![img01](./examples/multiple-bikes.jpeg) |
<!-- | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_01_no_gt.png) | ![img01](./examples/trained_ssd300_pascalVOC2007_test_pred_02_no_gt.png) | -->

Here are some prediction examples of an SSD7 (i.e. the small 7-layer version) partially trained on two road traffic datasets released by [Udacity](https://github.com/udacity/self-driving-car/tree/master/annotations) with roughly 20,000 images in total and 5 object categories (more info in [`ssd7_training.ipynb`](ssd7_training.ipynb)). The predictions you see below were made after 10,000 training steps at batch size 32. Admittedly, cars are comparatively easy objects to detect and I picked a few of the better examples, but it is nonetheless remarkable what such a small model can do after only 10,000 training iterations.

| | |
|---|---|
| ![img01](./examples/videopic1.jpeg) | ![img01](./examples/videopic2.jpeg) |
<!-- | ![img01](./examples/ssd7_udacity_traffic_pred_03.png) | ![img01](./examples/ssd7_udacity_traffic_pred_04.png) | -->
