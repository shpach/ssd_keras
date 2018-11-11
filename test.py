from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
import cv2
import copy
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

img_height = 300
img_width = 300
confidence_threshold = 0.5
MATCH_OVERLAP_THRESHOLD = 0.3

K.clear_session() # Clear previous models from memory.

model = ssd_300(image_size=(img_height, img_width, 3),
                n_classes=20,
                mode='inference',
                l2_regularization=0.0005,
                scales=[0.1, 0.2, 0.37, 0.54, 0.71, 0.88, 1.05], # The scales for MS COCO are [0.07, 0.15, 0.33, 0.51, 0.69, 0.87, 1.05]
                aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                         [1.0, 2.0, 0.5],
                                         [1.0, 2.0, 0.5]],
                two_boxes_for_ar1=True,
                steps=[8, 16, 32, 64, 100, 300],
                offsets=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                clip_boxes=False,
                variances=[0.1, 0.1, 0.2, 0.2],
                normalize_coords=True,
                subtract_mean=[123, 117, 104],
                swap_channels=[2, 1, 0],
                confidence_thresh=0.5,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400)

# 2: Load the trained weights into the model.

# TODO: Set the path of the trained weights.
weights_path = './models/VGG_VOC0712_SSD_300x300_iter_120000.h5'

model.load_weights(weights_path, by_name=True)

# 3: Compile the model so that Keras won't complain the next time you load it.

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

ssd_loss = SSDLoss(neg_pos_ratio=3, alpha=1.0)

model.compile(optimizer=adam, loss=ssd_loss.compute_loss)

orig_images = [] # Store the images here.
input_images = [] # Store resized versions of the images here.

# We'll only load one image in this example.
img_path = 'examples/fish_bike.jpg'

orig_images.append(imread(img_path))
img = image.load_img(img_path, target_size=(img_height, img_width))
img = image.img_to_array(img) 
input_images.append(img)
input_images = np.array(input_images)
print(input_images[0][1][1])
y_pred = model.predict(input_images)
confidence_threshold = 0.5
y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
np.set_printoptions(precision=2, suppress=True, linewidth=90)
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

frame=cv2.imread(img_path)
frame=cv2.resize(frame, (img_width, img_height)) 
newframe = (frame[...,::-1].astype(np.float32))
newframe=newframe[np.newaxis, :]
y_pred = model.predict(newframe)
y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
print("Predicted boxes:\n")
print('   class   conf xmin   ymin   xmax   ymax')
print(y_pred_thresh[0])

class BBox:
    def __init__(self,x,y,w,h,score,label):
        self.x=int(x)
        self.y=int(y)
        self.w=int(w)
        self.h=int(h)
        self.score=score
        self.label=label

    def size(self): 
        return self.w * self.h

def boxOverlap(b1, b2): 
	xmin = b1.x if (b1.x < b2.x ) else b2.x
	xmax = b1.x+b1.w if (b1.x+b1.w > b2.x+b2.w) else b2.x+b2.w
	ymin = b1.y if (b1.y < b2.y ) else b2.y
	ymax = b1.y+b1.h if (b1.y+b1.h > b2.y+b2.h) else b2.y+b2.h

	wid1 = int(b1.w)
	hei1 = int(b1.h)
	wid2 = int(b2.w)
	hei2 = int(b2.h)

	wid = int(xmax - xmin)
	hei = int(ymax - ymin)

	if (wid >= wid1 + wid2):
		return -1
	if (hei >= hei1 + hei2):
		return -1
	areaOR = float(np.sqrt(wid1 * hei1) + np.sqrt(wid2 * hei2))
	areaAnd = float(np.sqrt((wid1 + wid2 - wid) * (hei1 + hei2 - hei)))
	return areaAnd / (areaOR - areaAnd)

def creatTrackerList(bbox_list_track):
	trackercount=len(bbox_list_track)
	trackerlist=[]
	for i in range(0,trackercount):
		tracker=cv2.TrackerKCF_create()
		trackerlist.append(tracker)
	# Initialize tracker with frame and bounding box
	for (tracker,bbox) in zip(trackerlist,bbox_list_track):
		ok = tracker.init(frame,(bbox.x,bbox.y,bbox.w,bbox.h))
	return trackerlist

def detecting(frame):
	ori_width=frame.shape[1]
	ori_height=frame.shape[0]
	frame=cv2.resize(frame, (img_width, img_height)) 
	newframe = (frame[...,::-1].astype(np.float32))
	newframe=newframe[np.newaxis, :]
	y_pred = model.predict(newframe)
	y_pred_thresh = [y_pred[k][y_pred[k,:,1] > confidence_threshold] for k in range(y_pred.shape[0])]
	bbox_list_detect=[]
	for box in y_pred_thresh[0]:
	    # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
	    xmin = box[2] * ori_width/ img_width
	    ymin = box[3] * ori_height/ img_height
	    xmax = box[4] * ori_width/ img_width
	    ymax = box[5] * ori_height/ img_height
	    color = colors[int(box[0])]
	    bbox = BBox(xmin,ymin,xmax-xmin+1,ymax-ymin+1,box[1],box[0])
	    bbox_list_detect.append(bbox)
	return bbox_list_detect

def tracking(frame,trackerlist,bbox_list_track):
	for (tracker,bbox) in zip(trackerlist,bbox_list_track): 
		# Update tracker
		ok, newbbox = tracker.update(frame)
		if not ok:
			break
		bbox.x = newbbox[0]
		bbox.y = newbbox[1]
		bbox.w = newbbox[2]
		bbox.h = newbbox[3]
	return bbox_list_track

def drawFrame(frame,bbox_list):
	colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
	for bbox in bbox_list: 
		color = colors[int(bbox.label)]
		color = [colorname * 255 for colorname in color]
		display_txt = '%s: %.2f'%(classes[int(bbox.label)], bbox.score)
		cv2.rectangle(frame,(int(bbox.x),int(bbox.y)),(int(bbox.x)+int(bbox.w),int(bbox.y)+int(bbox.h)),color,2)
		cv2.putText(frame,display_txt,(int(bbox.x),int(bbox.y)), cv2.FONT_HERSHEY_SIMPLEX, 1,color,1,cv2.LINE_AA)
	return frame

def merging(bbox_list_detect,bbox_list_track):
	bbox_list_merge = []
	matchD2T = dict.fromkeys(range(len(bbox_list_detect)), -1)
	matchT2D = dict.fromkeys(range(len(bbox_list_track)), -1)
	bestT2D = dict.fromkeys(range(len(bbox_list_track)), -1)
	# Calculate IOU and cheak match pairs
	for i in range(0,len(bbox_list_track)):
		for j in range(0,len(bbox_list_detect)):
			d = boxOverlap(bbox_list_track[i],bbox_list_detect[j])
			if((d > MATCH_OVERLAP_THRESHOLD) and(bbox_list_track[i].label==bbox_list_detect[j].label)):
				matchD2T[j]=i;
				if((matchT2D[i] == -1) or (d > bestT2D[i])):
					matchT2D[i] = j;
					bestT2D[i] = d;
	# Update old vehicles from best overlap
	for i in range(0,len(bbox_list_track)):
		bbox = bbox_list_track[i];
		if (matchT2D[i] != -1):
			bbox = bbox_list_detect[matchT2D[i]];
		bbox_list_merge.append(bbox)
	# Create new object
	for i in range(0,len(bbox_list_detect)):
		if (matchD2T[i] == -1):
			bbox = bbox_list_detect[i];
			bbox_list_merge.append(bbox)
	return bbox_list_merge

colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
classes = ['background',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat',
           'chair', 'cow', 'diningtable', 'dog',
           'horse', 'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor']

video = cv2.VideoCapture("examples/videos/4.mp4")
if not video.isOpened():
	print ("Could not open video")
	sys.exit()
fps = video.get(cv2.CAP_PROP_FPS)
print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
video_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('examples/videos/4_out.mp4',fourcc, fps, (video_width,video_height))

framecount=0
while(True):
	print ('Framecount'+str(framecount))
	ok, frame = video.read()
	if not ok:
		break
	trackerlist = []
	if(framecount % 4 == 0):
		bbox_list_detect = detecting(frame)
		frame = drawFrame(frame,bbox_list_detect)
		out.write(frame)
		bbox_list_track = copy.deepcopy(bbox_list_detect)
		trackerlist = creatTrackerList(bbox_list_track)
	else:
		bbox_list_track = tracking(frame,trackerlist,bbox_list_track)
		bbox_list_detect = detecting(frame)
		bbox_list_track = merging(bbox_list_detect,bbox_list_track)
		trackerlist = creatTrackerList(bbox_list_track)
		frame = drawFrame(frame,bbox_list_track)
		out.write(frame)
	framecount+=1

print ("done")
out.release()
cv2.destroyAllWindows()

# for box in y_pred_thresh[0]:
#     # Transform the predicted bounding boxes for the 300x300 image to the original image dimensions.
#     xmin = box[2] * orig_images[0].shape[1] / img_width
#     ymin = box[3] * orig_images[0].shape[0] / img_height
#     xmax = box[4] * orig_images[0].shape[1] / img_width
#     ymax = box[5] * orig_images[0].shape[0] / img_height
#     color = colors[int(box[0])]
#     label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
#     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=2))  
#     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':color, 'alpha':1.0})