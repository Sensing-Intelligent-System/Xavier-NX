#!/usr/bin/env python

# Imports
import numpy as np
import os
from IPython.display import display

import cv2
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

import torchvision
# from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
# from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as Tf

import torch
import torch.utils.data
import gdown
import wget

import matplotlib.patches as patches
import random

class Detection():
    def __init__(self):

        self.cv_bridge = CvBridge()

        self.ycb = [
            "003_cracker_box.sdf", "004_sugar_box.sdf", "005_tomato_soup_can.sdf",
            "006_mustard_bottle.sdf", "009_gelatin_box.sdf", "010_potted_meat_can.sdf"
        ]

        # self.download_model()

        # Subscriber image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)

        ## Publisher for predict result and mask
        self.result = rospy.Publisher("/predict/result", Image, queue_size=10)
        self.mask = rospy.Publisher("/predict/mask", Image, queue_size=10)

    def rgb_callback(self, data):

        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")

        # num_classes = len(self.ycb)+1
        # model = self.get_instance_segmentation_model(num_classes)
        # model.load_state_dict(torch.load('clutter_maskrcnn_model.pt'))
        # model.eval()

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # model.to(device)

        # with torch.no_grad():
        #     prediction = model([Tf.to_tensor(cv_image).to(device)])

        # # mask
        # for i in range(prediction[0]["masks"].shape[0]):
        #     mask = (255 * np.array(prediction[0]['masks'][i,0].cpu().numpy())).astype('uint8') if i==0 else mask + (255 * np.array(prediction[0]['masks'][i,0].cpu().numpy())).astype('uint8')

        # self.mask.publish(cv_bridge.cv2_to_imgmsg(mask, encoding="passthrough"))

        # object detection
        # img_np = np.array(cv_image)
        # plt.figure()
        # fig, ax = plt.subplots(1, figsize=(12,9))
        # ax.imshow(img_np)

        # cmap = plt.get_cmap('tab20b')
        # colors = [cmap(i) for i in np.linspace(0, 1, 20)]

        # num_instances = prediction[0]['boxes'].shape[0]
        # bbox_colors = random.sample(colors, num_instances)
        # boxes = prediction[0]['boxes'].cpu().numpy()
        # labels = prediction[0]['labels'].cpu().numpy()

        # for i in range(num_instances):
        #     color = bbox_colors[i]
        #     bb = boxes[i,:]
        #     bbox = patches.Rectangle((bb[0], bb[1]), bb[2]-bb[0], bb[3]-bb[1],
        #             linewidth=2, edgecolor=color, facecolor='none')
        #     ax.add_patch(bbox)
        #     plt.text(bb[0], bb[0], s=self.ycb[labels[i]], 
        #             color='white', verticalalignment='top',
        #             bbox={'color': color, 'pad': 0})
        # plt.axis('off')

        self.result.publish(self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    
    def download_model(self):
        model_file = 'clutter_maskrcnn_model.pt'

        if not os.path.exists(model_file):
            wget.download("https://groups.csail.mit.edu/locomotion/clutter_maskrcnn_model.pt", out=model_file)
        
        print("download finished")


    def get_instance_segmentation_model(self, num_classes):
        # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

        # get the number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                        hidden_layer,
                                                        num_classes)

        return model

if __name__=="__main__":
    rospy.init_node("detection_node")
    detection = Detection()
    rospy.spin()