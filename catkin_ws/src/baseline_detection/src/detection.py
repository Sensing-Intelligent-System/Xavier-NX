#!/usr/bin/env python3

# Imports
import numpy as np
import os

import cv2
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as Tf

import torch
import torch.utils.data
import wget
import rospy
import copy

class Detection():
    def __init__(self):

        self.cv_bridge = CvBridge()

        self.ycb = [
            "cracker_box", "sugar_box", "tomato_soup_can",
            "mustard_bottle", "gelatin_box", "potted_meat_can"
        ]

        self.download_model()

        # Subscriber image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)

        ## Publisher for predict result and mask
        self.result = rospy.Publisher("/predict/result", Image, queue_size=10)
        self.mask = rospy.Publisher("/predict/mask", Image, queue_size=10)

    def rgb_callback(self, data):

        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")

        num_classes = len(self.ycb)+1
        model = self.get_instance_segmentation_model(num_classes)
        model.load_state_dict(torch.load('clutter_maskrcnn_model.pt'))
        model.eval()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model.to(device)

        with torch.no_grad():
            prediction = model([Tf.to_tensor(cv_image).to(device)])

        # mask
        for i in range(prediction[0]["masks"].shape[0]):
            mask = (255 * np.array(prediction[0]['masks'][i,0].cpu().numpy())).astype('uint8') if i==0 else mask + (255 * np.array(prediction[0]['masks'][i,0].cpu().numpy())).astype('uint8')

        self.mask.publish(self.cv_bridge.cv2_to_imgmsg(mask, encoding="passthrough"))

        # object detection
        img = cv_image.copy()

        num_instances = prediction[0]['boxes'].shape[0]
        boxes = prediction[0]['boxes'].cpu().numpy().astype('int16')
        labels = prediction[0]['labels'].cpu().numpy()

        for i in range(num_instances):
            bb = boxes[i,:]
            cv2.rectangle(img, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)

            cv2.putText(img, self.ycb[labels[i]], (bb[0], bb[1]), cv2.FONT_HERSHEY_TRIPLEX,0.5, (0, 255, 255), 1, cv2.LINE_AA)

        self.result.publish(self.cv_bridge.cv2_to_imgmsg(img, "bgr8"))

    
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