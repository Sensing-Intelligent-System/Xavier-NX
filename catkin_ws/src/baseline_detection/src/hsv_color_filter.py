#!/usr/bin/env python3

# Imports
import rospy
import numpy as np
import cv2  # OpenCV module

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class HSV_color_filter():
    def __init__(self):

        self.cv_bridge = CvBridge()

        # Subscriber image
        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.rgb_callback)

        ## Publisher for predict result and mask
        self.hsv_result = rospy.Publisher("/predict/hsv_result", Image, queue_size=10)

    def rgb_callback(self, data):

        cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")

        blurred_image = cv2.GaussianBlur(cv_image, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2HSV)

        #detect yellow
        lower_yellow = np.array([10, 43, 46])
        upper_yellow = np.array([50, 255, 255])
        mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
        
        # detect while
        lower_white = np.array([0, 0, 20])
        upper_white = np.array([180, 50, 153])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)

        # detect blue
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

        mask = mask_yellow + mask_white + mask_blue

        result = cv2.bitwise_and (cv_image, cv_image, mask=mask)	

        self.hsv_result.publish(self.cv_bridge.cv2_to_imgmsg(result, encoding="passthrough"))

if __name__=="__main__":
    rospy.init_node("hsv_color_filter_node")
    HSV_color_filter = HSV_color_filter()
    rospy.spin()