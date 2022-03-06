"""trt_yolo.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO

# ros package
import rospy
from sensor_msgs import point_cloud2
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import String
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Int32
from PIL import Image as PIL_Image
import cv_bridge
import numpy as np

WINDOW_NAME = 'YOLOv4-tiny TensorRT detection'
camera_frame_name = "camera_depth_optical_frame"

def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    args = parser.parse_args()
    return args


def loop_and_detect(trt_yolo, conf_th, vis):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
    """
    while(True):
        rospy.wait_for_message("/camera/color/image_raw",Image)
        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw",Image)
        number_of_obstacle = 0
        number_of_human = 0
        number_of_injury = 0
        start = time.time()
        img = detection.rgb_image
        boxes, confs, clss = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confs, clss)
        number_of_obstacle = np.count_nonzero(clss == 0.0)
        number_of_human = np.count_nonzero(clss == 1.0)
        number_of_injury = np.count_nonzero(clss == 2.0)
        img = cv2.putText(img, "obstacle= {}".format(number_of_obstacle), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img = cv2.putText(img, "human= {}".format(number_of_human), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        img = cv2.putText(img, "injury= {}".format(number_of_injury), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        detection.number_of_obstacle_pub.publish(number_of_obstacle)
        detection.number_of_human_pub.publish(number_of_human)
        detection.number_of_injury_pub.publish(number_of_injury)
        box_array = BoundingBoxArray()

        for i in range(len(boxes)):
            box = BoundingBox()
            top_left_x = boxes[i][0]
            top_left_y = boxes[i][1]
            bottom_right_x = boxes[i][2]
            bottom_right_y = boxes[i][3]
            center_x = int((bottom_right_x+top_left_x)/2)
            center_y = int((bottom_right_y+top_left_y)/2)
            box_position_z,cx,cy,x1,y1,x2,y2 = detection.transformation(top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y)
            box_size_x,box_size_y,box_size_z,box_position_x,box_position_y = detection.box_calculation(cx,cy,x1,y1,x2,y2)
            box.header.stamp = rospy.Time.now()
            box.header.frame_id = camera_frame_name
            box.pose.orientation.w =  1
            box.pose.position.x = box_position_x # increase in program  = move to right in rviz 
            box.pose.position.y = box_position_y # increase in program  = downward in rviz
            box.pose.position.z = box_position_z # increase in program  = move away in rviz (directly use the depth distance)
            box.dimensions.x = box_size_x
            box.dimensions.y = box_size_y
            box.dimensions.z = box_size_z
            box_array.boxes.append(box)
        box_array.header.frame_id = camera_frame_name
        box_array.header.stamp = rospy.Time.now()
        detection.box_array_pub.publish(box_array)

        curr_fps = 1.0 / (time.time() - start)
        img = cv2.putText(img, "FPS: %.2f"%(curr_fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, img)
        cv2.waitKey(1)


class object_detect:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        rospy.init_node('object_detection_tensorrt')
        
        # Subscribe color and depth image
        rospy.Subscriber("/camera/color/image_raw",Image,self.color_callback)
        rospy.Subscriber("/camera/aligned_depth_to_color/image_raw",Image,self.depth_callback)
        #rospy.Subscriber("/camera/depth/color/points",PointCloud2,self.pointcloud2_callback)

        # Subscribe camera info
        rospy.Subscriber("/camera/depth/camera_info",CameraInfo,self.depth_camera_info_callback)
        rospy.Subscriber("/camera/color/camera_info",CameraInfo,self.color_camera_info_callback)

        #self.box_pub = rospy.Publisher("/desired/input/box", BoundingBox, queue_size=1)
        self.box_array_pub = rospy.Publisher("/desired/input/box_array", BoundingBoxArray, queue_size=1)

        self.number_of_obstacle_pub  = rospy.Publisher("/detection_result/number_of_obstacle", Int32, queue_size=1)
        self.number_of_human_pub  = rospy.Publisher("/detection_result/number_of_human", Int32, queue_size=1)
        self.number_of_injury_pub  = rospy.Publisher("/detection_result/number_of_injury", Int32, queue_size=1)

    def depth_callback(self,data):
        # Depth image callback
        self.depth_image = self.bridge.imgmsg_to_cv2(data)
        self.depth_image = np.array(self.depth_image, dtype=np.float32)
        self.depth_array = self.depth_image/1000.0

    def color_callback(self,data):
        # RGB image callback
        self.rgb_image = self.bridge.imgmsg_to_cv2(data, "bgr8")


    # this is depth camera info callback
    def depth_camera_info_callback(self, data):
        self.depth_height = data.height
        self.depth_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.depth_u = data.P[2]
        self.depth_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.depth_fx = data.P[0]
        self.depth_fy = data.P[5]

    # this is color camera info callback
    def color_camera_info_callback(self, data):
        self.rgb_height = data.height
        self.rgb_width = data.width

        # Pixels coordinates of the principal point (center of projection)
        self.rgb_u = data.P[2]
        self.rgb_v = data.P[6]

        # Focal Length of image (multiple of pixel width and height)
        self.rgb_fx = data.P[0]
        self.rgb_fy = data.P[5]


    def transformation(self,top_left_x, top_left_y, bottom_right_x, bottom_right_y, center_x, center_y):
        
        if (top_left_x==0)&(top_left_y==0)&(bottom_right_x==0)&(bottom_right_y==0):
            expected_3d_center_distance=0.0
            expected_3d_center_x=0.0
            expected_3d_center_y=0.0
            expected_3d_top_left_x=0.0
            expected_3d_top_left_y=0.0
            expected_3d_bottom_right_x=0.0
            expected_3d_bottom_right_y=0.0

        distance_z = self.depth_array[center_y,center_x]
        expected_3d_center_distance = distance_z
        expected_3d_top_left_x = ((top_left_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_top_left_y = ((top_left_y - self.rgb_v)*distance_z)/self.rgb_fy

        expected_3d_bottom_right_x = ((bottom_right_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_bottom_right_y = ((bottom_right_y - self.rgb_v)*distance_z)/self.rgb_fy

        expected_3d_center_x = ((center_x - self.rgb_u)*distance_z)/self.rgb_fx
        expected_3d_center_y = ((center_y - self.rgb_v)*distance_z)/self.rgb_fy

        return expected_3d_center_distance,expected_3d_center_x,expected_3d_center_y, expected_3d_top_left_x, expected_3d_top_left_y, expected_3d_bottom_right_x, expected_3d_bottom_right_y


    def box_calculation(self,center_x,center_y,top_left_x,top_left_y,bottom_right_x,bottom_right_y):
        box_dimensions_x = abs(top_left_x)-abs(bottom_right_x)
        if (top_left_x < 0)&(bottom_right_x > 0):
            box_dimensions_x = abs(top_left_x)+abs(bottom_right_x)
        box_dimensions_x = abs(box_dimensions_x)

        box_dimensions_y = abs(top_left_y)-abs(bottom_right_y)
        if (top_left_y < 0)&(bottom_right_y > 0):
            box_dimensions_y = abs(top_left_y)+abs(bottom_right_y)
        box_dimensions_y = abs(box_dimensions_y)

        box_position_x = center_x
        box_position_y = center_y
        box_dimension_z = 1

        return box_dimensions_x,box_dimensions_y,box_dimension_z,box_position_x,box_position_y




def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('yolo/%s.trt' % args.model):
        raise SystemExit('ERROR: file (yolo/%s.trt) not found!' % args.model)

    cls_dict = {0: 'obstacle', 1: 'human', 2: 'injury'}
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    open_window(WINDOW_NAME, WINDOW_NAME,detection.rgb_width, detection.rgb_height)
    loop_and_detect(trt_yolo, args.conf_thresh, vis=vis)




if __name__ == '__main__':
    detection = object_detect()
    main()