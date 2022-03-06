from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue

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

classes = ["obstalce","human","injury"]
camera_frame_name = "camera_depth_optical_frame"

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="./cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="./cfg/coco.data",
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))


def convert2relative(bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h  = bbox
    _height     = darknet_height
    _width      = darknet_width
    return x/_width, y/_height, w/_width, h/_height


def convert2original(image, bbox):
    x, y, w, h = convert2relative(bbox)
    image_h, image_w, __ = image.shape

    orig_x       = int(x * image_w)
    orig_y       = int(y * image_h)
    orig_width   = int(w * image_w)
    orig_height  = int(h * image_h)

    bbox_converted = (orig_x, orig_y, orig_width, orig_height)

    return bbox_converted


def convert4cropping(image, bbox):
    x, y, w, h = convert2relative(bbox)

    image_h, image_w, __ = image.shape

    orig_left    = int((x - w / 2.) * image_w)
    orig_right   = int((x + w / 2.) * image_w)
    orig_top     = int((y - h / 2.) * image_h)
    orig_bottom  = int((y + h / 2.) * image_h)

    if (orig_left < 0): orig_left = 0
    if (orig_right > image_w - 1): orig_right = image_w - 1
    if (orig_top < 0): orig_top = 0
    if (orig_bottom > image_h - 1): orig_bottom = image_h - 1

    bbox_cropping = (orig_left, orig_top, orig_right, orig_bottom)

    return bbox_cropping


def video_capture(frame, frame_queue, darknet_image_queue):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_resized = cv2.resize(frame_rgb, (darknet_width, darknet_height),
                                interpolation=cv2.INTER_LINEAR)
    frame_queue.put(frame)
    img_for_detect = darknet.make_image(darknet_width, darknet_height, 3)
    darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
    darknet_image_queue.put(img_for_detect)



def inference(darknet_image_queue, detections_queue, fps_queue):
    darknet_image = darknet_image_queue.get()
    prev_time = time.time()
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
    detections_queue.put(detections)
    fps = int(1/(time.time() - prev_time))
    fps_queue.put(fps)
    print("FPS: {}".format(fps))
    darknet.print_detections(detections, False)
    darknet.free_image(darknet_image)


def drawing(frame_queue, detections_queue, fps_queue):
    frame = frame_queue.get()
    detections = detections_queue.get()
    fps = fps_queue.get()
    detections_adjusted = []
    number_of_obstacle = 0
    number_of_human = 0
    number_of_injury = 0
    box_coordinates = []
    for label, confidence, bbox in detections:
        bbox_adjusted = convert2original(frame, bbox)
        detections_adjusted.append((str(label), confidence, bbox_adjusted))
        box_coordinates.append(bbox_adjusted)
        if str(label)=="obstacle":
            number_of_obstacle=number_of_obstacle+1
        elif str(label)=="human":
            number_of_human=number_of_human+1
        elif str(label)=="injury":
            number_of_injury=number_of_injury+1
    number_of_object_detected = len(detections_adjusted)
    image = darknet.draw_boxes(detections_adjusted, frame, class_colors)
    image = cv2.putText(image, "FPS: %.2f"%(fps), (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image = cv2.putText(image, "obstacle= {}".format(number_of_obstacle), (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image = cv2.putText(image, "human= {}".format(number_of_human), (0, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    image = cv2.putText(image, "injury= {}".format(number_of_injury), (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Inference', image)
    cv2.waitKey(1)

    return number_of_obstacle,number_of_human,number_of_injury,box_coordinates

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



if __name__ == '__main__':
    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)
    detection = object_detect()

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
            args.config_file,
            args.data_file,
            args.weights,
            batch_size=1
        )

    darknet_width = darknet.network_width(network)
    darknet_height = darknet.network_height(network)
    video_width = detection.rgb_width
    video_height = detection.rgb_height
    while(True):
        rospy.wait_for_message("/camera/color/image_raw",Image)
        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw",Image)
        video_capture(detection.rgb_image, frame_queue, darknet_image_queue)
        inference(darknet_image_queue, detections_queue, fps_queue)
        number_of_obstacle,number_of_human,number_of_injury,box_coordinates=drawing(frame_queue, detections_queue, fps_queue)
        detection.number_of_obstacle_pub.publish(number_of_obstacle)
        detection.number_of_human_pub.publish(number_of_human)
        detection.number_of_injury_pub.publish(number_of_injury)
        box_array = BoundingBoxArray()
        for i in range(len(box_coordinates)):
            box = BoundingBox()
            center_x = box_coordinates[i][0]
            center_y = box_coordinates[i][1]
            box_width = box_coordinates[i][2]
            box_height = box_coordinates[i][3]
            top_left_x = abs(center_x-box_width/2)
            top_left_y = abs(center_y-box_height/2)
            bottom_right_x = abs(center_x+box_width/2)
            bottom_right_y = abs(center_y+box_height/2)

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