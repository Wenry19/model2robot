#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge

from sensor_msgs.msg import Image
from model2robot_msgs.msg import Prediction


import torch
from PIL import Image as PILImage
import importlib

from model2robot.inference.pytorch_inference import PyTorchInference
from model2robot.inference.tensorrt_inference import TensorRTInference

from model2robot.utils import read_config_file, find_device

class InferenceNode(Node):

    def __init__(self):
        super().__init__("inference_node")

        #### CONFIGURATION FILE ####

        self.declare_parameter("config_path", "")
        config_path = self.get_parameter("config_path").value
        if not config_path:
            raise ValueError("Parameter 'config_path' must be provided")
        self.config = read_config_file(config_path)

        #### INFERENCE INSTANCE  ####

        self.device = find_device()

        if self.config["inference"]["backend"] == "pytorch":
            model_instance = self.instantiate_class(self.config["model"]["module"],
                                                    self.config["model"]["class"],
                                                    **self.config["model"].get("args", {}))
            model_instance.to(self.device)
            self.inference = PyTorchInference(model_path=self.config["model"]["path"],
                                              model_instance=model_instance,
                                              device=self.device)
        elif self.config["inference"]["backend"] == "tensorrt":
            self.inference = TensorRTInference(model_path=self.config["model"]["path"],
                                               batch_size=1,
                                               input_height=self.config["input"]["transform"]["args"]["input_height"],
                                               input_width=self.config["input"]["transform"]["args"]["input_width"])
        else:
            raise ValueError(f"Unsupported backend: {self.config['inference']['backend']}")

        #### TRANSFORM INSTANCE  ####

        self.transform = self.import_function(self.config["input"]["transform"]["module"],
                                              self.config["input"]["transform"]["function"],
                                              **self.config["input"]["transform"].get("args", {}))

        #### SUBSCRIBER ####

        self.image_sub = self.create_subscription(Image,
                                                  "/image",
                                                  self.image_callback,
                                                  10)

        #### PUBLISHER ####

        self.predict_pub = self.create_publisher(Prediction,
                                                 self.config["output"]["topic"],
                                                 10)

        self.bridge = CvBridge()

        self.get_logger().info("InferenceNode has been created!")

    def image_callback(self, img: Image):

        self.get_logger().info(f"Received image: {img.width}x{img.height}, "
                               f"encoding={img.encoding}")

        preprocessed_img = self.preprocess_image(img)

        self.inference.run(preprocessed_img)
        outputs = self.inference.get_output()

        probabilities = torch.softmax(outputs, dim=1)
        prediction = torch.argmax(outputs, dim=1)[0]
        confidence = probabilities[0, prediction]

        class_id = int(prediction)
        
        msg = Prediction()
        msg.class_id = class_id
        msg.class_name = self.config["output"]["class_names"][class_id]
        msg.confidence = float(confidence)

        self.predict_pub.publish(msg)

    def preprocess_image(self, img: Image):

        # ROS Image -> NumPy array in RGB format
        preprocessed_img = self.bridge.imgmsg_to_cv2(
            img,
            desired_encoding=self.config["input"]["encoding"]
        )

        # NumPy array -> PIL Image
        preprocessed_img = PILImage.fromarray(preprocessed_img)

        # Same preprocessing used during training
        preprocessed_img = self.transform(preprocessed_img)

        # Add batch dimension
        preprocessed_img = preprocessed_img.unsqueeze(0)

        return preprocessed_img

    def instantiate_class(self, module_name, class_name, **kwargs):

        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)

        return cls(**kwargs)

    def import_function(self, module_name, function_name, **kwargs):

        module = importlib.import_module(module_name)
        function = getattr(module, function_name)

        return function(**kwargs)

    def close_inference(self):
        if self.inference is not None:
            self.inference.close()

def main(args=None):

    # initialize ROS2 communications
    rclpy.init(args=args)

    node = InferenceNode()
    # in order to keep the node alive, so it can execute the callbacks
    rclpy.spin(node)

    node.close_inference()
    # shutdown ROS2 communications
    rclpy.shutdown()

# Nodes used to publish images
# ros2 run image_tools cam2image --ros-args -p burger_mode:=true
# ros2 run image_tools cam2image
# ros2 run image_tools showimage
