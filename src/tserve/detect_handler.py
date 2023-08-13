import os
from ultralytics import YOLO
import torch
import cv2
import base64
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
import pickle as pkl
import numpy as np
import base64
import json


IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelHandler(object):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
    
    def initialize(self, context):
        print("[DETECT][initialize] START")
        self.manifest = context.manifest

        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        # self.model = torch.jit.load(model_pt_path)
        # TODO: 
        #   this implementation can be not fast enough
        #   for more performance try https://github.com/louisoutin/yolov5_torchserve
        self.model = YOLO(
            model_pt_path,  # "../weights/detect.torchscript"
            task="detect"
        )
        self.initialized = True
        print("[DETECT][initialize] END")
        

    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        print("[DETECT][handle] START")
        byte_array = data[0]["body"]
        file_bytes = np.asarray(bytearray(byte_array), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        print(f'handle: {type(data[0]["body"])=}')
        pred_out = self.model(img)
        print(f"{type(pred_out)=}")
        print(f"{len(pred_out)=}")
        print(f"{type(pred_out[0])=}")
        # boxes = [[int(x) for x in box] for box in pred_out[0].boxes.data.tolist()]
        # boxes = pred_out[0].boxes.data.tolist()
        boxes = [[int(x) for x in box] for box in pred_out[0].boxes.data.tolist()]

        print("[DETECT][handle] END")
        # result = base64.urlsafe_b64encode(
        #     json.dumps(
        #         {"boxes": boxes, "image": img.tolist()}
        #     ).encode()
        # ).decode()
        result = base64.urlsafe_b64encode(
            json.dumps(
                {"boxes": boxes, "image": data[0]["body"].decode("latin-1")}
            ).encode()
        ).decode()
        return [result]
