import os
import re
import json
import torch
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler
import numpy as np
import cv2
import base64
import json
from torchvision.transforms import Resize

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class ModelHandler(object):

    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
        self.resize = Resize(size=(280, 523))
    
    def initialize(self, context):
        print("[OCR][initialize] START")
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        self.device = torch.device("cpu")

        # Read model serialize/pt file
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = os.path.join(model_dir, serialized_file)
        if not os.path.isfile(model_pt_path):
            raise RuntimeError("Missing the model.pt file")

        self.model = torch.jit.load(model_pt_path)

        vocab_path = os.path.join(model_dir, "vocabulary.json")
        with open(vocab_path) as fp:
            tokens = json.load(fp=fp)
        self.idx2tok = {idx: tok for idx, tok in enumerate(tokens)}
        self.tok2idx = {tok: idx for idx, tok in enumerate(tokens)}
        # TODO: 
        #   this implementation can be not fast enough
        #   for more performance try https://github.com/louisoutin/yolov5_torchserve
        # self.model = YOLO(
        #     model_pt_path,  # "../weights/detect.torchscript"
        #     task="detect"
        # )
        self.initialized = True
        print("[OCR][initialize] END")
        

    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        print(f"[OCR][handle] START: {context=}")
        # data = json.loads(base64.urlsafe_b64decode(data.decode(.encode()).decode())
        print(f"[OCR][handle] {data[0].keys()=}")
        data = base64.urlsafe_b64decode(data[0]["body"])
        data = data.decode()
        data = json.loads(data)
        
        print(f"[OCR][handle] {type(data)=}")
        print(f"[OCR][handle] {len(data)=}")
        # print(f"[OCR][handle] {type(data[0])=}")
        print(f"[OCR][handle] {data.keys()=}")
        boxes = data["boxes"]
        print(f"[OCR][handle] {boxes=}")
        byte_array = data["image"].encode("latin-1")
        file_bytes = np.asarray(bytearray(byte_array), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image = np.asarray(img)
        result = []
        # boxes = [[int(x) for x in box] for box in data[0].boxes.data.tolist()]
        for box in boxes:
            box_image = image[box[1]:box[3], box[0]:box[2],:]
            box_image = box_image / 255.0
            box_image = torch.FloatTensor(box_image).permute(2, 0, 1)
            box_image=self.resize(box_image)
            print(f"[OCR][handle] {box_image.shape=}")
            if 3 == len(box_image.shape):
                box_image = box_image.unsqueeze(0)
            pred = self.model(box_image).argmax(2).T.squeeze().tolist()
            chars = []
            for idx in pred:
                c = self.idx2tok[idx]
                chars.append(c)
            string = re.sub(
                r'([0-9\_])\1+', 
                r'\1', 
                "".join(chars)
            ).replace("_","")
            result.append(
                {
                    "bbox": {
                        "x_min": min(box[0],box[2]),
                        "x_max": max(box[0],box[2]),
                        "y_min": min(box[1],box[3]),
                        "y_max": max(box[1],box[3]),
                    },
                    "value": string
                }
            )
        print("[OCR][handle] END")
        # return json.dumps({"barcodes": result})
        return [{"barcodes": result}]
