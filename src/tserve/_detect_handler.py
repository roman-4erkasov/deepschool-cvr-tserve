import torch
from torchvision import transforms
from ts.torch_handler.vision_handler import VisionHandler

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
        

    
    def handle(self, data, context):
        """
        Invoke by TorchServe for prediction request.
        Do pre-processing of data, prediction using model and postprocessing of prediciton output
        :param data: Input data for prediction
        :param context: Initial context contains model server system properties.
        :return: prediction output
        """
        pred_out = self.model(data)
        return pred_out
