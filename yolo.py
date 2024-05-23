import numpy as np
from pathlib import Path
from darknet import darknet
from PIL import Image


class Yolo:
    def __init__(self, weights_path: Path):

        super().__init__()

        self.cfg_file = weights_path.joinpath("yolo.cfg")
        self.data_file = weights_path.joinpath("obj.data")
        self.weights_path = weights_path.joinpath("yolo.weights")

        # objectDetector.initial_trainer_and_model()
        self.dn_binary = "./darknet/darknet"
        self.conf_thresh = 0.0
        self.network, self.class_names, self.class_colors = darknet.load_network(
            str(self.cfg_file), str(self.data_file), str(self.weights_path), 1
        )

        self.width = darknet.network_width(self.network)
        self.height = darknet.network_height(self.network)
        self.image_buffer = darknet.make_image(self.width, self.height, 3)

    def yolo_bbox_2_original(self, bbox_xywh, original_rgb_shape):
        bbox = [
            (bbox_xywh[0] - bbox_xywh[2] / 2),
            (bbox_xywh[1] - bbox_xywh[3] / 2),
            (bbox_xywh[0] + bbox_xywh[2] / 2),
            (bbox_xywh[1] + bbox_xywh[3] / 2),
        ]

        ih, iw = original_rgb_shape
        scale = max(iw / self.width, ih / self.height)

        nw, nh = int(scale * self.width), int(scale * self.height)

        dw, dh = (iw - nw) // 2, (ih - nh) // 2

        bbox[0] = int(bbox[0] * scale + dw)
        bbox[1] = int(bbox[1] * scale + dh)
        bbox[2] = int(bbox[2] * scale + dw)
        bbox[3] = int(bbox[3] * scale + dh)

        return bbox

    def image_detection(self, image):

        original_shape = image.shape[:2]
        image_resized = self.resize_image_with_aspect(image, (self.width, self.height))

        darknet.copy_image_from_bytes(self.image_buffer, image_resized.tobytes())
        detections = darknet.detect_image(
            self.network, self.class_names, self.image_buffer, thresh=0.01
        )

        rescaled = []
        for label, confidence, bbox in detections:
            bbox = self.yolo_bbox_2_original(bbox, original_shape)
            rescaled.append((label, confidence, bbox))

        return rescaled

    def resize_image_with_aspect(self, image, target_size):
        """
        image [0, 1]
        rescale an image to target_size
        """
        ih, iw = target_size
        h, w, _ = image.shape
        scale = min(iw / w, ih / h)
        nw, nh = int(scale * w), int(scale * h)
        dw, dh = (iw - nw) // 2, (ih - nh) // 2

        image_resized = Image.fromarray(image).resize((nw, nh))
        image_paded = Image.new("RGB", (iw, ih), (128, 128, 128))
        image_paded.paste(image_resized, (dw, dh))
        image_paded = np.array(image_paded)

        return image_paded
