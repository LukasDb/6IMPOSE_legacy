import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from flask import Flask, request, jsonify
import subprocess

import numpy as np
import os
import base64
from PIL import Image
import io
from pathlib import Path

import tensorflow as tf

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

from pvn3d import PVN3D
from yolo import Yolo


app = Flask(__name__)


def unpack(file):
    decoded = base64.b64decode(file.read())
    return io.BytesIO(decoded)


# simple main page
@app.route("/")
def main():
    return subprocess.run(["bash -c nvidia-smi"], shell=True, capture_output=True).stdout.decode(
        "utf-8"
    )


@app.route("/predict", methods=["POST"])
def predict():
    print("Received POST request")
    # Get the image file from the request
    image_file = request.files["rgb"]

    # convert it into bytes
    img = np.asarray(Image.open(unpack(image_file)))

    detections = object_detector.image_detection(img)

    detections = [
        {
            "cls": x[0],
            "confidence": float(
                x[1],
            ),
            "bbox": x[2],
        }
        for x in detections
    ]

    detections = [x for x in detections if x["cls"] != "wrench_13"]

    depth_file = request.files.get("depth", None)
    intrinsics_file = request.files.get("intrinsic", None)
    if depth_file and intrinsics_file:
        intrinsics = np.load(unpack(intrinsics_file))
        depth = np.load(unpack(depth_file))

        for detection in detections:
            box = detection["bbox"]
            cls = detection["cls"]
            detected, affine_matrix = pose_detector.inference(cls, box, img, depth, intrinsics)
            if not detected:
                continue
            detection.update({"pose": affine_matrix.tolist()})

    return jsonify(detections)


# TODO have endpoints only for first|second stage or deal with with arguments of the POST request

if __name__ == "__main__":
    object_detector = Yolo(Path("weights/darknet/cps/"))
    pose_detector = PVN3D()
    print("Serving 6IMPOSE ...")
    app.run(host="0.0.0.0", debug=False)
