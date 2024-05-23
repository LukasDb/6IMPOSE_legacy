import simpose as sp

from pathlib import Path
import requests
import cv2
import base64
import io
import numpy as np


endpoint = "http://localhost:5000/predict"

intrinsics = np.array(
    [
        [1350.0448696056146, 0.0, 969.0358916438173],
        [0.0, 1350.0448696056146, 542.0742335509645],
        [0.0, 0.0, 1.0],
    ]
)

img = cv2.imread("test3.jpg")  # BGR

# raw_bytes = open(test_img_path, "rb").read()
rgb_raw_bytes = cv2.imencode(".jpg", img)[1].tobytes()
rgb_encoded_bytes = base64.b64encode(rgb_raw_bytes)


# load a depth image
# memfile = io.BytesIO()
# np.save(memfile, depth)
# depth_encoded_bytes = base64.b64encode(memfile.getvalue())

# memfile = io.BytesIO()
# np.save(memfile, intrinsics)
# intrinsics_encoded_bytes = base64.b64encode(memfile.getvalue())

# POST to endpoint
response = requests.post(
    endpoint,
    files={
        "rgb": rgb_encoded_bytes,
        # "depth": depth_encoded_bytes,
        # "intrinsic": intrinsics_encoded_bytes,
    },
)

# Print the JSON response
detections = response.json()

img = np.array(img[:, :, ::-1])  # BGR to RGB
for detection in detections:
    cls = detection["cls"]
    bbox = detection["bbox"]
    bbox = [int(x) for x in bbox]
    confidence = float(detection["confidence"])
    if confidence < 20:
        continue

    print(f"{cls = } {bbox = } {confidence = }")
    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    cv2.putText(
        img,
        f"{cls} {confidence:.2f}",
        (bbox[0], bbox[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )

    try:
        affine_matrix = np.array(detection["pose"])

        rot = affine_matrix[:3, :3]
        t = affine_matrix[:3, 3]

        rvec = cv2.Rodrigues(rot)
        rvec = rvec[0].flatten()
        tvec = t.flatten()

        print(f"{rvec = } {tvec = }")
        # draw frame
        cv2.drawFrameAxes(img, intrinsics, np.zeros((4, 1)), rvec, tvec, 0.1, 3)  # type: ignore
    except KeyError:
        pass

cv2.imshow("img", img[:, :, ::-1])
cv2.waitKey(0)
