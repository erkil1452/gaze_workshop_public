"""
This file contains solution of the small assignments in the notebooks.
You can use it to copy paste solution and move on to the next section.

Petr Kellnhofer, 2022.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import cv2
import imageio
import dlib
import nbimporter

import gaze_utils as gu

# Lecture 1

def capture_frame() -> np.array:
    """ Captures and returns a frame from a camera. """
    camera_index = 0
    camera_resolution = [1280, 720]
    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_resolution[1])
    # Capture frame from camera.
    res, frame = cap.read()
    # Release camera.
    cap.release()
    # Left <-> right mirror (if needed).
    frame = np.fliplr(frame)
    # Color space conversion BGR -> RGB.
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

def detect_face(image: np.array) -> dlib.rectangle:
    """ Detects one face in an image and returns it rectangle. """
    # Color to gray.
    frame_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Run detector.
    face_detector = dlib.get_frontal_face_detector()
    faces = face_detector(frame_gray, 1)

    # Find largest face
    max_area = 0
    face = None
    for rect in faces:
        if rect.area() > max_area:
            max_area = rect.area()
            face = rect

    return face

def crop_image(image: np.array, rectangle: dlib.rectangle, scale_factor: float = 1.2) -> float:
    """ Return a rectangular crop of the image. Enlarge the size of the rectangle by a factor. """
    # 1. Compute left and right corner coordinates.
    face_size = max(rectangle.width(), rectangle.height()) * scale_factor
    lt = np.array([rectangle.center().x - face_size / 2, rectangle.center().y - face_size / 2], int)
    rb = np.array([rectangle.center().x + face_size / 2, rectangle.center().y + face_size / 2], int) + 1

    # 2. Determine how much is out of bounds.
    image_size = np.array([image.shape[1], image.shape[0]]) # width, height
    lt_clipped = np.clip(lt, 0, image_size)
    rb_clipped = np.clip(rb, lt, image_size)
    crop = image[lt_clipped[1]:rb_clipped[1], lt_clipped[0]:rb_clipped[0]]

    # 3. Add zero padding for the out-of-bounds region.
    pad_left = np.maximum(lt_clipped - lt, 0)
    pad_right = np.maximum(rb - rb_clipped, 0)
    padded = np.pad(crop, [(pad_left[1], pad_right[1]), (pad_left[0], pad_right[0]), (0,0)])
    return padded

# Lecture 3

def numpy_image_to_tensor(im: np.array, input_size: tuple = (127, 127)) -> torch.Tensor:
    """ Converts a numpy image to network input tensor of a fixed size input_size. """
    # 1. Resize to expected size.
    im = cv2.resize(im, tuple(np.array(input_size).astype(int)), interpolation=cv2.INTER_AREA)
    # 2. Convert to float32 and normalize to [-1,1].
    im = im.astype(np.float32) / 255 * 2 - 1
    # 3. Convert to torch.tensor.
    net_input = torch.from_numpy(im)
    # 4. Reshape from HxWxC to CxHxW
    net_input = net_input.permute(2,0,1)
    return net_input