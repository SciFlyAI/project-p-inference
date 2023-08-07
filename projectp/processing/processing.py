import math as m

from functools import partial

import cv2 as cv
import numpy as np

from matplotlib import cm, colors, pyplot as plt

from ..utils import LogStub


log = LogStub()


class Tile:
    def __init__(self, x1, y1, x2, y2, bboxes=None):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.w = x2 - x1
        self.h = y2 - y1
        self.slice = np.s_[y1:y2, x1:x2, ...]
        self.bboxes = bboxes


def get_tiles(image, size_crop, log=log, verbose=False):
    width, height = image.shape[1::-1]

    assert (
        isinstance(size_crop, (int, float)) or
        isinstance(size_crop, (dict, list, set, tuple)) and
        len(size_crop) == 2
    ), (f"Size_crop must be a single value or a collection of two values,"
        f" got {type(size_crop)} = {size_crop}!")
    if isinstance(size_crop, (dict, list, set, tuple)):
        size_crop = tuple(size_crop)[:2]
    else:
        size_crop = (size_crop, size_crop)

    assert width >= size_crop[0] and height >= size_crop[1], (
        f"One or more image dimensions less than crop size,"
        f" got image {(width, height)} vs crop {size_crop})!"
    )

    ratio_x, ratio_y = size_crop[0] / width, size_crop[1] / height

    num_patches_x = m.ceil(1 / ratio_x)
    num_patches_y = m.ceil(1 / ratio_y)
    tiles = np.array(
        [None] * num_patches_x * num_patches_y
    ).reshape((num_patches_y, num_patches_x))
    if verbose:
        log.info(f"Size = {(width, height)},"
                 f" num patches = {(num_patches_x, num_patches_y)},"
                 f" tiles.shape = {tiles.shape}")
    offset_x = 0
    for i in range(num_patches_x):
        offset_y = 0
        for j in range(num_patches_y):
            x1 = offset_x
            y1 = offset_y
            x2 = offset_x + size_crop[0]
            y2 = offset_y + size_crop[1]
            tiles[j, i] = Tile(x1, y1, x2, y2)
            if verbose:
                log.info(f"Patch {(i, j)} {(x1, y1, x2, y2)}")
            # Clips relative to image dimensions
            clip_min_x = offset_x / width
            clip_min_y = offset_y / height
            # Clip minimum values (sizes intact)
            clip_min = np.array([clip_min_x, clip_min_y, 0, 0])
            clip_max_x = (offset_x + size_crop[0]) / width
            clip_max_y = (offset_y + size_crop[1]) / height
            # Clip maximum values (sizes intact)
            clip_max = np.array([clip_max_x, clip_max_y, 1, 1])
            # Update y-offset (end of iteration)
            shift_y = (((num_patches_y - j) - (height - offset_y) / size_crop[1]) /
                       (num_patches_y - j - 1 + 1e-15))
            offset_y += round(size_crop[1] - size_crop[1] * shift_y)
        # Update x-offset (end of iteration)
        shift_x = (((num_patches_x - i) - (width - offset_x) / size_crop[0]) /
                   (num_patches_x - i - 1 + 1e-15))
        offset_x += round(size_crop[0] - size_crop[0] * shift_x)
    return tiles


def yolo_to_xyxy(boxes, image, size, offset_x=0.0, offset_y=0.0):
    boxes_norm = boxes.copy()
    limy, limx = image.shape[:2]
    try:
        size_x, size_y = size
    except TypeError:
        # Unable to unpack non-iterable
        size_x, size_y = size, size

    boxes_norm[:, 0] -= boxes_norm[:, 2] / 2  # x1
    boxes_norm[:, 1] -= boxes_norm[:, 3] / 2  # y1

    boxes_norm[:, 2] += boxes_norm[:, 0]  # x2
    boxes_norm[:, 3] += boxes_norm[:, 1]  # y2

    if np.any([offset_x, offset_y]):
        boxes_norm[:, 0:4:2] *= size_x / image.shape[1]  # scale to global size (relative)
        boxes_norm[:, 0:4:2] += offset_x / image.shape[1]  # x1 + offset (relative)
        boxes_norm[:, 0:4:2] *= image.shape[1]  # scale to absolute
        boxes_norm[:, 1:4:2] *= size_y / image.shape[0]  # scale to global size (relative)
        boxes_norm[:, 1:4:2] += offset_y / image.shape[0]  # y1 + offset (relative)
        boxes_norm[:, 1:4:2] *= image.shape[0]  # scale to absolute

    return boxes_norm[(boxes_norm[:, 0] > 0) & (boxes_norm[:, 1] > 0) &
                      (boxes_norm[:, 2] < limx) & (boxes_norm[:, 3] < limy)]


def normalize_boxes(boxes, image, denormalize=False):
    boxes_norm = boxes.copy()
    if not denormalize:
        boxes_norm[:, 0:4:2] /= image.shape[1]  # width -> 0..1
        boxes_norm[:, 1:4:2] /= image.shape[0]  # height -> 0..1
    else:
        boxes_norm[:, 0:4:2] *= image.shape[1]  # width -> 0..image.width
        boxes_norm[:, 1:4:2] *= image.shape[0]  # height -> 0..image.height

    return boxes_norm


def get_colors(detections):
    cmap = plt.get_cmap('gist_rainbow')
    cnorm = colors.Normalize(0, len(detections))

    return cm.ScalarMappable(norm=cnorm, cmap=cmap)


def draw_detections(frame, detections, font_face=cv.FONT_HERSHEY_COMPLEX_SMALL,
                    font_scale=2, font_thickness=3,
                    text_origin_x=None, text_origin_y=None,
                    shape=None):
    scma = get_colors(detections)  # TODO: replace with numpy array

    assert shape in ['box', 'dot'], "Detection shape must be either 'box' or 'dot'!"
    if shape == 'box':
        coords_all = detections
    elif shape == 'dot':
        coords_all = (
                detections[..., 0:2] +
                (detections[..., 2:4] - detections[..., 0:2]) / 2
        )
    else:
        raise "This should never happen!"
    for i, coords in enumerate(coords_all):
        color = (
                np.array(scma.to_rgba(i)[:3]) * 255
        ).round().astype('uint8').tolist()
        if shape == 'box':
            frame = cv.rectangle(frame, coords[:2].round().astype('int'),
                                 coords[2:4].round().astype('int'), color=color,
                                 thickness=2)
        else:
            frame = cv.circle(frame, (coords.round().astype('int')), radius=2,
                              color=color, thickness=2)
    text = f"Detected: {len(detections)}"
    text_size = cv.getTextSize(text, font_face, font_scale, font_thickness)
    text_origin = text_origin_x or 32, text_origin_y or text_size[0][1] * 1.5

    cv.putText(frame, text, tuple(map(lambda x: round(x + 1), text_origin)),
               font_face, font_scale, (0, 0, 0), font_thickness, 8)
    cv.putText(frame, text, tuple(map(lambda x: round(x), text_origin)),
               font_face, font_scale, (255, 255, 255), font_thickness, 8)
    return frame


def get_percentile(detections: np.ndarray, percentile: int = 99):
    detections = detections[:, 0]
    if not len(detections):
        return 0.0
    detections_count, bins = np.histogram(detections,
                                          bins=detections.max().astype(int) + 1)
    return np.percentile(detections_count, percentile)


absolute_to_relative = partial(normalize_boxes, denormalize=False)
relative_to_absolute = partial(normalize_boxes, denormalize=True)
