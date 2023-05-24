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


def get_tiles(image, size_crop, verbose=False):
    width, height = image.shape[1::-1]

    assert tuple(image.shape[:2]) == (height, width)

    # bbcoords = np.array(bboxes['bbox'].tolist())
    # Coordinates: center point + sizes (all realtive to image dimensions)
    # bbcoords[:, 0:2] += bbcoords[:, 2:4] / 2
    # bbcoords[:, 0::2] /= width
    # bbcoords[:, 1::2] /= height
    assert (
        isinstance(size_crop, (int, float)) or
        isinstance(size_crop, (dict, list, set, tuple)) and
        len(size_crop) == 2
    ), (f"size_crop must be a single value or a collection of two values,"
        f" got {type(size_crop)} = {size_crop}!")
    if isinstance(size_crop, (dict, list, set, tuple)):
        size_crop = tuple(size_crop)[:2]
    else:
        size_crop = (size_crop, size_crop)

    ratio_x, ratio_y = size_crop[0] / width, size_crop[1] / height

    num_patches_x = m.ceil(1 / ratio_x)
    num_patches_y = m.ceil(1 / ratio_y)
    tiles = np.array(
        [None] * num_patches_x * num_patches_y
    ).reshape((num_patches_y, num_patches_x))
    # log.debug(f"{path} size = {(width, height)}, num patches = {(num_patches_x, num_patches_y)}")
    if verbose:
        log.info(f"Size = {(width, height)},"
                 f" num patches = {(num_patches_x, num_patches_y)},"
                 f" tiles.shape = {tiles.shape}")
    # count_bboxes = 0
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
            # bbcoords_clip = np.clip(bbcoords, clip_min, clip_max)
            # array_bboxes = bbcoords_clip[((bbcoords_clip[:, :2] != clip_min[:2]) &
            #                               (bbcoords_clip[:, :2] != clip_max[:2])).all(axis=1)]
            # array_bboxes = bbcoords[(bbcoords[:, 0] > clip_min_x) &
            #                         (bbcoords[:, 1] > clip_min_y) &
            #                         (bbcoords[:, 0] < clip_max_x) &
            #                         (bbcoords[:, 1] < clip_max_y)]
            # Shift coordinates
            # array_bboxes[:, 0:1] -= clip_min_x
            # array_bboxes[:, 1:2] -= clip_min_y
            # Scale coordinates
            # array_bboxes /= np.array([ratio_x, ratio_y, ratio_x, ratio_y])
            # num_bboxes = len(array_bboxes)
            # count_bboxes += num_bboxes
            # log.debug(f"Objects in the patch = {num_bboxes}")
            # path_output = osp.join(target, f"{filename[0]}.{offset_x:05d}.{offset_y:05d}{filename[1]}")
            # log.debug(path_output)
            # cv.imwrite(path_output, image[offset_y:offset_y + size_crop[1] + 1,
            #                               offset_x:offset_x + size_crop[0] + 1])
            # with open(f"{osp.splitext(path_output)[0]}.txt", 'w') as file_annotation:
            #     # FIXME: hardcoded 0 index (take from list of categories)
            #     file_annotation.writelines([f"0 {b[0]} {b[1]} {b[2]} {b[3]}\n" for b in array_bboxes])
            # Update y-offset (end of iteration)
            shift_y = (((num_patches_y - j) - (height - offset_y) / size_crop[1]) /
                       (num_patches_y - j - 1 + 1e-15))
            offset_y += round(size_crop[1] - size_crop[1] * shift_y)
            # log.debug(shift_y, height - offset_y)
            # pass
        # Update x-offset (end of iteration)
        shift_x = (((num_patches_x - i) - (width - offset_x) / size_crop[0]) /
                   (num_patches_x - i - 1 + 1e-15))
        offset_x += round(size_crop[0] - size_crop[0] * shift_x)
        # log.debug(shift_x, width - offset_x)
    # log.info(f"Total bboxes (with overlap) = {count_bboxes} ({path})!")
    # break
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

    # if np.any([offset_x, offset_y]):
    #     boxes_norm[:, 0] += offset_x / image.shape[1]  # x1 + offset
    #     boxes_norm[:, 1] += offset_y / image.shape[0]  # y1 + offset

    boxes_norm[:, 2] += boxes_norm[:, 0]  # x2
    boxes_norm[:, 3] += boxes_norm[:, 1]  # y2

    if np.any([offset_x, offset_y]):
        # log.debug('Yololo!')
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
                    text_origin_x=None, text_origin_y=None):
    scma = get_colors(detections)  # TODO: replace with numpy array

    # frame = frame.copy()
    # detections = detections.copy()
    coords_all = (
            detections[..., 0:2] +
            (detections[..., 2:4] - detections[..., 0:2]) / 2
    )
    # for i, detection in enumerate(detections):  # boxes_total[0]:
    for i, coords in enumerate(coords_all):
        # coords = detection[0:2] + (detection[2:4] - detection[0:2]) / 2
        color = (
                np.array(scma.to_rgba(i)[:3]) * 255
        ).round().astype('uint8').tolist()
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
    detections_count, bins = np.histogram(detections,
                                          bins=detections.max().astype(int) + 1)
    return np.percentile(detections_count, percentile)


absolute_to_relative = partial(normalize_boxes, denormalize=False)
relative_to_absolute = partial(normalize_boxes, denormalize=True)
