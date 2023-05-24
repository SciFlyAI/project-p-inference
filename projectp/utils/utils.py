from functools import partial
from os import path as osp
from typing import Dict

import numpy as np


class LogStub:
    def _message(self, message, tip='INFO'):
        print(f"[{tip}]: {message}")

    def debug(self, message):
        self._message(message, 'DEBUG')

    def info(self, message):
        self._message(message, 'INFO')

    def warning(self, message):
        self._message(message, 'WARNING')

    def error(self, message):
        self._message(message, 'ERROR')

    def fatal(self, message):
        self._message(message, 'FATAL')
        raise RuntimeError


log = LogStub()


def nms(boxes, scores, thresh):
    '''
    dets is a numpy array : num_dets, 4
    scores is a  nump array : num_dets,
    '''
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1] # get boxes with more ious first

    keep = []
    while order.size > 0:
        i = order[0] # pick maxmum iou box
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1) # maximum width
        h = np.maximum(0.0, yy2 - yy1 + 1) # maxiumum height
        area = w * h
        overlap = area / (areas[i] + areas[order[1:]] - area)

        indices = np.where(overlap <= thresh)[0]
        order = order[indices + 1]

    return keep


def convert_to_pandas(boxes_total: Dict[str, np.ndarray]):
    try:
        import pandas as pd
    except ImportError:
        log.error(f"install Pandas to be able to convert to DataFrame!")
        return None

    frame_data = pd.DataFrame()

    for key in boxes_total:
        print(key, boxes_total[key].shape)
        frame_video = pd.DataFrame(
            boxes_total[key],
            columns=['frame', 'center_x', 'center_y',
                     'width', 'height', 'confidence', 'class']
        )
        frame_video['path'] = osp.realpath(osp.abspath(osp.dirname(key)))
        frame_video['filename'] = osp.basename(key)
        frame_data = pd.concat(
            [frame_data, frame_video.apply(partial(pd.to_numeric,
                                                   downcast='integer'),
                                           errors='ignore')])

    return frame_data
