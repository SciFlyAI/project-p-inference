from os import path as osp
from time import perf_counter

import cv2 as cv
import numpy as np

from onnxruntime import InferenceSession
from ensemble_boxes import nms, weighted_boxes_fusion

from ..processing import (
    get_tiles, yolo_to_xyxy, draw_detections,
    absolute_to_relative, relative_to_absolute
)
from ..utils import LogStub


try:
    from tqdm import tqdm
except ImportError:
    class ProgressStub(tuple):
        def __init__(self, *args, **kwargs):
            ...

    tqdm = ProgressStub


log = LogStub()


class InferenceONNX:
    def __init__(self, path: str, prefix: str = None):
        """
        path: path to the model to predict from (a single model)
        prefix: path prefix for the model(s)
        """
        assert isinstance(path, str), f"path must be a single path!"
        prefix = prefix or '.'
        assert osp.isdir(prefix), f"prefix must be a valid directory or None!"
        self.session = InferenceSession(osp.join(prefix, path))
        self.input = self.session.get_inputs()[0].shape

    def _process_tiles(self, frame, results, confidence=0.45, time_start=None,
                       debug=False):
        time_start = time_start or perf_counter()
        tiles = get_tiles(frame, self.input[-1:-3:-1])
        boxes_frame = np.zeros((0, 6), dtype=np.float32)
        for i in range(tiles.shape[1]):
            for j in range(tiles.shape[0]):
                tile = tiles[j, i]
                image = cv.cvtColor(frame[tile.slice],
                                    cv.COLOR_BGR2RGB)
                # log.debug(f"Image shape = {image.shape},"
                #           f" slice = {tile.slice}")

                # ONNX inference
                batch = np.moveaxis(
                    image, -1, 0
                )[None, ...] / np.float32(255)
                # TODO: batch size > 1
                boxes = self.session.run(
                    None,
                    {self.session.get_inputs()[0].name: batch}
                )[0][0]
                tile.bboxes = np.zeros((0, 6), dtype=np.float32)  # <- boxes
                if debug:
                    log.debug(
                        f"Patch {(i, j)} "
                        f"{(tile.x1, tile.y1, tile.x2, tile.y2)},"
                        f" boxes = {boxes.shape}"
                    )
                boxes_norm = yolo_to_xyxy(boxes, frame, (1, 1),
                                          tile.x1, tile.y1)
                boxes_norm = absolute_to_relative(boxes_norm,
                                                  frame)
                # boxes_norm_ = boxes_norm

                # NMS
                boxes_nms = boxes_norm[boxes_norm[..., 4] > confidence]

                if debug:
                    log.debug(f"Boxes frame = {boxes_frame.shape},"
                              f" boxes NMS = {boxes_nms.shape}")
                boxes_wbf = weighted_boxes_fusion(
                    (np.clip(boxes_nms[..., :4], 0, 1),
                     np.clip(boxes_frame[..., :4], 0, 1)
                     ),
                    (boxes_nms[..., 4],
                     boxes_frame[..., 4]
                     ),
                    (boxes_nms[..., 5].round(),
                     boxes_frame[..., 5].round()
                     ),
                    iou_thr=0.175,  # 0.175
                    skip_box_thr=confidence,
                    conf_type='max'
                )
                boxes_wbf = np.column_stack(boxes_wbf)

                boxes_frame = np.vstack((boxes_frame, boxes_wbf))

        if len(boxes_frame):
            boxes_nms = nms(
                (boxes_frame[..., :4],
                 ),
                (boxes_frame[..., 4],
                 ),
                (boxes_frame[..., 5].round(),
                 ), iou_thr=0.65  # 0.175
            )
            boxes_nms = np.column_stack(boxes_nms)
        else:
            boxes_nms = boxes_frame
        boxes_frame = relative_to_absolute(boxes_nms, frame)
        # Prepend frame# to detection vector
        boxes_frame = np.insert(boxes_frame, 0, results['index_frame'], -1)
        results['boxes'] = np.vstack([results['boxes'], boxes_frame])
        results['tiles'].append(tiles)
        results['times']['frames'].append(perf_counter() - time_start)

        # Draw detections and save to video
        frame = draw_detections(frame, boxes_frame[:, 1:])
        return frame

    def process_video(self, filename_source, prefix_target=None,
                      suffix_target=None, confidence=0.45, codec='mp4v',
                      max_frames=0, progress=True, debug=False, feedback=None):
        prefix_target = prefix_target or osp.dirname(filename_source)
        assert osp.isdir(prefix_target), \
            f"prefix '{prefix_target}' is not a valid directory!"
        suffix_target = suffix_target or 'output'
        video_source = cv.VideoCapture(filename_source)
        results = {
            'index_frame': -1,
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'tiles': [],
            'times': {
                'frames': [],
                'total': None
            }
        }
        # boxes_video = np.zeros((0, 7), dtype=np.float32)  # []
        # tiles_video = []
        # times_video = {
        #     'frames': [],
        #     'total': None
        # }
        count = 0
        try:
            if video_source.isOpened():
                count_total = int(video_source.get(cv.CAP_PROP_FRAME_COUNT))
                max_frames = max_frames or count_total
                index_frame = -1
                fps = video_source.get(cv.CAP_PROP_FPS)
                w, h = (video_source.get(cv.CAP_PROP_FRAME_WIDTH),
                        video_source.get(cv.CAP_PROP_FRAME_HEIGHT))
                w, h = tuple(map(int, (w, h)))
                filename_target = osp.join(prefix_target, osp.basename(
                    f"{osp.splitext(filename_source)[0]}.{suffix_target}.mp4"
                ))
                video_target = cv.VideoWriter(filename_target,
                                              cv.VideoWriter_fourcc(*codec),
                                              fps, (w, h), True)
                # boxes_video = np.zeros((0, 7), dtype=np.float32)
                with tqdm(total=max_frames, position=0, leave=True,
                          disable=not progress or debug) as progress_file:
                    while True:
                        time_start = perf_counter()
                        ok, frame = video_source.read()
                        results['index_frame'] += 1
                        if not ok or results['index_frame'] >= max_frames:
                            break
                        else:
                            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        frame = self._process_tiles(frame,
                                                    results,
                                                    confidence=confidence,
                                                    time_start=time_start,
                                                    debug=debug)
                        video_target.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                        if debug:
                            log.debug(f"Frame {count:05d} done in"
                                      f" {results['times']['frames'][-1]:.3f} sec")
                        count += 1
                        progress_file.update(1)
                        # break
                video_target.release()
            else:
                log.error(f"Can't open source file '{filename_source}'...")
        except KeyboardInterrupt:
            log.info(f"Interrupt at frame#{count:05d}...")
            # Inform calling routine with feedback (if any) that
            # KeyboardInterrupt has been occurred in order to finish gracefully
            if isinstance(feedback, dict):
                feedback['continue'] = False
        finally:
            try:
                video_target.release()
            except (NameError, AttributeError):
                pass
            video_source.release()
        results['times']['frames'] = np.array(results['times']['frames'])
        results['times']['total'] = results['times']['frames'].sum()
        if debug:
            log.debug(f"File '{filename_source}' done in"
                      f" {results['times']['total'] / 60:.3f} min")
        # break
        return results['boxes'], results['tiles'], results['times']

    def process_videos(self, filenames, prefix_target=None, suffix_target=None,
                       confidence=0.45, codec='mp4v',
                       max_files=0, max_frames=0,
                       progress=True, debug=False):
        boxes_total = {}
        tiles_total = {}
        times_total = {}
        feedback = {
            'continue': True
        }
        max_files = (max_files
                     if isinstance(max_files, int) and max_files
                     else None)

        filenames = filenames[:max_files]
        count = 0
        # TODO: multiple progress bars
        for filename_source in tqdm(filenames, position=0,
                                    leave=True, disable=True):
            log.info(f"Processing file {count + 1}/{len(filenames)}...")
            boxes_video, tiles_video, times_video = self.process_video(
                filename_source=filename_source,
                prefix_target=prefix_target,
                suffix_target=suffix_target,
                confidence=confidence,
                codec=codec,
                max_frames=max_frames,
                progress=progress,
                debug=debug,
                feedback=feedback
            )
            boxes_total[filename_source] = boxes_video
            tiles_total[filename_source] = tiles_video
            times_total[filename_source] = times_video
            if not feedback['continue']:
                break
            count += 1
        return boxes_total, tiles_total, times_total
