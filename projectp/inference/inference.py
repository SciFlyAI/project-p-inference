from os import path as osp
from time import perf_counter, strftime, time_ns

import cv2 as cv
import numpy as np

from ensemble_boxes import nms, weighted_boxes_fusion
from filetype import guess
from onnxruntime import InferenceSession

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
    def __init__(self, path: str, prefix: str = None, log=log, sess_options=None, providers=None):
        """
        path: path to the model to predict from (a single model)
        prefix: path prefix for the model(s)
        log: logging facility
        sess_options: ONNX Runtime session options
        providers: a collection of ONNX execution providers
        """
        assert isinstance(path, str), f"Path must be a single path!"
        prefix = prefix or '.'
        assert osp.isdir(prefix), f"Prefix must be a valid directory or None!"
        self.session = InferenceSession(osp.join(prefix, path), sess_options=sess_options,
                                        providers=providers)
        self.input = self.session.get_inputs()[0].shape
        self.log = log

    def _process_tiles(self, frame, results, confidence=0.45, shape=None, debug=False):
        """Low-level tile-wise inference method

        Args:
            frame (numpy.ndarray): source image for detection
            results (dict): dictionary with 'boxes' (absolute [index, x1, y1, x2, y2, score, class]),
             'tiles' and 'times' entries
            confidence (float, optional): confidence score threshold. Defaults to 0.45.
            shape (str, optional): processing:draw_detections shape. Defaults to None.
            debug (bool, optional): debugging mode - verbose output. Defaults to False.

        Returns:
            numpy.ndarray: frame with detections (processing:draw_detections)
        """
        time_start = results['time_start'] or perf_counter()
        boxes_frame = np.zeros((0, 6), dtype=np.float32)  # [cx, cy, w, h, confidence, label]
        try:
            tiles = get_tiles(frame, size_crop=self.input[-1:-3:-1],
                              log=self.log)
        except AssertionError as ex:
            log.error(ex)
            tiles = np.array([])
            boxes_frame = np.insert(boxes_frame, 0, results['index_frame'], -1)
            results['boxes'] = np.vstack([results['boxes'], boxes_frame])
            results['tiles'].append(tiles)
            results['times']['frames'].append(perf_counter() - time_start)
            return None
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
                )[0][0]  # relative yolo coordinates
                tile.bboxes = np.zeros((0, 6), dtype=np.float32)  # <- boxes
                if debug:
                    log.debug(
                        f"Patch {(i, j)} "
                        f"{(tile.x1, tile.y1, tile.x2, tile.y2)},"
                        f" boxes = {boxes.shape}"
                    )
                boxes_norm = yolo_to_xyxy(boxes, frame, (1, 1),
                                          tile.x1, tile.y1)  # absolute [x1, y1, x2, y2, score, class]
                boxes_norm = absolute_to_relative(boxes_norm,
                                                  frame)  # relative [x1, y1, x2, y2, score, class]

                # Initial NMS/WBF
                boxes_nms = boxes_norm[boxes_norm[..., 4] > confidence]  # relative [x1, y1, x2, y2, score, class]

                if debug:
                    log.debug(f"Boxes frame = {boxes_frame.shape},"
                              f" boxes NMS = {boxes_nms.shape}")
                boxes_wbf = weighted_boxes_fusion(
                    (np.clip(boxes_nms[..., :4], 0, 1),  # current boxes
                     np.clip(boxes_frame[..., :4], 0, 1)  # accumulated boxes
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
                )  # (boxes, scores, classes)
                boxes_wbf = np.column_stack(boxes_wbf)  # relative [x1, y1, x2, y2, score, class]

                boxes_frame = np.vstack((boxes_frame, boxes_wbf))  # relative [x1, y1, x2, y2, score, class]

        # NMS over WBF / TODO: ablation study
        if len(boxes_frame):
            boxes_nms = nms(
                (boxes_frame[..., :4],
                 ),
                (boxes_frame[..., 4],
                 ),
                (boxes_frame[..., 5].round(),
                 ), iou_thr=0.65  # 0.175
            )  # (boxes, scores, classes)
            boxes_nms = np.column_stack(boxes_nms)  # relative [x1, y1, x2, y2, score, class]
        else:
            boxes_nms = boxes_frame
        boxes_frame = relative_to_absolute(boxes_nms, frame)  # absolute [x1, y1, x2, y2, score, class]
        # Prepend frame# to detection vector
        boxes_frame = np.insert(boxes_frame, 0, results['index_frame'], -1)  # absolute [index, x1, y1, x2, y2, score, class]
        results['boxes'] = np.vstack([results['boxes'], boxes_frame])  # absolute [index, x1, y1, x2, y2, score, class]
        results['tiles'].append(tiles)
        results['times']['frames'].append(perf_counter() - time_start)

        # Draw detections and save to video
        if shape is not None:
            frame = draw_detections(frame, boxes_frame[:, 1:], shape=shape)
        return frame

    def process_image(self, source, prefix_target=None,
                      suffix_target=None, confidence=0.45, shape=None,
                      save=True, debug=False, feedback=None):
        """Tile-wise image processing (inference) method

        Args:
            source (str | numpy.ndarray): image source - either path or numpy array
            prefix_target (str, optional): source file parent directory. Defaults to None.
            suffix_target (str, optional): target file suffix. Defaults to '.output'.
            confidence (float, optional): detection confidence threshold. Defaults to 0.45.
            shape (str, optional): processing:draw_detections shape. Defaults to None.
            save (bool, optional): whether save output with predictions. Defaults to True.
            debug (bool, optional): debugging mode - verbose output. Defaults to False.
            feedback (dict, optional): dictionary with control variables. Defaults to None.

        Returns:
            tuple: a tuple of predicted boxes (with confidences and labels), tile map, times
        """
        results = {
            'index_frame': 0,
            'time_start': None,
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'tiles': [],
            'times': {
                'frames': [],
                'total': None
            }
        }
        if isinstance(source, str):
            filename_source = source
            image_source = cv.imread(filename_source, cv.IMREAD_COLOR)
        elif isinstance(source, np.ndarray):
            filename_source = f"/tmp/{strftime('%Y-%m-%d-%H-%M-%S')}-{int(time_ns() % 10e9)}.npy"
            image_source = source
        else:
            log.error(f"{type(source)} is invalid source type!")
            return results['boxes'], results['tiles'], results['times']
        prefix_target = prefix_target or osp.dirname(filename_source)
        assert osp.isdir(prefix_target), \
            f"Prefix '{prefix_target}' is not a valid directory!"
        suffix_target = '.output' if suffix_target is None else suffix_target  # TODO: move to const
        try:
            if image_source is not None:
                filename_target = osp.join(prefix_target, osp.basename(
                    f"{osp.splitext(filename_source)[0]}{suffix_target}.jpg"
                ))
                results['time_start'] = perf_counter()
                frame = self._process_tiles(cv.cvtColor(image_source,
                                                        cv.COLOR_BGR2RGB),
                                            results, confidence=confidence,
                                            shape=shape, debug=debug)
                assert frame is not None, f"Failed to process {filename_source}!"
                if save:
                    cv.imwrite(filename_target, cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                if debug:
                    log.debug(f"Image '{filename_target}' done in"
                              f" {results['times']['frames'][-1]:.3f} sec")
            else:
                log.error(f"Can't open source file '{filename_source}'...")
        except KeyboardInterrupt:
            log.info(f"Interrupt at image '{filename_target}'...")
            # Inform calling routine with feedback (if any) that
            # KeyboardInterrupt has been occurred in order to finish gracefully
            if isinstance(feedback, dict):
                feedback['continue'] = False
        except AssertionError as ex:
            log.error(ex)
        finally:
            ...
        results['times']['frames'] = np.array(results['times']['frames'])
        results['times']['total'] = results['times']['frames'].sum()
        if debug:
            log.debug(f"File '{filename_source}' done in"
                      f" {results['times']['total'] / 60:.3f} min")
        return results['boxes'], results['tiles'], results['times']

    def process_video(self, filename_source, prefix_target=None,
                      suffix_target=None, confidence=0.45, codec='mp4v',
                      ext='mp4', max_frames=0, progress=True, shape=None,
                      debug=False, feedback=None):
        prefix_target = prefix_target or osp.dirname(filename_source)
        assert osp.isdir(prefix_target), \
            f"Prefix '{prefix_target}' is not a valid directory!"
        suffix_target = suffix_target or 'output'
        video_source = cv.VideoCapture(filename_source)
        results = {
            'index_frame': -1,
            'time_start': None,
            'boxes': np.zeros((0, 7), dtype=np.float32),
            'tiles': [],
            'times': {
                'frames': [],
                'total': None
            },
            'frames_failed': 0
        }
        count = 0
        try:
            if video_source.isOpened():
                count_total = int(video_source.get(cv.CAP_PROP_FRAME_COUNT))
                max_frames = max_frames or count_total
                fps = video_source.get(cv.CAP_PROP_FPS)
                w, h = (video_source.get(cv.CAP_PROP_FRAME_WIDTH),
                        video_source.get(cv.CAP_PROP_FRAME_HEIGHT))
                w, h = tuple(map(int, (w, h)))
                filename_target = osp.join(prefix_target, osp.basename(
                    f"{osp.splitext(filename_source)[0]}.{suffix_target}.{ext}"
                ))  # TODO: file extension from codec mapping
                video_target = cv.VideoWriter(filename_target,
                                              cv.VideoWriter_fourcc(*codec),
                                              fps, (w, h), True)
                with tqdm(total=max_frames, position=0, leave=True,
                          disable=not progress or debug) as progress_file:
                    while True:
                        results['time_start'] = perf_counter()
                        ok, frame = video_source.read()
                        ok = ok and frame is not None  # paranoid mode: on
                        results['index_frame'] += 1
                        if not ok or results['index_frame'] >= max_frames:
                            break
                        else:
                            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                        frame = self._process_tiles(frame,
                                                    results,
                                                    confidence=confidence,
                                                    shape=shape, debug=debug)
                        if frame is None:
                            break  # exception occurred in _process_tiles
                        count += 1
                        progress_file.update(1)
                        # assert frame is not None, f"Failed to process frame#{count}!"
                        if frame is None:
                            results['frames_failed'] += 1
                            continue
                        video_target.write(cv.cvtColor(frame, cv.COLOR_RGB2BGR))
                        if debug:
                            log.debug(f"Frame {count:05d} done in"
                                      f" {results['times']['frames'][-1]:.3f} sec")
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
            log.debug(f"Frames failed to process = {results['frames_failed']}")
        return results['boxes'], results['tiles'], results['times']

    def process_images(self, filenames, prefix_target=None, suffix_target=None,
                       confidence=0.45, max_files=0,
                       progress=True, shape=None, save=True, debug=False):
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
        # TODO: multiple progress bars
        log.info(f"Processing {len(filenames)} images...")
        for filename_source in tqdm(filenames, position=0,
                                    leave=True, disable=not progress or debug):
            boxes_image, tiles_image, times_image = self.process_image(
                filename_source=filename_source,
                prefix_target=prefix_target,
                suffix_target=suffix_target,
                confidence=confidence,
                shape=shape,
                save=save,
                debug=debug,
                feedback=feedback
            )
            boxes_total[filename_source] = boxes_image
            tiles_total[filename_source] = tiles_image
            times_total[filename_source] = times_image
            if not feedback['continue']:
                break
        return boxes_total, tiles_total, times_total

    def process_videos(self, filenames, prefix_target=None, suffix_target=None,
                       confidence=0.45, codec='mp4v', ext='mp4',
                       max_files=0, max_frames=0,
                       progress=True, shape=None, debug=False):
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
            log.info(f"Processing video {count + 1}/{len(filenames)}...")
            boxes_video, tiles_video, times_video = self.process_video(
                filename_source=filename_source,
                prefix_target=prefix_target,
                suffix_target=suffix_target,
                confidence=confidence,
                codec=codec,
                ext=ext,
                max_frames=max_frames,
                progress=progress,
                shape=shape,
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

    def process_files(self, filenames, prefix_target=None, suffix_target=None,
                      confidence=0.45, codec='mp4v', ext='mp4',
                      max_files=0, max_frames=0,
                      progress=True, shape=None, debug=False):
        filenames_images = []
        filenames_videos = []
        filenames_unknown = []

        boxes_total = {}
        tiles_total = {}
        times_total = {}

        log.info(f"Searching for files of known types...")
        for filename in tqdm(filenames, position=0, leave=True,
                             disable=not progress or debug):
            kind = guess(filename)

            if kind is None:
                if debug:
                    log.warning(f"Skipping file '{filename}'"
                                f" of unknown type...")
                filenames_unknown.append(filename)
            else:
                media, filetype = kind.mime.split('/')[:2]

                if media == 'image':
                    filenames_images.append(filename)
                elif media == 'video':
                    filenames_videos.append(filename)
                else:
                    ...

        if filenames_unknown:
            log.warning(f"Skipping {len(filenames_unknown)}"
                        f" files of unknown type...")

        boxes_images, tiles_images, times_images = self.process_images(
            filenames_images,
            prefix_target=prefix_target,
            suffix_target=suffix_target,
            confidence=confidence,
            max_files=max_files,
            progress=progress,
            shape=shape,
            debug=debug
        )
        boxes_total.update(boxes_images)
        tiles_total.update(tiles_images)
        times_total.update(times_images)

        boxes_videos, tiles_videos, times_videos = self.process_videos(
            filenames_videos,
            prefix_target=prefix_target,
            suffix_target=suffix_target,
            confidence=confidence,
            codec=codec,
            max_files=max_files,
            max_frames=max_frames,
            progress=progress,
            shape=shape,
            debug=debug
        )
        boxes_total.update(boxes_videos)
        tiles_total.update(tiles_videos)
        times_total.update(times_videos)

        return boxes_total, tiles_total, times_total
