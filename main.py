"""File to run via cli."""

from os import makedirs
from os import path as osp


from projectp.inference import InferenceOnnxFullEnd2End, InferenceOnnxTileNms
from projectp.processing import get_percentile
from projectp.utils.logs import log


MODE_FULL = 'full'
MODE_TILE = 'tile'

if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        'sources',
        nargs='+',
        metavar='FILE',
        help="images/videos to predict",
        type=str,
    )
    parser.add_argument(
        '-o',
        '--output',
        default='output',
        metavar='PATH',
        help="path to predictions output",
        type=str,
    )
    parser.add_argument(
        '-m',
        '--model',
        required=True,
        metavar='PATH',
        help="path to model to inference (multiple)",
        type=str,
    )
    parser.add_argument(
        '-i',
        '--inference',
        default=MODE_TILE,
        metavar='MODE',
        choices=[MODE_FULL, MODE_TILE],
        help=f"inference mode (default: {MODE_TILE})",
        type=str,
    )
    parser.add_argument(
        '-p',
        '--prefix',
        default=None,
        metavar='PATH',
        help="path prefix for the input files",
        type=str,
    )
    parser.add_argument(
        '--max-files',
        default=0,
        metavar='COUNT',
        help="limit input files (may be negative)",
        type=int,
    )
    parser.add_argument(
        '--max-frames',
        default=0,
        metavar='COUNT',
        help="limit video frames (may be negative)",
        type=int,
    )
    parser.add_argument(
        '--shape',
        default='dot',
        metavar='TYPE',
        help="detections shape type ('box' or 'dot')",
        type=str,
    )
    parser.add_argument(
        '-s',
        '--silent',
        action='store_true',
        help="do not display progress (display by default)",
    )
    parser.add_argument(
        '-d',
        '--debug',
        action='store_true',
        help="print debug info (off by default)",
    )
    args = parser.parse_args()

    prefix = args.prefix or '.'
    if args.inference == MODE_TILE:
        inference_onnx = InferenceOnnxTileNms(args.model)
    elif args.inference == MODE_FULL:
        inference_onnx = InferenceOnnxFullEnd2End(args.model)
    else:
        raise RuntimeError(f"Unknown inference mode '{args.inference}'!")

    sources = []
    # max_files = args.max_files or None
    for source in args.sources:
        source = osp.join(prefix, source)
        if not osp.isfile(source) or not osp.getsize(source):
            # Skip non-existent or empty files
            continue
        sources.append(source)
    prefix_target = osp.realpath(args.output)
    if osp.isdir(prefix_target):
        log.warning(
            f"output directory '{prefix_target}' already exists,"
            f" conflicting files will be overwritten!"
        )
    else:
        makedirs(prefix_target)
    boxes_total, tiles_total, times_total = inference_onnx.process_files(
        sources,
        prefix_target=prefix_target,
        max_files=args.max_files,
        max_frames=args.max_frames,
        progress=not args.silent,
        shape=args.shape,
        debug=args.debug,
    )
    time_wall = 0.0
    for k, v in times_total.items():
        # log.debug(f"Boxes shape = {boxes_total[k].shape}")
        percentile = round(get_percentile(boxes_total[k], 95))
        log.info(
            f"File '{k}' done in {v['total'] / 60:.3f} min,"
            f" detected {percentile} targets (95-percentile)"
        )
        time_wall += v['total']
    log.info(f"Wall time = {time_wall / 60:.3f} min")
