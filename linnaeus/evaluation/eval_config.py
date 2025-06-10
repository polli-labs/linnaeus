# linnaeus/evaluation/eval_config.py

from yacs.config import CfgNode as CN


def get_default_eval_config():
    _C = CN()

    # Throughput testing parameters
    _C.THROUGHPUT = CN()
    _C.THROUGHPUT.BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
    _C.THROUGHPUT.NUM_ITERATIONS = 100
    _C.THROUGHPUT.WARM_UP_ITERATIONS = 10
    _C.THROUGHPUT.META_DIMS = []

    return _C


def update_eval_config(eval_config, args):
    eval_config.defrost()

    if args.eval_config:
        eval_config.merge_from_file(args.eval_config)

    if args.eval_opts:
        eval_config.merge_from_list(args.eval_opts)

    eval_config.freeze()
    return eval_config
