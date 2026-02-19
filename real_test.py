# real_test.py
import json
import os
import yaml
from absl import app, flags, logging
from absl.logging import info
from easydict import EasyDict
from pudb import set_trace
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch
from launch_real_test import ParallelLaunchRealTest # 新しいクラスをインポート

print("HELLO-HPC3-REAL-TEST")

FLAGS = flags.FLAGS
flags.FLAGS.mark_as_parsed()

flags.DEFINE_string("yaml_file", None, "The config file.")
flags.DEFINE_string("RESUME_PATH", None, "The RESUME.PATH.")
flags.DEFINE_string("RESUME_TYPE", "best", "The RESUME.TYPE.")
flags.DEFINE_boolean("RESUME_SET_EPOCH", False, "The RESUME.PATH.")
flags.DEFINE_boolean("RESUME_STRICT", True, "The RESUME_STRICT.")
flags.DEFINE_boolean("TEST_ONLY", True, "The test only.")
flags.DEFINE_boolean("PUDB", False, "The debug switch.")
flags.DEFINE_boolean("VISUALIZE", True, "The visualization switch.")
flags.DEFINE_integer("local-rank", None, "The local rank of the process.")
flags.DEFINE_integer("VAL_BATCH_SIZE", None, "The test batch size.")


def init_config(yaml_path):
    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)
    os.makedirs(FLAGS.log_dir, exist_ok=True)
    logging.set_verbosity(logging.DEBUG)
    info(f"log_dir: {FLAGS.log_dir}")
    logging.get_absl_handler().use_absl_log_file()
    config["SAVE_DIR"] = FLAGS.log_dir
    if FLAGS.RESUME_PATH:
        config["RESUME"]["PATH"] = FLAGS.RESUME_PATH
        config["RESUME"]["TYPE"] = FLAGS.RESUME_TYPE
        config["RESUME"]["SET_EPOCH"] = FLAGS.RESUME_SET_EPOCH
    config["VISUALIZE"] = FLAGS.VISUALIZE
    if FLAGS.VAL_BATCH_SIZE:
        info(f"Update VAL_BATCH_SIZE to {FLAGS.VAL_BATCH_SIZE}")
        config["VAL_BATCH_SIZE"] = FLAGS.VAL_BATCH_SIZE
    config["TEST_ONLY"] = FLAGS.TEST_ONLY
    config["RESUME_STRICT"] = FLAGS.RESUME_STRICT
    if FLAGS.PUDB:
        set_trace()
    info(f"Launch Config: {json.dumps(config, indent=4, sort_keys=True)}")
    return EasyDict(config)


def main(args):
    config = init_config(FLAGS.yaml_file)
    # 0. logging
    # 1. init launcher and run
    launcher = ParallelLaunchRealTest(config)
    launcher.run()


if __name__ == "__main__":
    app.run(main)