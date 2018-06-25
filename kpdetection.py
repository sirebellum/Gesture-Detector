"""Perform keypoint detection on frame"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import logging
import os
import sys
import yaml

from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import load_cfg
from detectron.core.config import merge_cfg_from_cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.logging import setup_logging
import detectron.core.test_engine as model_engine
import detectron.utils.c2 as c2_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# infer.py
#   --im [path/to/image.jpg]
#   --rpn-model [path/to/rpn/model.pkl]
#   --rpn-config [path/to/rpn/config.yaml]
#   [model1] [config1] [model2] [config2] ...

workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
setup_logging(__name__)
logger = logging.getLogger(__name__)

cfg_orig = load_cfg(yaml.dump(cfg))

#Download model of choice from https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md"
#Yaml files available at https://github.com/facebookresearch/Detectron/tree/master/configs
pkl = "model_files/kps_R-50-FPN.pkl"
yml = "model_files/kps_R-50-FPN.yaml"
if not os.path.isfile(pkl):
    exit("Please download model files into model_files")

cfg.immutable(False)
merge_cfg_from_cfg(cfg_orig)
merge_cfg_from_file(yml)
weights_file = pkl
cfg.NUM_GPUS = 1
assert_and_infer_cfg(cache_urls=False)
model = model_engine.initialize_model_from_cfg(weights_file)

def detect(im):
    
    proposal_boxes, cls_boxes, cls_segms, cls_keyps \
      = None, None, None, None
    
    with c2_utils.NamedCudaScope(0):
        cls_boxes_, cls_segms_, cls_keyps_ = \
            model_engine.im_detect_all(model, im, proposal_boxes)
    cls_boxes = cls_boxes_ if cls_boxes_ is not None else cls_boxes
    cls_segms = cls_segms_ if cls_segms_ is not None else cls_segms
    cls_keyps = cls_keyps_ if cls_keyps_ is not None else cls_keyps

    return cls_boxes, cls_segms, cls_keyps

def prune(kps, bxes, boxes_thresh=0.9, kps_thresh=2):

    keypoints = list()
    boxes = list()
    for k in range(0, len(bxes)):
        if bxes[k][-1] >= 0.9: #-1 index is probability
            keypoints.append(kps[k])
            boxes.append(bxes[k])
            
    return keypoints, boxes
    
def cleanup():
    workspace.ResetWorkspace()