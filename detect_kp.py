"""Runs keypoint recognition on arbitrary media via opencv streams
   and custom kpdetection Detectron API."""

import cv2
import kpdetection #Uses Detectron API to return keypoints
import detectron.utils.vis as vis_utils

try:
  video = cv2.VideoCapture(0)
  _, im = video.read()
  #im = cv2.imread("image.jpg")
  while im is not None:
    
    #Detect stuff and convert to usable form
    cls_boxes, cls_segms, cls_keyps = kpdetection.detect(im)
    boxes, segms, keyps, classes = \
        vis_utils.convert_from_cls_format(cls_boxes,
                                          cls_segms,
                                          cls_keyps)
    
    #Remove keypoints below thresholds
    keyps, boxes = kpdetection.prune(keyps, boxes)
    
    #Draw keypoints on frame
    visualize = True
    if visualize:
        vis = vis_utils.vis_one_image_opencv(im,
                                      cls_boxes,
                                      keypoints=cls_keyps)
        cv2.imshow("image", vis)
        cv2.waitKey(1000)
    
    #Read next frame
    _, im = video.read()
    #im = None
    
  kpdetection.cleanup()
  
except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()