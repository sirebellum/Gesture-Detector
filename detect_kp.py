"""Runs keypoint recognition on arbitrary media via opencv streams
   and custom kpdetection Detectron API. Uses custom trained SVM
   to recognize pose. """

import cv2
import kpdetection #Uses Detectron API to return keypoints
import detectron.utils.vis as vis_utils
import pickle
from functions import normalize_kp
import numpy as np

#Load SVM model
filename = "svm.p"
clf = pickle.load(open(filename, "rb"))

try:
  video = cv2.VideoCapture(0)
  if not video.isOpened():
      exit("Couldn't access webcam (check permissions?)")
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
        cv2.waitKey(250)
    
    #SVM
    if len(keyps) > 0: #if anything detected
        instance = normalize_kp(keyps[0])
        blackbox = np.zeros(shape=(255, 255), dtype=np.uint8)
        for x in range(0, 17):
            cv2.circle(blackbox,
                       (instance[0][x], instance[1][x]),
                       1, (255, 255, 255),
                       thickness=2, lineType=8, shift=0)
        cv2.imshow("keypoints", blackbox)
        cv2.waitKey(250)
        instance = np.nan_to_num(np.concatenate((instance[0], instance[1])))
        prediction = clf.predict([instance])
        print(prediction)
    
    #Read next frame
    _, im = video.read()
    #im = None
    
  kpdetection.cleanup()
  
except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()