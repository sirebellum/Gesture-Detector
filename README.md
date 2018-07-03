# Gesture-Detector

Must install detectron from https://github.com/facebookresearch/Detectron

MISC:
  kps: keypoints
  svm.p: pickled svm trained on sitting and standing classes

USAGE:
  python2 detect_kp.py #Performs keypoint detection and classification
  python2 train_svm.py #Trains svm using a pickled kps-dictionary and stores model file 
  python2 data_processing/collect_data.py #Script to collect keypoints with user specified class from arbitrary media (video, webcam, image). Code needs to be edited to change input