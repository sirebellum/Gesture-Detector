# Gesture-Detector

Must install detectron from https://github.com/facebookresearch/Detectron

MISC: <br>
&nbsp;  kps = keypoints <br>
&nbsp;  svm.p = pickled svm trained on sitting and standing classes <br>

USAGE: <br>
&nbsp;  python2 detect_kp.py      #Performs keypoint detection and classification <br>
&nbsp;  python2 train_svm.py      #Trains svm using a pickled kps-dictionary and stores model file <br>
&nbsp;  python2 data_processing/collect_data.py     #Script to collect keypoints with user specified class from arbitrary media (video, webcam, image). Code needs to be edited to change input <br>
