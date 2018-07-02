import sys
sys.path.append("../") #for custom modules
import cv2
import kpdetection #Uses Detectron API to return keypoints
import detectron.utils.vis as vis_utils
import numpy as np
import pickle
from functions import store_data, normalize_kp

#Review dictionary for use in svm
def review_data(keypoints):
    
    return False
    
#Normalize keypoints within a square
#kps[instance][x, y, logit, prob]
def normalize_data(keypoints):
    
    kps_norm = {key: np.asarray(keypoints[key]) for key in keypoints}
    for clas in kps_norm:
        for instance in kps_norm[clas]:
            instance = normalize_kp(instance)
            
    return kps_norm

#open stream from media source
media = cv2.VideoCapture("train.mp4")
if not media.isOpened():
    exit("Couldn't access webcam (check permissions?)")

#Get classes from user
classes = raw_input("Please enter classes separated by spaces:\n")
classes = classes.split(" ")
#data{"class": datapoint, datapoint, ... , datapoint}
keypoints = {clas: list() for clas in classes}
print "Collecting data for ", keypoints.keys()

#Begin data collection
driver = "c"
try:
  while driver == "c":
  
    ###User defined recording class###
    print classes
    clas = raw_input("what class would you like to record for?: ")
    record = ''
    if clas not in classes:
       print "Please specify a real class"
       record = "n"
    
    ###Visualize and record keypoints###
    while record == '': #while enter is the only thing pressed
      data = list()
      for x in range(0, 30): #read 30 frames at a time
        #Read next frame
        _, frame = media.read()
        if frame is None: #end of stream
            break
        
        #Detect stuff and convert to usable form
        cls_boxes, cls_segms, cls_keyps = kpdetection.detect(frame)
        boxes, segms, keyps, _ = \
                vis_utils.convert_from_cls_format(cls_boxes,
                                              cls_segms,
                                              cls_keyps)
            
        keyps, boxes = kpdetection.prune(keyps, boxes)
        if len(keyps) > 0: #if anything still detected
            instance = normalize_kp(keyps[0])
            data.append(instance)
            
      #Visualize normalized kps
      blackbox = np.zeros(shape=(255, 255), dtype=np.uint8)
      for x in range(0, 17):
          cv2.circle(blackbox,
                     (instance[0][x], instance[1][x]),
                     1, (255, 255, 255),
                     thickness=2, lineType=8, shift=0)
      cv2.imshow("keypoints", blackbox)
      #Visualize keypoints on image
      vis = vis_utils.vis_one_image_opencv(frame,
                                       cls_boxes,
                                       keypoints=cls_keyps)
      cv2.imshow("image", vis)
      cv2.waitKey(100)
            
      #Prompt user for saving individual instance
      record = raw_input("Continue recording for "+clas+"? (enter/n): ")
      if record == '':
          keypoints[clas] = keypoints[clas] + data
            
 
    print "Done recording for", clas+"."
    for clas in classes:
        print len(keypoints[clas]), "total keypoints recorded for", clas+"."
    driver = raw_input("Continue, review, or quit? (c/r/q): ")
  
  ###Clean up and pickle###
  #fit all keypoints within box
  keypoints = normalize_data(keypoints)
  
  #Option to review all keypoints
  if driver == "r":
      keypoints = review_data(keypoints)
  
  #Pickle keypoints
  store_data(keypoints, "data.p")
  kpdetection.cleanup()
  exit()

#Stop data collection, ask to store
except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()