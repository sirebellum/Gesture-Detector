import cv2
import kpdetection #Uses Detectron API to return keypoints
import detectron.utils.vis as vis_utils
import numpy as np

#Pickle dictionary for use in svm
def store_data(keypoints):
    
    return False
    
#Review dictionary for use in svm
def review_data(keypoints):
    
    return False
    
#Normalize keypoints within a square
def normalize_data(keypoints):
    
    kps_norm = {key: np.asarray(keypoints[key]) for key in keypoints}
    
    return False

#open stream from media source
media = cv2.VideoCapture(0)
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
  
    #Define class for this round
    print classes
    clas = raw_input("what class would you like to record for?: ")
    record = ''
    if clas not in classes:
       print "Please specify a real class"
       record = "n"
    
    #Visualize and store keypoints
    while record == '': #while enter is the only thing pressed
        #Read next frame
        _, frame = media.read()
        if frame is None: #end of stream
            raise ValueError
        
        #Detect stuff and convert to usable form
        cls_boxes, cls_segms, cls_keyps = kpdetection.detect(frame)
        boxes, segms, keyps, _ = \
                vis_utils.convert_from_cls_format(cls_boxes,
                                              cls_segms,
                                              cls_keyps)
        
        #Remove keypoints below thresholds
        keyps, boxes = kpdetection.prune(keyps, boxes)
        
        #Draw keypoints on frame
        vis = vis_utils.vis_one_image_opencv(frame,
                                         cls_boxes,
                                         keypoints=cls_keyps)
        cv2.imshow("image", vis)
        cv2.waitKey(100)
        
        record = raw_input("Continue recording for "+clas+"? (enter/n): ")
        if record == '':
            keypoints[clas].append(keyps[0])
            
    print "Done recording for", clas+"."
    for clas in classes:
        print len(keypoints[clas]), "total keypoints recorded for", clas+"."
    driver = raw_input("Continue, review, or quit? (c/r/q): ")
  
  #fit all keypoints within box
  normalize_data(keypoints)
  
  if driver == "r":
      review_data(keypoints)
  
  store_data(keypoints)
  kpdetection.cleanup()
  exit()

#Stop data collection, ask to store
except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()