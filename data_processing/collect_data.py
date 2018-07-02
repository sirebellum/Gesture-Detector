import cv2
#import kpdetection #Uses Detectron API to return keypoints
#import detectron.utils.vis as vis_utils

#open stream from media source
media = cv2.VideoCapture("../video.avi")

#Get classes from user
classes = raw_input("Please enter classes separated by spaces:\n")
classes = classes.split(" ")
#data{"class": datapoint, datapoint, ... , datapoint}
keypoints = {clas: list() for clas in classes}
print "Collecting data for ", keypoints.keys()

#Begin data collection
collect = True
try:
  while collect is True:
  
    #Define class for this round
    print classes
    clas = raw_input("what class would you like to collect for?: ")
    record = ""
    if clas not in classes:
       print "Please specify a real class"
       record = "n"
    
    #Visualize and store keypoints
    while record == "": #while enter is the only thing pressed
        #Read next frame
        _, frame = media.read()
        if frame is None: #end of stream
            raise ValueError
        
        #Detect stuff and convert to usable form
        #cls_boxes, cls_segms, cls_keyps = kpdetection.detect(frame)
        #boxes, segms, keyps, classes = \
        #        vis_utils.convert_from_cls_format(cls_boxes,
        #                                      cls_segms,
        #                                      cls_keyps)
        
        #Remove keypoints below thresholds
        #keyps, boxes = kpdetection.prune(keyps, boxes)
        
        #Draw keypoints on frame
        #vis = vis_utils.vis_one_image_opencv(frame,
        #                                 cls_boxes,
        #                                 keypoints=cls_keyps)
        vis = frame
        cv2.imshow("image", vis)
        cv2.waitKey(1)
        
        record = raw_input("Record for "+clas+"? (enter/n):")

#Stop data collection, ask to store
except KeyboardInterrupt:
    exit()