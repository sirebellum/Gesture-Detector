import cv2
import kpdetection #Uses Detectron API to return keypoints
import detectron.utils.vis as vis_utils

#open stream from media source
media = cv2.VideoCapture("video.avi")

#Get classes from user
classes = raw_input("Please enter classes separated by spaces:\n")
classes = classes.split(" ")
#data{"class": datapoint, datapoint, ... , datapoint}
keypoints = {clas: list() for clas in classes}
print "Collecting data for ", keypoints.keys()

exit()

#Begin data collection
try:
    _, frame = media.read()
    #Detect stuff and convert to usable form
    cls_boxes, cls_segms, cls_keyps = kpdetection.detect(frame)
    boxes, segms, keyps, classes = \
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
    cv2.waitKey(1)

  
except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()