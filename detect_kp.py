import cv2
import kpdetection
import detectron.utils.vis as vis_utils

try:
  video = cv2.VideoCapture("video.avi")
  _, im = video.read()
  #im = cv2.imread("image.jpg")
  while im is not None:
    
    #Detect stuff and convert to usable form
    cls_boxes, cls_segms, cls_keyps = kpdetection.detect(im)
    boxes, segms, keyps, classes = \
        vis_utils.convert_from_cls_format(cls_boxes,
                                          cls_segms,
                                          cls_keyps)

    #Draw keypoints on frame
    visualize = False
    if visualize:
        vis = vis_utils.vis_one_image_opencv(im,
                                      cls_boxes,
                                      keypoints=cls_keyps)
        cv2.imshow("image", vis)
        cv2.waitKey(1000)
    
    #Remove keypoints below thresholds
    keyps = kpdetection.prune(keyps, boxes)
    
    print len(keyps)
    
    #Read next frame
    _, im = video.read()
    #im = None
    
  kpdetection.cleanup()

except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()