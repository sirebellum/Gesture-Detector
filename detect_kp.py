import cv2
import kpdetection
import detectron.utils.vis as vis_utils

try:
  video = cv2.VideoCapture("video.avi")
  _, im = video.read()
  while im is not None:
    cls_boxes, cls_segms, cls_keyps = kpdetection.detect(im)

    visualize = False
    if visualize:
        vis = vis_utils.vis_one_image_opencv(im,
                                      cls_boxes,
                                      keypoints=cls_keyps)
        cv2.imshow("image", vis)
        cv2.waitKey(1000)
    
    _, im = video.read()
    
  kpdetection.cleanup()

except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()