import cv2
import kpdetection
import detectron.utils.vis as vis_utils

try:
    im = cv2.imread("image.jpg")
    cls_boxes, cls_segms, cls_keyps = kpdetection.detect(im)

    visualize = False
    if visualize:
        vis = vis_utils.vis_one_image_opencv(im,
                                      cls_boxes,
                                      keypoints=cls_keyps)
        cv2.imshow("image", vis)
        cv2.waitKey(1000)
    
    kpdetection.cleanup()

except KeyboardInterrupt:
    kpdetection.cleanup()
    exit()