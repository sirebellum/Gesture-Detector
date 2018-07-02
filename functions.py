import pickle
import numpy as np

#Pickle dictionary for use in svm
def store_data(data, filename):
    
    pickle.dump(data, open( filename, "wb" ))
    print "Wrote file to", filename
    
#Normalize individual keypoint instance
def normalize_kp(keypoint):
    
    xmax = np.nanmax(keypoint[0])
    ymax = np.nanmax(keypoint[1])
    xmin = np.nanmin(keypoint[0])
    ymin = np.nanmin(keypoint[1])
    
    width = xmax - xmin
    height = ymax - ymin
    
    #Normalize keypoints within 255x255 square without warping
    kps_norm = np.zeros(shape=(2,17), dtype=np.float16)
    kps_norm[0] = np.subtract(keypoint[0], xmin-1.0)
    kps_norm[1] = np.subtract(keypoint[1], ymin-1.0)
    if width >= height:
        kps_norm[0] *= 255.0/np.nanmax(kps_norm[0])
        kps_norm[1] *= (255.0/np.nanmax(kps_norm[1])) * height/width
    else:
        kps_norm[1] *= 255.0/(ymax-ymin-1.0)
        kps_norm[0] *= (255.0/np.nanmax(kps_norm[0])) * width/height
    #Quantize
    kps_norm = kps_norm.astype(np.uint8)
    
    return kps_norm