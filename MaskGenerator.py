import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main(mask):
    CL = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    cv2.drawContours(CL,[largest_contour],0,(0,250,0),2)
    M = cv2.moments(largest_contour)
    cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
    cv2.circle(CL,(cx,cy),3,(0,0,255),3)
    cv2.imshow('Contornos', CL)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cx,cy

if __name__=="__main__":
    main(mask)