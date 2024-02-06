import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main(imagen,punto_a, punto_b):
    cv2.circle(imagen, (int(punto_b[0]), int(punto_b[1])), 5, (255, 0, 0), -1) #center_new
    cv2.circle(imagen, (int(punto_a[0]), int(punto_a[1])), 5, (0, 0, 255), -1) #center_old
    cv2.imshow('Puntos', imagen)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__=="__main__":
    main(imagen,punto_a, punto_b)