import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def main(image,mask,cy_center):
    # Rutas de los archivos
    """ mask_path = 'mask_0scissors.png'
    image_path = 'frame.jpg' """

    """ # Cargar la máscara binaria y la imagen original
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR) """

    # Convertir la imagen a espacio de color HSV
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Obtener las coordenadas de los píxeles blancos en la máscara
    coords = np.column_stack(np.where(mask > 0))

    # Obtener los valores de los canales H y S correspondientes en la imagen original
    hs_values = image_hsv[coords[:, 0], coords[:, 1], 0:2]

    # Especificar el número de clusters (en este caso, 2)
    num_clusters = 2

    # Realizar el clustering utilizando K-Means en los valores de H y S
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit(hs_values)

    # Asignar cada píxel de la máscara a su respectivo cluster
    labels = kmeans.labels_

    # Crear una máscara para cada cluster
    cluster_masks = []
    up = True
    mango = False
    for cluster_label in range(num_clusters):
        cluster_mask = np.zeros_like(mask, dtype=np.uint8)
        cluster_mask[coords[labels == cluster_label, 0], coords[labels == cluster_label, 1]] = 255
        cluster_masks.append(cluster_mask)

    # Mostrar las máscaras resultantes
    for i, cluster_mask in enumerate(cluster_masks):
        cv2.imshow(f'Cluster {i+1}', cluster_mask)


    # Dilate the images to solve problems in the detection
    filled_cluster_masks = []
    kernel = np.ones((3, 3), np.uint8)
    for cluster_mask in cluster_masks:
        contours, _ = cv2.findContours(cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        dilated_mask = cv2.dilate(cluster_mask, kernel, iterations=1)
        #dilated_mask = cv2.morphologyEx(cluster_mask, cv2.MORPH_CLOSE, kernel)
        filled_cluster_masks.append(dilated_mask)

    # Mostrar las máscaras rellenadas
    for i, filled_mask in enumerate(filled_cluster_masks):
        cv2.imshow(f'Cluster {i+1} Rellenado', filled_mask)

    # Calcular histograma para el canal H y S de cada cluster
    for cluster_label in range(num_clusters):
        cluster_hs = hs_values[labels == cluster_label]
        hist_h, _ = np.histogram(cluster_hs[:, 0], bins=180, range=(0, 180))
        hist_s, _ = np.histogram(cluster_hs[:, 1], bins=256, range=(0, 256))

        # Mostrar histograma
        plt.subplot(2, num_clusters, cluster_label + 1)
        plt.plot(hist_h, color='r')
        plt.title(f'Cluster {cluster_label+1} - H')
        plt.subplot(2, num_clusters, num_clusters + cluster_label + 1)
        plt.plot(hist_s, color='b')
        plt.title(f'Cluster {cluster_label+1} - S')

    # Restar la máscara inicial - la máscara del segundo cluster
    diff_mask = cv2.subtract(mask, filled_cluster_masks[1])
    DM = cv2.cvtColor(diff_mask,cv2.COLOR_GRAY2BGR)


    area1 = cv2.countNonZero(mask)
    area2 = cv2.countNonZero(filled_cluster_masks[0])
    area3 = cv2.countNonZero(filled_cluster_masks[1])
    diff_mask2 = (abs(area1 - area3) / area1)*100
    diff_mask = (abs(area1 - area2) / area1)*100
    print("Dif1: {}".format(diff_mask))
    print("Dif2: {}".format(diff_mask2))



    if diff_mask >=35 or diff_mask2 >=35:
        #* Caso en el que si tenemos mango
        # Calcular histograma para el canal H y S de cada cluster
        for cluster_label in range(num_clusters):
            cluster_hs = hs_values[labels == cluster_label]
            hist_h, _ = np.histogram(cluster_hs[:, 0], bins=180, range=(0, 180))
            hist_s, _ = np.histogram(cluster_hs[:, 1], bins=256, range=(0, 256))

            # Mostrar histograma
            plt.subplot(2, num_clusters, cluster_label + 1)
            plt.plot(hist_h, color='r')
            plt.title(f'Cluster {cluster_label+1} - H')
            plt.subplot(2, num_clusters, num_clusters + cluster_label + 1)
            plt.plot(hist_s, color='b')
            plt.title(f'Cluster {cluster_label+1} - S')

            #print("Valor de S:  {}".format(hist_s))

            if any(hist_s[170:-1]):
                #Este es el mango
                CL = cv2.cvtColor(filled_cluster_masks[cluster_label],cv2.COLOR_GRAY2BGR)
                contours, _ = cv2.findContours(filled_cluster_masks[cluster_label], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(CL,[largest_contour],0,(0,250,0),2)
                M = cv2.moments(largest_contour)
                cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
                cv2.circle(CL,(cx,cy),3,(0,0,255),3)
                cv2.imshow('Mango', CL)

                contours2, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                largest_contour2 = max(contours2, key=cv2.contourArea)
                # Calcular el centroide del contorno
                M1 = cv2.moments(largest_contour)
                maskx, masky = int(M1['m10'] / M1['m00']), int(M1['m01'] / M1['m00'])
                if cy < masky:
                    orientacion = 'Hacia abajo'
                    up = False
                    mango = True
                else:
                    orientacion = 'Hacia arriba'
                    up = True
                    mango = True

    else:
        #* Caso en el que no tenemos mango
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        CL2 = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)

        # Calcular el centroide del contorno
        M = cv2.moments(largest_contour)
        cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])

        # Determinar la orientación basándose en la posición del centroide
        #alto, ancho = cuchillo_binario.shape
        alto = 640
        ancho = 480
        print(cy)
        print(cx)
        if cy < cy_center:
            orientacion = 'Hacia arriba'
            up = True
            mango = False
        else:
            orientacion = 'Hacia abajo'
            up = False
            mango = False
        
        print(orientacion)
        cv2.circle(CL2,(cx,cy),3,(255,0,0),3)
        cv2.imshow('Centroide', CL2)

    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return cx,cy, up, mango

if __name__=="__main__":
    main(image,mask,cy_center)