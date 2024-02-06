#LIBRERIA DE ESTE CODIGO
from collections import Counter
import os
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image, ImageDraw
import time

def normalize(value):
    x_min, x_max = -1.0, 1.0
    y_min, y_max = -1.0, 1.0
    x = x_min + (value[0] / 640) * (x_max - x_min)
    y = y_min + (1.0 - value[1] / 480) * (y_max - y_min)

    return x,y

def checkExact(results,masks,img_path,clases_old):
    masks =[]
    # Extrae las coordenadas de las mascaras y las clases detectadas
    for r in results:
        names_cls = r.names
        coordenadas = r.masks.xy    #Devuelve un array 1xN, donde [N] es cada mascara
        clases = r.boxes.cls.cpu().numpy() #Devuelve un array 1xN, donde [N] es cada clase
        #clases = [43, 42, 44]
        print(clases)


        # Convierte los arrays en conjuntos y Counter de frecuencia de los elementos
        set1 = set(clases_old)
        set2 = set(clases)
        count1 = Counter(clases_old)
        count2 = Counter(clases)

        # Comprueba si los conjuntos son iguales
        if set1 == set2 and count1 == count2:
            print("Los objetos coinciden con la deteccion inicial.")
            #* Ordenar en función de clases_old


            # Genera la imagen binaria de la segmentacion
            center_f = []
            size_f = []
            angle_f = []
            vector_df = []
            clases_new =[]

            for i in range(len(coordenadas)):
                # Tamaño de la imagen
                img = Image.open(img_path)
                width, height = img.size

                # Crea una imagen en blanco
                binary_image = Image.new("L", (width, height), 0)

                # Crea un objeto para dibujar en la imagen
                draw = ImageDraw.Draw(binary_image)

                # Lista de puntos que definen el polígono
                #lista_coordenadas = coordenadas.tolist()
                #points_float = coordenadas[i]

                #! Buscar que no coja solo por orden
                print("Clase old:{}".format(clases_old[i]))
                #index = list(coordenadas).index(clases_old[i])
                index = np.where(clases == clases_old[i])[0]
                index = index[0]
                print("Index: {}".format(index))
                points_float = coordenadas[index]
                clases_new.append(clases[index])


                # Convertir los puntos a enteros
                points_int = [(int(round(x)), int(round(y)) ) for x, y in points_float]

                # Dibuja el polígono con relleno 255 y contorno 0
                draw.polygon(points_int, outline=0, fill=255)

                # Guarda la imagen binaria
                binary_image.save("mask_" + str(i) + names_cls[clases_new[i]] + ".png")   

                cv2_mask = np.array(binary_image)
                #cv2_mask = cv2.cvtColor(cv2_mask, cv2.COLOR_GRAY2BGR)

                ###     AQUI EMPIEZA LA EXTRACCION DE LAS OBB   ###

                # Load the image in OpenCV format
                img = cv2.imread(img_path)

                # Resize the image to 640x480
                img = cv2.resize(img, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))
                masks.append(cv2_mask)

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect).astype(int)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = 90 - rect[2] 
                        size = [rect[1][1],rect[1][0]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[2])

                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])
                    else:
                        angle = 180 - rect[2]
                        size = [rect[1][0],rect[1][1]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[0])
                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])

                    # Draw the rotated rectangle on the original image
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Guardar las variables
                    center_f.append(rect[0])
                    size_f.append(size)
                    angle_f.append(angle)

                    # Cierra la imagen binaria
                    binary_image.close()
                    
                    
                    
                else:
                    print("No contours found in the binary_image.")
                    binary_image.close()
                    return None, None, None,None,None, None
            return center_f,size_f,angle_f,img,masks, vector_df
        else:
            print("Los objetos NO coinciden con la deteccion inicial.")
            return None, None, None,None,None, None


def checkAprox(results,masks,img_path,clases_old,cont):
    masks =[]
    # Extrae las coordenadas de las mascaras y las clases detectadas
    for r in results:
        names_cls = r.names
        coordenadas = r.masks.xy    #Devuelve un array 1xN, donde [N] es cada mascara
        clases = r.boxes.cls.cpu().numpy() #Devuelve un array 1xN, donde [N] es cada clase
        print(clases)

        # Convierte los arrays en conjuntos y Counter de frecuencia de los elementos
        set1 = set(clases_old)
        set2 = set(clases)

        # Comprueba si los conjuntos son iguales
        print(set1==set2)
        print(len(clases_old)+cont == len(clases))
        if set1 == set2 and len(clases_old)+cont == len(clases):
            print("APROXIMADO:Los objetos coinciden con la deteccion inicial.")
            # Genera la imagen binaria de la segmentacion
            center_f = []
            size_f = []
            angle_f = []
            vector_df = []
            clases_new = []

            for i in range(len(clases_old)):
                print("Iteracion: {}".format(i))
                # Tamaño de la imagen
                img = Image.open(img_path)
                width, height = img.size

                # Crea una imagen en blanco
                binary_image = Image.new("L", (width, height), 0)

                # Crea un objeto para dibujar en la imagen
                draw = ImageDraw.Draw(binary_image)

                # Lista de puntos que definen el polígono
                #lista_coordenadas = coordenadas.tolist()
                #! Buscar que no coja solo por orden
                print("Clase old:{}".format(clases_old[i]))
                print(len(coordenadas))
                #index = list(coordenadas).index(clases_old[i])
                index = np.where(clases == clases_old[i])[0]
                index = index[0]
                print("Index: {}".format(index))
                points_float = coordenadas[index]
                clases_new.append(clases[index])

                # Convertir los puntos a enteros
                points_int = [(int(round(x)), int(round(y)) ) for x, y in points_float]

                # Dibuja el polígono con relleno 255 y contorno 0
                draw.polygon(points_int, outline=0, fill=255)

                # Guarda la imagen binaria
                binary_image.save("mask_" + str(i) + names_cls[clases_new[i]] + ".png")
                


                cv2_mask = np.array(binary_image)
                #cv2_mask = cv2.cvtColor(cv2_mask, cv2.COLOR_GRAY2BGR)

                ###     AQUI EMPIEZA LA EXTRACCION DE LAS OBB   ###

                # Load the image in OpenCV format
                img = cv2.imread(img_path)

                # Resize the image to 640x480
                img = cv2.resize(img, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))
                masks.append(cv2_mask)

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)
                    box_points = cv2.boxPoints(rect).astype(int)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = 90 - rect[2] 
                        size = [rect[1][1],rect[1][0]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[2])

                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])
                    else:
                        angle = 180 - rect[2] 
                        size = [rect[1][0],rect[1][1]]

                        x1,y1 = normalize(value=box_points[3])
                        x2,y2 = normalize(value=box_points[0])
                        dx = (x2-x1)
                        dy = (y2-y1)
                        vector_df.append([dx,dy])

                    # Draw the rotated rectangle on the original image
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Guardar las variables
                    center_f.append(rect[0])
                    size_f.append(size)
                    angle_f.append(angle)

                    # Cierra la imagen binaria
                    binary_image.close()
                else:
                    print("APROXIMADO: No contours found in the binary_image.")
                    binary_image.close()

                    return None, None, None,None,None, None
                #clases[index]=100 #! Valor muy alto para que no elija siempre el mismo elemento en caso de que haya varios seguidos [42,42,42] cogeria siempre el primer 42
            return center_f,size_f,angle_f,img,masks, vector_df
        else:
            return None, None, None,None,None, None

def main(clases_old, size_old):
    # Load a model
    model = YOLO('yolov8x-seg.pt')  # load an official model

    # Directorio que contiene las imagenes
    #directorio = 'Stable_images/' #! Modificar esto para las pruebas
    directorio = 'Pruebas_buenas/' #! Modificar esto para las pruebas

    masks =[]

    # Itera a traves de los archivos en el directorio
    for img_file in os.listdir(directorio):
        if img_file.endswith('.jpg') or img_file.endswith('.png'):
        #if img_file.endswith('.jpg'):
            img_path = os.path.join(directorio, img_file)
            # Realiza la deteccion en la imagen
            results = model(img_path, stream=True, conf=0.45)
            center_f,size_f,angle_f,img,masks,vector_df = checkExact(results,masks,img_path,clases_old)
            print(img_path)
            print(center_f)

            if center_f!=None:
                break
    cont=1
    while center_f==None:
        for img_file in os.listdir(directorio):
            if img_file.endswith('.jpg') or img_file.endswith('.png'):
            #if img_file.endswith('.jpg'):
                img_path = os.path.join(directorio, img_file)
                # Realiza la deteccion en la imagen
                results = model(img_path, stream=True, conf=0.45)
                center_f,size_f,angle_f,img,masks,vector_df = checkAprox(results,masks,img_path,clases_old,cont)
                if center_f!=None:
                    break
            cont=cont+1
    
    #! Hacer resize de size_f
    # Asegurémonos de que ambas listas tengan la misma longitud
    if len(size_old) != len(size_f):
        print("Different scenes.")
    else:
    # Inicializamos una nueva lista para almacenar los resultados de la división
        w = []
        h = []

        # Iteramos sobre el rango de la longitud de una de las listas
        for i in range(len(size_old)):
            # Realizamos la división y agregamos el resultado a la nueva lista
            w.append(size_f[i][0] / size_old[i][0])
            h.append(size_f[i][1] / size_old[i][1])
        
        size_n=[]
        # Corregimos los valores de size_f
        for i in range(len(size_f)):
            print(len(size_f))
            print(len(w))
            print(size_f[i][0] * w[i])
            size_n.append([size_f[i][0] / w[i], size_f[i][1] / h[i]])
            

        # Imprimimos el resultado
        print(w)
        print(h)
        print(size_f)
        print(size_old)
        print(size_n)
    
    return center_f,size_f,angle_f,img,masks,vector_df

if __name__=="__main__":
    main(clases_old = clases_old, size_old = size_old)

