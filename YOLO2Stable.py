from collections import Counter
import os
import numpy as np
import cv2
import time
from ultralytics import YOLO
from PIL import Image, ImageDraw

def main():

    # Diccionario para mapear numeros a sus representaciones en palabras
    number_words = {
        1: "One",
        2: "Two",
        3: "Three",
        4: "Four",
        5: "Five",
        6: "Six",
        7: "Seven",
        8: "Eight",
        9: "Nine",
        10: "Ten"
    }

    # Load a model
    model = YOLO('yolov8x-seg.pt')  # load an official model

    # Inicializa la camara
    cap = cv2.VideoCapture(4)  # El argumento 0 selecciona la camara predeterminada

    # Espera 1 segundo
    time.sleep(1)

    # Captura un frame
    ret, frame = cap.read()

    # Verifica si la captura fue exitosa
    if ret:
        results = model(frame, stream = True)
        cv2.imwrite('frame.jpg', frame)
        # Analisis de los resultados
        for r in results:
            names_cls = r.names
            clases = r.boxes.cls.cpu().numpy() #Devuelve un array 1xN, donde [N] es cada clase
            print(clases)
            coordenadas = r.masks.xy
            clases_old = clases

            # Generacion de los strings de objetos para el prompt
            element_count = Counter(clases)  # Contar las repeticiones de elementos

            result_strings = []

            for element, count in element_count.items():
                result_strings.append(str(number_words[count])+ " " + str(names_cls[element]))

            # Unir los resultados en un solo string, usando "and" para el ultimo elemento
            if len(result_strings) > 1:
                result = ", ".join(result_strings[:-1]) + " and " + result_strings[-1]
            else:
                result = result_strings[0]

            print(result)  # Esto imprimira el resultado en la forma deseada

            # Generacion de los prompts
            prompt1 = result + ", top view of a wooden table, photography, ordered, best quality, photorealistic, hyperdetailed, realistic, 4k"
            prompt2 = "Top view of a white table with " + result + ", photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            prompt3 = result + ", view from above of a white table, photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            prompt4 = "View from above of a wooden table with " + result + ", photography, ordered, best quality,  photorealistic, hyperdetailed, realistic, 4k"
            print("Prompt 1: " + prompt1)
            print("Prompt 2: " + prompt2)
            print("Prompt 3: " + prompt3)
            print("Prompt 4: " + prompt4)

            # Genera la imagen binaria de la segmentacion
            center_old = []
            size_old = []
            angle_old = []

            for i in range(len(coordenadas)):
                # Tamaño de la imagen
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                im_pil = Image.fromarray(img)
                width, height = im_pil.size

                # Crea una imagen en blanco
                binary_image = Image.new("L", (width, height), 0)

                # Crea un objeto para dibujar en la imagen
                draw = ImageDraw.Draw(binary_image)

                # Lista de puntos que definen el polígono
                #lista_coordenadas = coordenadas.tolist()
                points_float = coordenadas[i]

                # Convertir los puntos a enteros
                points_int = [(int(round(x)), int(round(y)) ) for x, y in points_float]

                # Dibuja el polígono con relleno 255 y contorno 0
                draw.polygon(points_int, outline=0, fill=255)

                # Guarda la imagen binaria
                binary_image.save("mask_" + str(i) + names_cls[clases[i]] + ".png")   

                cv2_mask = np.array(binary_image)
                #cv2_mask = cv2.cvtColor(cv2_mask, cv2.COLOR_GRAY2BGR)

                ###     AQUI EMPIEZA LA EXTRACCION DE LAS OBB   ###

                # Load the image in OpenCV format
                #img = cv2.imread(img_path)

                # Resize the image to 640x480
                img = cv2.resize(frame, (640, 480))

                # Resize the binary_image to 640x480
                cv2_mask = cv2.resize(cv2_mask, (640, 480))

                # Find the contours of the objects in the binary_image
                contours, _ = cv2.findContours(cv2_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Find the largest contour
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)

                    # Fit a rotated rectangle to the largest contour
                    rect = cv2.minAreaRect(largest_contour)

                    # Calculate the angle
                    if rect[1][0]<rect[1][1]:
                        angle = rect[2] + 90
                    else:
                        angle = rect[2] 

                    # Draw the rotated rectangle on the original image
                    box_points = cv2.boxPoints(rect).astype(int)
                    cv2.drawContours(img, [box_points], 0, (0, 255, 0), 2)  # Draw the green rectangle
                    print(str(rect[0][0]) + ' ' + str(rect[0][1]) + ' ' + str(rect[1][0]) + ' ' + str(rect[1][1]) + ' ' + '(Angle)' + ' ' + str(angle) + '\n' )
                    center_old.append(rect[0])
                    size_old.append(rect[1])
                    angle_old.append(angle)
                    prompts = (prompt1, prompt2, prompt3, prompt4)

                    # Display the image with the oriented bounding box
                    cv2.imshow('Oriented Bounding Box', img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    # Cierra la imagen binaria
                    binary_image.close() 
                    
                else:
                    print("No contours found in the binary_image.")
                    binary_image.close()


    # Libera la camara
    cap.release()

    # Cierra todas las ventanas de OpenCV
    cv2.destroyAllWindows()
    return center_old, size_old, angle_old, prompts, clases_old

if __name__=="__main__":
    main()