import numpy as np
import cv2 as cv

def main(img, mask):
    # Crear una imagen en blanco del mismo tamaño que las imágenes originales
    result_image = np.zeros_like(mask[0])
    background = cv.imread('background.png')
    background = cv.resize(background, (640, 480))

    # Sumar las imágenes
    for i in range(len(mask)):
        result_image = cv.add(result_image, mask[i])

    # Mostrar la imagen resultante
    cv.imshow('Result Image', result_image)

    # Crear una máscara inversa para los objetos
    inverse_mask = cv.bitwise_not(result_image)

    # Aplicar la máscara inversa a la imagen original para obtener el fondo blanco
    #background = np.ones_like(img, dtype=np.uint8) * 255
    white_background = cv.bitwise_and(background, background, mask=inverse_mask)

    # Agregar los objetos de la imagen original al fondo blanco
    result_image = cv.bitwise_or(white_background, cv.bitwise_and(img, img, mask=result_image))

    # Mostrar la imagen final con fondo blanco
    cv.imshow('Final Result', result_image)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main(img, mask)
