import pygame
import csv
import math
import numpy as np
import time
import matplotlib.pyplot as plt

# Inicializar Pygame
pygame.init()

# Configuración de la ventana
width, height = 800, 600
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Captura de Datos 2D")

# Definir el rango deseado en X e Y
x_min, x_max = -1.0, 1.0
y_min, y_max = -1.0, 1.0

# Variables para capturar datos
capturing = False
data = []
current_path = []

# Función para mapear coordenadas del ratón al rango deseado
def map_mouse_to_range(mouse_pos):
    x = x_min + (mouse_pos[0] / width) * (x_max - x_min)
    y = y_min + (1.0 - mouse_pos[1] / height) * (y_max - y_min)
    return x, y

# Función para calcular la matriz de rotación 3x3 a partir del ángulo de orientación
def rotation_matrix(orientation,dx,dy):
    c = math.cos(orientation)
    s = math.sin(orientation)
    return np.array([[1, 0, 0], [0, dx, dy], [0, dy, dx]])
    #return np.array([[1, 0, 0], [0, c, -s], [0, s, c]]) #Rotacion sobre x
    #return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) # Rotacio nsobre z

# Función para calcular la orientación entre dos puntos
def calculate_orientation(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    return math.atan2(dy, dx),dx,dy

# Loop principal
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Botón izquierdo del ratón
                capturing = True
                start_pos = map_mouse_to_range(event.pos)
                data.append(["Start", start_pos, None, np.identity(3)])  # Orientación inicial se establecerá como matriz de identidad
                current_path = [start_pos]
            elif event.button == 3:  # Botón derecho del ratón
                capturing = True
                end_pos = map_mouse_to_range(event.pos)
                data.append(["End", end_pos, None, np.identity(3)])  # Orientación final se establecerá como matriz de identidad
                current_path = [end_pos]
        elif event.type == pygame.MOUSEMOTION and capturing:
            pos = map_mouse_to_range(event.pos)
            orientation,dx,dy = calculate_orientation(current_path[-1], pos)
            data.append(["Intermediate", pos, orientation, rotation_matrix(orientation,dx,dy)])
            current_path.append(pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1 or event.button == 3:
                capturing = False

    # Actualizar la pantalla
    window.fill((255, 255, 255))

    for label, pos, orientation, _ in data:
        color = (0, 0, 0) if label == "Intermediate" else (255, 0, 0)
        pygame.draw.circle(window, color, (int((pos[0] - x_min) * width / (x_max - x_min)), int((1.0 - (pos[1] - y_min) / (y_max - y_min)) * height)), 5)

    if len(current_path) >= 2:
        pygame.draw.lines(window, (0, 0, 0), False, [(int((pos[0] - x_min) * width / (x_max - x_min)), int((1.0 - (pos[1] - y_min) / (y_max - y_min)) * height)) for pos in current_path])

    pygame.display.flip()

    # Comprobar si el último punto capturado fue un punto final
    if data and data[-1][0] == "End":
        running = False

# Guardar los datos en un archivo CSV (HumanDemonstration.csv)
with open("HumanDemonstration.csv", "w", newline="") as csvfile:
    csvwriter = csv.writer(csvfile)
    for label, pos, orientation, rotation in data:
        orientation_degrees = math.degrees(orientation) if orientation is not None else None
        csvwriter.writerow([label, pos[0], pos[1], orientation_degrees, rotation[0, 0], rotation[0, 1], rotation[1, 0], rotation[1, 1]])

pygame.quit()
data[0]=data[1]
data[-1]=data[-2]

#Calculate the orientation
result_init = tuple(x - y for x, y in zip(data[20][1], data[0][1]))
result_end = tuple(x - y for x, y in zip(data[-20][1], data[-1][1]))
initRot = rotation_matrix(0,result_init[1],result_init[0])
endRot = rotation_matrix(0,result_end[1],result_end[0])

#time.sleep(10000)

#? Save in txt
# Supongamos que tienes una lista de puntos llamada data
# y quieres que tenga 75 puntos.

A_init = initRot#data[0][3]
b_init = data[0][1]
A_end = endRot#data[-1][3]
b_end = data[-1][1]
print(A_init)
print(A_end)
print(b_init[0])
print(b_end)


# Calcula cuántos puntos adicionales necesitas o cuántos puntos debes eliminar.
total_puntos_deseados = 75
x_originales =[]
y_originales = []
puntos_actuales = len(data)
print("Puntos path demo: {}".format(puntos_actuales))
originales = [punto[1] for punto in data]
""" print("Originales: {}".format(originales)) """
for i in range(len(originales)):
    x_originales.append(originales[i][0])
    y_originales.append(originales[i][1])
puntos_a_agregar = total_puntos_deseados - puntos_actuales

if puntos_a_agregar > 0:
    #! Corregir en un futuro
    print(data[-1][1][0])
    distancia_x = (data[-1][1][0] - data[0][1][0]) / (total_puntos_deseados - 1)
    distancia_y = (data[-1][1][1] - data[0][1][1]) / (total_puntos_deseados - 1)
    
    for i in range(puntos_a_agregar):
        nuevo_x = data[0][1][0] + (i + 1) * distancia_x
        nuevo_y = data[0][1][1] + (i + 1) * distancia_y
        data.insert(-1, (nuevo_x, nuevo_y))
        
elif puntos_a_agregar < 0:
    # Elimina puntos intermedios entre el primero y el último.
    puntos_a_eliminar = abs(puntos_a_agregar)
    
    if puntos_a_eliminar >= puntos_actuales - 2:
        # No se pueden eliminar puntos intermedios sin afectar el primero y el último.
        print("No es posible eliminar puntos sin afectar el primero y el último.")
    else:
        # Elimina puntos intermedios de manera equitativa.
        espaciado = (puntos_actuales - 2) // puntos_a_eliminar
        indices_a_eliminar = [1 + i * espaciado for i in range(puntos_a_eliminar)]
        indices_a_eliminar.reverse()  # Reversa la lista de índices para evitar problemas con el rango
        for indice in indices_a_eliminar:
            del data[indice]
#print("Puntos finales: {}".format(len(data)))
# Ahora, data tiene 75 puntos sin modificar el primero ni el último.

#? Plot both paths
x_news=[]
y_news=[]
news = [punto[1] for punto in data]

for i in range(len(news)):
    try:
        x_news.append(news[i][0])
        y_news.append(news[i][1])
    except TypeError as e:
        continue

print("Puntos finales: {}".format(len(x_news)))


plt.figure(figsize=(12, 6))
plt.subplot(121)
plt.scatter(x_originales, y_originales, label='Datos Originales')
plt.title('Datos Originales')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.subplot(122)
plt.scatter(x_news, y_news, label='Datos Procesados')
plt.title('Datos Procesados')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()

plt.tight_layout()
plt.show()

#! Save the data

#* Datos para los init
nombre_archivo = "sample20_frame1_b.txt"

with open(nombre_archivo, 'w') as archivo:
    fila_1 = ','.join([str(0)] * 75)
    archivo.write(fila_1 + '\n')
    
    fila_2 = ','.join([str(b_init[0])] * 75)
    archivo.write(fila_2 + '\n')

    fila_3 = ','.join([str(b_init[1])] * 75)
    archivo.write(fila_3+ '\n')

Archivo_A = "sample20_frame1_A.txt"

with open(Archivo_A, 'w') as archivo:
    fila_1 = ','.join([str(A_init[0][0])+','+str(A_init[0][1])+','+str(A_init[0][2])] * 75)
    archivo.write(fila_1 + '\n')
    
    fila_2 = ','.join([str(A_init[1][0])+','+str(A_init[1][1])+','+str(A_init[1][2])] * 75)
    archivo.write(fila_2 + '\n')
    
    fila_3 = ','.join([str(A_init[2][0])+','+str(A_init[2][1])+','+str(A_init[2][2])] * 75)
    archivo.write(fila_3 + '\n')



#* Datos para los end
nombre_archivo2 = "sample20_frame2_b.txt"

with open(nombre_archivo2, 'w') as archivo:
    fila_1 = ','.join([str(0)] * 75)
    archivo.write(fila_1 + '\n')
    
    fila_2 = ','.join([str(b_end[0])] * 75)
    archivo.write(fila_2 + '\n')

    fila_3 = ','.join([str(b_end[1])] * 75)
    archivo.write(fila_3+ '\n')

Archivo_A2 = "sample20_frame2_A.txt"

with open(Archivo_A2, 'w') as archivo:
    fila_1 = ','.join([str(A_end[0][0])+','+str(A_end[0][1])+','+str(A_end[0][2])] * 75)
    archivo.write(fila_1 + '\n')
    
    fila_2 = ','.join([str(A_end[1][0])+','+str(A_end[1][1])+','+str(A_end[1][2])] * 75)
    archivo.write(fila_2 + '\n')
    
    fila_3 = ','.join([str(A_end[2][0])+','+str(A_end[2][1])+','+str(A_end[2][2])] * 75)
    archivo.write(fila_3 + '\n')

DatosPath2 = "sample20_Data.txt"
with open(DatosPath2, 'w') as archivo:
    valores = [round(x, 2) for x in [0.01 * i for i in range(1, 76)]]
    fila_1 = ','.join(map(str, valores))
    archivo.write(fila_1 + '\n')
    
    fila_2 = ','.join(map(str, x_news))
    archivo.write(fila_2 + '\n')
    
    fila_3 = ','.join(map(str, y_news))
    archivo.write(fila_3 + '\n')