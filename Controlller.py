import YOLO2Stable
import Test_dx
#import Stable
import Stable_img
import Stable2YOLO
import Collision
import Cluster
import MaskGenerator
import plotImage
import InPaint
import time
import math 
import numpy as np
import matplotlib.pyplot as plt
from Task_Parameterized_Gaussian_Mixture_Model import MultiGMM
from scipy.io import savemat
#from Task_Parameterized_Gaussian_Mixture_Model import MultiGMM


def transitions(n):
    # Create an original list with n elements
    original_list = list(range(1, n + 1))

    # Create a new list to store the results
    modified_list = []

    # Add a zero at the beginning of the list
    modified_list.append(0)

    # Iterate over the original list
    for value in original_list:
        # Add the current value to the modified list
        modified_list.append(value)
        # Add a zero after each value, except at the end
        if value != original_list[-1]:
            modified_list.append(0)

    # Add a zero at the end of the list
    modified_list.append(0)

    # Print the original list and the modified list
    print("Original List:", original_list)
    print("Modified List:", modified_list)
    return modified_list



#* YOLO2Stable-> we obtain the values of what camera see
center_old, size_old, angle_old, prompts, clases_old, data_A,img,masks,vector_d,Z_Points,Points = Test_dx.main()
print(center_old)
print("Angles: {}".format(angle_old))
print(len(masks))
""" Gx =[]
Gy =[] """
C_img = []
UD =[]
mg =[]
for i in range(len(masks)):
    #* If up=1, hacia arriba
    #cx,cy,up,mango=Cluster.main(img,masks[i],center_old[i][1])
    mango=False
    up=1
    cx=center_old[i][0]
    cy=center_old[i][1]
    """ Gx.append(cx)
    Gy.append(cy) """
    C_img.append([cx,cy])
    UD.append(up)
    mg.append(mango)
    
print("*******************")
""" print(Gx)
print(Gy)
print(UD) """
#* Generate the .txt with the prompts
Bot_prompts = "Bot_prompts.txt"
with open (Bot_prompts, 'w') as archivo:
    for i in range(len(prompts)):
        fila = prompts[i]
        archivo.write(fila +'\n')

#* Execute Stable Baseline
#Stable_img.main() #! Comentar esto para que vaya mas rápido
#! Reescalar imagenes

#* Stable2YOLO-> select the correct image with YOLO
center_f,size_f,angle_f,img_stable,masks_stable, vector_df = Stable2YOLO.main(clases_old,size_old)  #! Añadir size_old, para hacer el resize

#? InPaint process
InPaint.main(img_stable,masks_stable)

C_stable =[]
print(len(masks_stable))
for i in range(len(masks_stable)):
    cx,cy = MaskGenerator.main(masks_stable[i])
    C_stable.append([cx,cy])

print("*******************")
print(C_stable)

indexYolo = list(range(len(clases_old)))

index = Collision.main(center_old, size_old, angle_old, C_stable, angle_f)

print(indexYolo)
print(index)
#index = [1,0]
#* Correct the angle depending of the direction
for i in range(len(UD)):
    if UD[i] == False:
        angle_old[i] = angle_old[i] + 180
    else:
        angle_old[i]=angle_old[i]

if indexYolo != index:
    #print(center_old)
    # Crear una lista temporal reorganizando los elementos según el nuevo orden
    temp = [center_old[i] for i in index]
    temp_a = [center_f[i] for i in index]
    temp_b = [angle_old[i] for i in index]
    temp_c = [C_img[i] for i in index]
    temp_d = [angle_f[i] for i in index]
    temp_e = [mg[i] for i in index]
    temp_f = [Points[i] for i in index]
    temp_g = [Z_Points[i] for i in index]
    temp_h = [UD[i] for i in index]

    # Actualizar la lista original con la lista temporal
    center_old = temp
    center_f = temp_a
    angle_old = temp_b
    C_img = temp_c
    angle_f = temp_d
    mg = temp_e
    Points = temp_f
    Z_Points = temp_g
    UD = temp_h
else: 
    print("son iguales")

print("old:  {}".format(center_old))
print("cimg:  {}".format(C_img))

#? Calculate dx dy for IL algortihm
DX = []
DY = []
DXF = []
DYF=[]
angle_old_TP = []
angle_f_TP = []
Dir_LR = []
for i in range(len(angle_old)):
    
    #? We assume that Stable imgs objects are always Up
    if UD[i] == True:
        if center_old[i][0] < center_f[i][0]:
            angle_old_TP.append(angle_old[i] - 90)
            angle_f_TP.append(angle_f[i] + 90)
            direction = True    #*  If True: Left ---> Rigth Movement   #? Direccion meter en TPGMM
        else:
            angle_old_TP.append(angle_old[i] + 90)
            angle_f_TP.append(angle_f[i] - 90)
            direction = False   #*  If False: Right ---> Left Movement
    else:
        if center_old[i][0] < center_f[i][0]:
            angle_old_TP.append(angle_old[i] + 90)
            angle_f_TP.append(angle_f[i] + 90)
            direction = True    #*  If True: Left ---> Rigth Movement   #? Direccion meter en TPGMM
        else:
            angle_old_TP.append(angle_old[i] - 90)
            angle_f_TP.append(angle_f[i] - 90)
            direction = False   #*  If False: Right ---> Left Movement

    dx_old = np.cos(np.deg2rad(angle_old_TP[i]))/8
    dy_old = np.sin(np.deg2rad(angle_old_TP[i]))/8

    dx_f = np.cos(np.deg2rad(angle_f_TP[i]))/8
    dy_f = np.sin(np.deg2rad(angle_f_TP[i]))/8

    DX.append(dx_old)
    DY.append(dy_old)
    DXF.append(dx_f)
    DYF.append(dy_f)
    Dir_LR.append(direction)


#? Transform coordinates to TPGMM (Using centroid as grasping point)
C_final = []
if len(C_img) == len(center_old):
    for i in range(len(C_img)):
        if mg[i]:   #* Only to correct handle objects grasp point
            #* Calculate the offset of the original image
            diff = math.sqrt((center_old[i][0] - C_img[i][0])**2 + (center_old[i][1] - C_img[i][1])**2)
            print("old:  {}".format(center_old[i]))
            print("c:  {}".format(C_img[i]))
            #* Decompose with the angle
            diff_x = diff * math.cos(np.deg2rad(angle_f[i]))
            diff_y = diff * math.sin(np.deg2rad(angle_f[i]))
            print("DIFY: {}".format(diff_y))
            if UD[i]:
                new_C = ([center_f[i][0] + diff_x, center_f[i][1] + diff_y])
            else:
                new_C = ([center_f[i][0] - diff_x, center_f[i][1] - diff_y])
        else:
            new_C = ([center_f[i][0], center_f[i][1]])
            
        C_final.append(new_C)
        
else: 
    print("Error en el numero de datos")

for i in range(len(center_old)):
    plotImage.main(img, center_old[i], C_img[i])
    plotImage.main(img_stable, center_f[i], C_final[i])


""" # Plotear los puntos y la línea
plt.plot([C_img[0][0], C_stable[0][0]], [C_img[0][1], C_stable[0][1]], label='Original AB')
plt.scatter([C_img[0][0], C_stable[0][0], C_final[0][0]], [C_img[0][1], C_stable[0][1], C_final[0][1]], color='red')
plt.plot([C_stable[0][0], C_final[0][0]], [C_stable[0][1], C_final[0][1]], linestyle='dashed', label=f'Offset ({diff})')

# Etiquetas y leyenda
plt.text(C_img[0][0], C_img[0][1], '  A', verticalalignment='bottom', horizontalalignment='right')
plt.text(C_stable[0][0], C_stable[0][1], '  B', verticalalignment='bottom', horizontalalignment='right')
plt.text(C_final[0][0], C_final[0][1], '  Nuevo', verticalalignment='bottom', horizontalalignment='right')

plt.xlabel('Coordenada X')
plt.ylabel('Coordenada Y')
plt.legend()
plt.grid(True)
plt.show() """


#? Calling the imitation learning based on TPGMM code
l = 2 + (len(DX)+(len(DX)-1))
print("Longuitud: {}".format(l))
k=0
j = 0
pathRobot = []
Init_O = []
Final_O = []
Altura = []
trans = transitions(len(DX))
for i in range(l):
    #* Always start form the same point
    #* The rest of the points are the normal one
    print("---------------------------")
    print(C_final)
    print(C_img)
    print("Valor de i: {}".format(i))
    if (i >= 1) and (i != l-1):
        if trans[i]!=0:
            distance = math.dist(C_final[k],C_img[k])
            print("Distancia entre valores: {}".format(distance))
            time.sleep(2)
            rData = MultiGMM.main(C_img[k],C_final[k],DX[k],DY[k],DXF[k],DYF[k],Dir_LR[k],distance)
            k = k+1 #! Creo que va aqui
            """ C_final = copyFinal
            C_img = copyImg """
        elif trans[i]==0:
            print("Transicion entre estados")
            #! Aqui hay que meter el valor de transicion de ir de un punto de dejada a un punto de recogida
            #! Puede ocurrir que el punto de salida este a la izquierda o a la derecha del punto de llegada, hay que ver los dos casos
            C_final_l = [C_img[k][0],C_img[k][1]]
            C_img_l = [C_final[k-1][0],C_final[k-1][1]]
            if C_img_l[0] < C_final_l[0]:
                Dir_LR_l = True # Izq-Dcha
                if (DXF[k-1] <= 0.0 and DYF[k-1] <=1.0):
                    DX_l = DXF[k-1] *-1
                    DY_l = DYF[k-1] *-1
                else:
                    DX_l = DXF[k-1] 
                    DY_l= DYF[k-1] 

                if (DX[k] <= 0.0 and DY[k] <=1.0):
                    DXF_l = DX[k] 
                    DYF_l = DY[k]
                else:
                    DXF_l = DX[k] *-1
                    DYF_l= DY[k] *-1
            else:
                Dir_LR_l = False
                if (DXF[k-1] >= 0.0 and DYF[k-1] <=1.0):
                    DX_l = DXF[k-1] *-1
                    DY_l = DYF[k-1] *-1
                else:
                    DX_l = DXF[k-1]
                    DY_l= DYF[k-1]

                if (DX[k] >= 0.0 and DY[k] <=1.0):
                    DXF_l = DX[k]
                    DYF_l = DY[k]
                else:
                    DXF_l = DX[k]*-1
                    DYF_l= DY[k]*-1
            distance = math.dist(C_final_l,C_img_l)
            print("Distancia entre valores: {}".format(distance))
            time.sleep(2)
            rData = MultiGMM.main(C_img_l,C_final_l,DX_l,DY_l,DXF_l,DYF_l,Dir_LR_l,distance)
    elif i==0:
        C_final_l = [C_img[0][0],C_img[0][1]]
        if (DX[0] <= 0.0 and DY[0] <=1.0):
            DXF_l = DX[0]
            DYF_l = DY[0]
        else:
            DXF_l = DX[0] * -1
            DYF_l = DY[0] * -1
        Dir_LR_l = True
        DX_l = 0.0
        DY_l = 0.1
        C_img_l = [70,300]
        distance = math.dist(C_final_l,C_img_l)
        print("Distancia entre valores: {}".format(distance))
        time.sleep(2)
        rData = MultiGMM.main(C_img_l,C_final_l,DX_l,DY_l,DXF_l,DYF_l,Dir_LR_l,distance)
    elif i==l-1:
        C_final_l = [70,300]
        C_img_l = [C_final[k-1][0],C_final[k-1][1]]
        if C_img_l[0] < C_final_l[0]:
            Dir_LR_l = True
        else:
            Dir_LR_l = False

        if (DX[0] <= 0.0 and DY[0] <=1.0):
            DX_l = DX[0]
            DY_l = DY[0]
        else:
            DX_l = DX[0] * -1
            DY_l = DY[0] * -1
        
        DXF_l = 0.0
        DYF_l = 0.1
        distance = math.dist(C_final_l,C_img_l)
        print("Distancia entre valores: {}".format(distance))
        time.sleep(2)
        rData = MultiGMM.main(C_img_l,C_final_l,DX_l,DY_l,DXF_l,DYF_l,Dir_LR_l,distance)
    
    if i ==0:
        print("Inicio")
        Init_O = 90
        Final_O= angle_old[i]
        Altura = 1
    elif i== l-1:
        Init_O=angle_f[j]
        Final_O=90
        Altura = 1
    elif trans[i]==0 and i!=0 and i!=l-1:
        print("elif)")
        j=j+1
        print(j)
        Init_O=angle_f[j-1]
        Final_O=angle_old[j]
        Altura = Z_Points[j]
        
    else:
        print("else)")
        print(j)
        Init_O=angle_old[j]
        Final_O=angle_f[j]
        Altura = Z_Points[j]
        
        

    mdic = {"Patht": rData[0], "Pathx": rData[1],"Pathy": rData[2], "Init_O": Init_O, "Final_O": Final_O, "Heigth":Altura}
    pathRobot.append(mdic)


#? Exportar datos
mdicfin = {"value": pathRobot}
print("Angulos camara: {}".format(angle_old))
print("Angulos Stable: {}".format(angle_f))
savemat("MatlabData.mat", mdicfin)