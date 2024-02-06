
import numpy as np
import matplotlib.pyplot as plt
import time


def main(center_old, size_old, angle_old, C_stable, angle_f):

    class Rectangle:
        def __init__(self, center: np.ndarray, dims: np.ndarray, angle: float):
            self.corners = self.get_rect_points(center, dims, angle)
            self.area = dims[0] * dims[1]

        @staticmethod
        def get_rect_points(center: np.ndarray, dims: np.ndarray, angle: float):
            """
            returns four corners of the rectangle.
            bottom left is the first conrner, from there it goes
            counter clockwise.
            """
            center = np.asarray(center)
            center[1]= center[1]
            length, breadth = dims
            angle = np.deg2rad(angle)
            corners = np.array([[-length/2, -breadth/2],
                                [length/2, -breadth/2],
                                [length/2, breadth/2],
                                [-length/2, breadth/2]])
            rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
            corners = rot.dot(corners.T) + center[:, None]
            return corners.T

        def is_point_in_collision(self, p: np.ndarray):
            """
            check if a point is in collision with the rectangle.
            """
            def area_triangle(a, b, c):
                return abs((b[0] * a[1] - a[0] * b[1]) + (c[0] * b[1] - b[0] * c[1]) + (a[0] * c[1] - c[0] * a[1])) / 2

            area = 0
            area += area_triangle(self.corners[0], p, self.corners[3])
            area += area_triangle(self.corners[3], p, self.corners[2])
            area += area_triangle(self.corners[2], p, self.corners[1])
            area += area_triangle(self.corners[1], p, self.corners[0])
            return area > self.area
        
        def intersect(self,p1, p2, p3, p4):
            x1,y1 = p1
            x2,y2 = p2
            x3,y3 = p3
            x4,y4 = p4
            denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
            if denom == 0: # parallel
                return None
            ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
            if ua < 0 or ua > 1: # out of range
                return None
            ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
            if ub < 0 or ub > 1: # out of range
                return None
            #x = x1 + ua * (x2-x1)
            #y = y1 + ua * (y2-y1)
            return True
        
        def is_intersect(self, rect_2):
            
            #print("Area st: {}".format(self.area))
            #print("Area rect_2: {}".format(rect_2.area))
            x_self=[]
            y_self=[]
            x=[]
            y=[]
            
            x_self = self.corners[:,0]
            y_self = self.corners[:,1]
            x = rect_2.corners[:,0]
            y = rect_2.corners[:,1]
            
            dx = min(max(x_self), max(x)) - max(min(x_self), min(x))
            dy = min(max(y_self), max(y)) - max(min(y_self), min(y))
            
            
            if (dx>=0) and (dy>=0):
                for i in range(4):
                    print("Valor de i dentro:{}".format(i))
                    for j in range(4):
                        if i==3 and j==3:
                            L = self.intersect(self.corners[3], self.corners[0],rect_2.corners[3], rect_2.corners[0])
                        elif j == 3:
                            L = self.intersect(self.corners[i], self.corners[i+1],rect_2.corners[3], rect_2.corners[0])
                        elif i == 3:
                            L = self.intersect(self.corners[3], self.corners[0],rect_2.corners[j], rect_2.corners[j+1])
                        else:
                            L = self.intersect(self.corners[i], self.corners[i+1],rect_2.corners[j], rect_2.corners[j+1])
                        print(L)
                        if L == True:
                            return dx*dy
                return None
            
        
    def plot_rect(p1, p2, p3, p4, color='r'):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color)
        ax.plot([p2[0], p3[0]], [p2[1], p3[1]], color)
        ax.plot([p3[0], p4[0]], [p3[1], p4[1]], color)
        ax.plot([p4[0], p1[0]], [p4[1], p1[1]], color)
        mid_point = 0.5 * (p1 + p3)
        plt.scatter(mid_point[0], mid_point[1], marker='*')
        plt.xlim([0, 640])
        plt.ylim([480, 0])


    """ posiciones= np.array([(100,300),(300,300),(300,200),(550,300)])
    angles = np.array([0.,20.,50.,175.])
    size = np.array([(100,50),(100,50),(120,30),(150,60)]) """
    posiciones= np.array(center_old)
    angles = np.array(angle_old)
    print(angles)
    size = np.array(size_old)
    print("Size:  {}".format(size))
    listaObjects=[]

    """ posFinales = np.array([(100,300),(200,200),(540,305),(525,400)])
    anglesFinales = np.array([0.,20.,45.,175.]) """
    posFinales = np.array(C_stable)
    anglesFinales = np.array(angle_f)
    print(anglesFinales)
    


    for i in range(len(posiciones)):
        st = Rectangle(posiciones[i],size[i]*0.9,angles[i])
        listaObjects.append(st)
        #gripper = Rectangle((167, 400),(21,16), 45)


    for l in range(len(listaObjects)):
        ax = plt.subplot(111)
        plot_rect(*listaObjects[l].corners)
    plt.show()
    Ordered = []
    indx= list(range(0,len(angles)))
    check=[]

    NotCollided = True
    #Check the collision of the objects
    k=0
    i=indx[k]

    while len(Ordered) < len(indx):
        NotCollided = True
        print("Valor de k:{}".format(k))
        print("Valor de i:{}".format(i))
        print("Valor de angle:{}".format(anglesFinales[i]))
        st_move = Rectangle(posFinales[i],size[i]*0.9,anglesFinales[i])
        """ if i ==3:
            print(posFinales[i])
            plt.figure()
            ax = plt.subplot(111)
            plot_rect(*st_move.corners, color='r')
            plt.show() """
        check = indx.copy()
        print("Check:{}".format(check))
        check.remove(i)
        print("Check despues:{}".format(check))
        #print("Lista de check:{}".format(check))
        for j in range(len(check)):
            NotCollided = True
            print("Valor de j:{}".format(j))
            f=check[j]
            print("Valor de f:{}".format(f))
            plt.close()
            plt.figure()
            ax = plt.subplot(111)
            collision = listaObjects[f].is_intersect(st_move)
            plot_rect(*listaObjects[f].corners, color='b')
            plot_rect(*st_move.corners, color='r')
            print(collision)
            plt.show()
            
            if collision != None:
                print("Hay colision")
                print(i)
            #listaObjects[i].move(posiciones[i])
                
                """ print("Index:{}".format(indx))
                print("Valor que quita:{}".format(indx.pop(i))) """
                indx.append(indx.pop(i))
                print("Indx:{}".format(indx))
                i=indx[k]
                print(i)
                break
            print("Hola")
            if NotCollided == True and j >= len(check)-1:
                Ordered.append(listaObjects[i])
                listaObjects[i]=st_move #Update the position of the object if tehre is no collision
                if k < len(indx)-1:
                    k=k+1
                    i=indx[k]
                    
    #print(indx)
    return indx

if __name__=="__main__":
    main(center_old, size_old, angle_old, C_stable, angle_f)



