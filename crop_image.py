import numpy as np
import cv2

def crop_image(num_img):
    ##These methods are for the picture resizing
    global mouse_X, mouse_Y, numC
    mouse_X = []
    mouse_Y = []
    numC = 0

    ###Event 4 means that the right key was clicked
    ###This saves the points that are clicked on the image

    def choosePoints(event,x,y,flags,param):

        global mouse_X, mouse_Y, numC

        if event == 4:
            numC += 1
            mouse_X.append(x)
            mouse_Y.append(y)

    

    #Opens up the color image for user to click on

    imgC = cv2.imread('img/clearn%d.png' %num_img, -1)
    cv2.imshow('image',imgC)
    cv2.setMouseCallback('image', choosePoints)


    #checks and makes sure 2 points were clicked
    #if 2 points were clicked it exits the loop

    while(numC != 2):
        key = cv2.waitKey(1) & 0xFF
    

    #Closes color image once user clicks twice
    cv2.destroyAllWindows()
    print(mouse_X, mouse_Y, "mouse X and mouse Y")
    return mouse_X, mouse_Y