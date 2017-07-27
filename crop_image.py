import numpy as np
import cv2

def crop_image(numImg):
    ##These methods are for the picture resizing
    global mouseX, mouseY, numC
    mouseX = []
    mouseY = []
    numC = 0

    ###Event 4 means that the right key was clicked
    ###This saves the points that are clicked on the image

    def choosePoints(event,x,y,flags,param):

        global mouseX, mouseY, numC

        if event == 4:
            numC += 1
            mouseX.append(x)
            mouseY.append(y)

    

    #Opens up the color image for user to click on

    imgC = cv2.imread('img/clearn%d.png' %numImg, -1)
    cv2.imshow('image',imgC)
    cv2.setMouseCallback('image', choosePoints)


    #checks and makes sure 2 points were clicked
    #if 2 points were clicked it exits the loop

    while(numC != 2):
        key = cv2.waitKey(1) & 0xFF
    

    #Closes color image once user clicks twice
    cv2.destroyAllWindows()
    print(mouseX, mouseY, "mouse X and mouse Y")
    return mouseX, mouseY