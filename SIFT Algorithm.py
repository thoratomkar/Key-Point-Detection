# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:57:25 2018

@author: omkar
"""
import math
import cv2
import numpy as np
import timeit

PI = math.pi

def initialize(m, n):
    #initializing a 2d matrix to store the feature points of each octave
    blank = []
    for i in range(m):        
        rowList = []
        for j in range(n):                        
            
            rowList.append(0)
        blank.append(rowList)
    return blank
# storing all the bandwidth parameters
def octave_creation():
    octave=[]
    s = [1/2,math.sqrt(2),2*math.sqrt(2),4*math.sqrt(2)]
    for i in range(0,4):
        rowList = [] 
        r = s[i]*math.sqrt(2)
        for j in range(0,5):             
            rowList.append(r)
            r = r*math.sqrt(2)
        octave.append(rowList)
    return octave

def sigma_calc(octave_number, octave_value, octave):
    return octave[octave_number][octave_value]

def gaussian_filter_calc(sigma):
    gaussian_filter_temp = []
    total_sum = 0
    for i in range(0,7):
            rowList = []
            for j in range(0,7):
                x = float(1/(2*PI*sigma**2))
                y = float(x*math.exp(-((j-3)**2+(3-i)**2)/(2*sigma**2)))
                total_sum = total_sum + y
                rowList.append(y)
            gaussian_filter_temp.append(rowList)
    #normalizing the gaussian filter            
    gaussian_filter = np.array(gaussian_filter_temp, dtype=np.float)/total_sum
    
    return gaussian_filter

def imgblur(gaussian_filter, img, a, b, m, n):
    temp_image = []
    for i in range(3,m-3):
            rowList = []
            for j in range(3,n-3):
                g = (gaussian_filter[0][0]*img[i-3][j-3]) + (gaussian_filter[0][1]*img[i-3][j-2]) + (gaussian_filter[0][2]*img[i-3][j-1]) + (gaussian_filter[0][3]*img[i-3][j]) + (gaussian_filter[0][4]*img[i-3][j+1]) + (gaussian_filter[0][5]*img[i-3][j+2]) + (gaussian_filter[0][6]*img[i-3][j+3]) +\
                    (gaussian_filter[1][0]*img[i-2][j-3]) + (gaussian_filter[1][1]*img[i-2][j-2]) + (gaussian_filter[1][2]*img[i-2][j-1]) + (gaussian_filter[1][3]*img[i-2][j]) + (gaussian_filter[1][4]*img[i-2][j+1]) + (gaussian_filter[1][5]*img[i-2][j+2]) + (gaussian_filter[1][6]*img[i-2][j+3]) +\
                    (gaussian_filter[2][0]*img[i-1][j-3]) + (gaussian_filter[2][1]*img[i-1][j-2]) + (gaussian_filter[2][2]*img[i-1][j-1]) + (gaussian_filter[2][3]*img[i-1][j]) + (gaussian_filter[2][4]*img[i-1][j+1]) + (gaussian_filter[2][5]*img[i-1][j+2]) + (gaussian_filter[2][6]*img[i-1][j+3]) +\
                    (gaussian_filter[3][0]*img[i][j-3]) + (gaussian_filter[3][1]*img[i][j-2]) + (gaussian_filter[3][2]*img[i][j-1])+(gaussian_filter[3][3]*img[i][j])+(gaussian_filter[3][4]*img[i][j+1])+(gaussian_filter[3][5]*img[i][j+2])+(gaussian_filter[3][6]*img[i][j+3]) +\
                    (gaussian_filter[4][0]*img[i+1][j-3]) + (gaussian_filter[4][1]*img[i+1][j-2]) + (gaussian_filter[4][2]*img[i+1][j-1]) + (gaussian_filter[4][3]*img[i+1][j]) + (gaussian_filter[4][4]*img[i+1][j+1]) + (gaussian_filter[4][5]*img[i+1][j+2]) + (gaussian_filter[4][6]*img[i+1][j+3]) +\
                    (gaussian_filter[5][0]*img[i+2][j-3]) + (gaussian_filter[5][1]*img[i+2][j-2]) + (gaussian_filter[5][2]*img[i+2][j-1]) + (gaussian_filter[5][3]*img[i+2][j]) + (gaussian_filter[5][4]*img[i+2][j+1]) + (gaussian_filter[5][5]*img[i+2][j+2]) + (gaussian_filter[5][6]*img[i+2][j+3]) +\
                    (gaussian_filter[6][0]*img[i+3][j-3]) + (gaussian_filter[6][1]*img[i+3][j-2]) + (gaussian_filter[6][2]*img[i+3][j-1]) + (gaussian_filter[6][3]*img[i+3][j]) + (gaussian_filter[6][4]*img[i+3][j+1]) + (gaussian_filter[6][5]*img[i+3][j+2]) + (gaussian_filter[6][6]*img[i+3][j+3])
                rowList.append(g)   
            temp_image.append(rowList)
    c = str(a) + str(b)           
    blur_img = np.array(temp_image, dtype=np.uint8)     
    
    #cv2.imshow('blur',m_img)
    cv2.imwrite('Blur_' + c + '.jpg', blur_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

def resize(img, m, n, a):
    a = a + 1
    img_resize_temp = []
    for i in range(0,m):
        rowList = []
        for j in range(0,n):
            if i%(2**a) == 0 and j % (2**a) == 0:
                rowList.append(img[i][j])
            else:
                pass
        img_resize_temp.append(rowList)
        
    temp = img_resize_temp[::2**a]    
    img_resize = np.array(temp, dtype=np.uint8)
    
    return img_resize

def Generating_Blur_Images(img, m, n,octave):
    for a in range(0,4):
        for b in range(0,5):
            sigma = sigma_calc(a, b, octave)
            gf = gaussian_filter_calc(sigma)
            imgblur(gf, img, a, b, m, n)
            
        img = cv2.imread('task2.jpg',0)
        m = img.shape[0]
        n = img.shape[1]
        img = resize(img, m, n, a)
        m = img.shape[0]
        n = img.shape[1]

def Generating_DoG(dog_array):
    for a in range(0,4):
        rowList = []
        for b in range(0,4):
            e = int(b+1)
            d='Blur_' + str(a) + str(b) + '.jpg'
            img1 = cv2.imread(d,0)
            c='Blur_' + str(a) + str(e) + '.jpg'
            img2 = cv2.imread(c,0)
            
            x = DoG(img2, img1, a, b) 
            rowList.append(x)
        dog_array.append(rowList)
        
def DoG(img2, img1, a, b):
    m = img1.shape[0]
    n = img1.shape[1]
    dog_temp = []
    
    #generating a empty matrix to store the DoG
    for i in range(0,m):
        rowList = []
        for j in range(0,n):
            x = int(img2[i][j]) - int(img1[i][j])
            rowList.append(x)
        dog_temp.append(rowList)
        
    dog = np.array(dog_temp, dtype=np.uint8)
    c = str(a) + str(b)  
    #cv2.imshow('dog'+c+'.jpg',dog1)
    cv2.imwrite('DoG_' + c + '.jpg', dog)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return dog
    
def Generating_KeyPoints():
    #Reading 3 Images at a time to detect keypoints
    
    for a in range(0,4):        
        for b in range(1,3):
            c = int(b-1)
            d = 'DoG_' + str(a) + str(c) + '.jpg'
            prev_img = cv2.imread(d,0)
            e = 'DoG_' + str(a) + str(b) + '.jpg'
            current_img = cv2.imread(e,0)
            f = int(b+1)
            g = 'DoG_' + str(a) + str(f) + '.jpg'
            next_img = cv2.imread(g,0)
            KeyPoint(prev_img, current_img, next_img, a)        
        #displaying final image of each octave
    
    final_image = np.array(blank, dtype=np.uint8)
    d = 'Final Octave_' + str(a) + '.jpg'
    cv2.imshow(d,final_image)
    cv2.imwrite(d,final_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def KeyPoint(prev_img, current_img, next_img, a):
    m = current_img.shape[0]
    n = current_img.shape[1]
    #iterating through each pixel in the current image
    for i in range(1,m-1):
        for j in range(1,n-1):            
            temp = current_img[i][j]
            #setting counter to check if the conditions are satisfied
            countermin = False
            countermax = False
            #comparing 8 neighbouring values
            if temp < current_img[i-1][j-1] and temp < current_img[i-1][j] and temp < current_img[i-1][j+1] and temp < current_img[i][j-1] and temp < current_img[i][j+1] and temp < current_img[i+1][j-1] and temp < current_img[i+1][j] and temp < current_img[i+1][j+1]:
                countermin = True
            if temp > current_img[i-1][j-1] and temp > current_img[i-1][j] and temp > current_img[i-1][j+1] and temp > current_img[i][j-1] and temp > current_img[i][j+1] and temp > current_img[i+1][j-1] and temp > current_img[i+1][j] and temp > current_img[i+1][j+1]:
                countermax = True
            
            #comparing pixels of the lower image
            if countermin == True or countermax == True:
                if countermin == True:
                    if temp < prev_img[i-1][j-1] and temp < prev_img[i-1][j] and temp < prev_img[i-1][j+1] and temp < prev_img[i][j-1] and temp < prev_img[i][j+1] and temp < prev_img[i+1][j-1] and temp < prev_img[i+1][j] and temp < prev_img[i+1][j+1]:
                        countermin= True
                    else:
                        countermin = False
                if countermax == True:        
                    if temp > prev_img[i-1][j-1] and temp > prev_img[i-1][j] and temp > prev_img[i-1][j+1] and temp > prev_img[i][j-1] and temp > prev_img[i][j+1] and temp > prev_img[i+1][j-1] and temp > prev_img[i+1][j] and temp > prev_img[i+1][j+1]:
                        countermax == True
                    else:
                        countermax = False            
            
            #comparing pixels of the higher image
            if countermin == True or countermax == True:
                if countermin == True:
                    if temp < next_img[i-1][j-1] and temp < next_img[i-1][j] and temp < next_img[i-1][j+1] and temp < next_img[i][j-1] and temp < next_img[i][j+1] and temp < next_img[i+1][j-1] and temp < next_img[i+1][j] and temp < next_img[i+1][j+1]:
                        countermin= True
                    else:
                        countermin = False
                if countermax == True:        
                    if temp > next_img[i-1][j-1] and temp > next_img[i-1][j] and temp > next_img[i-1][j+1] and temp > next_img[i][j-1] and temp > next_img[i][j+1] and temp > next_img[i+1][j-1] and temp > next_img[i+1][j] and temp > next_img[i+1][j+1]:
                        countermax == True
                    else:
                        countermax = False
                        
            #plotting the point in the empty image
            if countermin == True or countermax == True:
                I = i*2**a
                J = j*2**a
                blank[I][J] = 255
    #return blank    

def DisplayPixels(blank, m, n):
    print("The 5 leftmost pixels are:")
    counter = 0
    pixellist = []
    for i in range(0,m):
        for j in range(0,16):
            if blank[i][j] == 255:                
                counter += 1
                pixellist.append((i,j))
                break
            
        if counter == 5:
            break
    print(pixellist)                
                       

print("Program Start") 
start = timeit.default_timer()
img = cv2.imread('task2.jpg',0)

m = img.shape[0]
n = img.shape[1]
blank = initialize(m,n)
dog_array = []    
octave = octave_creation()
Generating_Blur_Images(img, m, n, octave)
Generating_DoG(dog_array)
Generating_KeyPoints()
DisplayPixels(blank, m, n)
print(" Program End")
stop = timeit.default_timer()
print("Time Taken to Detect Features:", stop - start)