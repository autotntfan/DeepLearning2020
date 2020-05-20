#這是用來圖片前處理的檔案
import numpy as np
from PIL import Image
import scipy.misc
import scipy.io
import matplotlib.pyplot as plt
import cv2
from sklearn.preprocessing import PolynomialFeatures,binarize
from sklearn.linear_model import LinearRegression
import os
#這是用來處理train.mat、test.mat其他.mat資料集的x_train
#搭配使用DLmidterm.py
#num:第幾張圖片,datatype:train or test
def Deal_With_Img(num,datatype):
    data=scipy.io.loadmat('D:/DLmidterm/'+datatype+".mat") #讀取資料
    x,y,y_onehot=data["x"],data["y"],data["y_onehot"]      #讀取資料
    data=x[0][num]                           #處理第num張圖片
    path="D:/DLmidterm/"+datatype+"dealed/"  #第一次處理完的圖片儲存位置
    dic = [[0] * 2 for i in range(100)]      #計算回歸時參考的高度
    for i in range(100):
        dic[i][0]=25
        dic[i][1]=25
    dic[50][0]=26
    dic[50][1]=24
    dic[48][0]=23
    dic[48][1]=30
    dic[46][0]=27
    dic[46][1]=25
    dic[45][0]=21
    dic[45][1]=30

    plt.rcParams.update({'figure.max_open_warning': 0}) #fix the memory error
    img = cv2.fastNlMeansDenoisingColored(data, None, 31, 31 ,7 ,21) #去除雜訊
    height1, width1, channels1 = img.shape 
    plt.figure(figsize=(width1, height1), dpi=100)
    plt.axis('off')
    plt.imshow(img)
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)#去白邊
    plt.savefig(path+str(num)+'.jpg',dpi=10)      #將圖放大10倍再第二次處理
    img2 = np.array(Image.open(path+str(num)+'.jpg'))   #讀取圖片
    plt.close()


    ret,thresh = cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)#黑白化
    height, width, channels = thresh.shape
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgarr[:,100:width-40] = 0
    imagedata = np.where(imgarr == 255) #找到白色的地方
    X = np.array([imagedata[1]])
    Y = height - imagedata[0]

    poly_reg= PolynomialFeatures(degree = 2) #二次多項式
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()       #回歸線方式
    regr.fit(X_, Y)
    ##以下用回歸處理線條
    X2 = np.array([[g for g in range(0,width)]])
    X2_ = poly_reg.fit_transform(X2.T)

    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = height - int(ele[0])
        thresh[pos-int(dic[height1][0]):pos+int(dic[height1][1]), int(ele[1])] = 255 - thresh[pos-int(dic[height1][0]):pos+int(dic[height1][1]),int(ele[1])] #這裡可以更改回歸線條上下範圍

    newdst=np.array(Image.fromarray(thresh).resize((140,48))) #resize (h,w)
    print("now is dealing with:",num)
    return newdst

#這是用來處理爬蟲爬下來的圖片前處理
def Deal_With_TestImg(num,data,datatype):
    path="D:/DLmidterm/"+datatype+"dealed/"
    dic = [[0] * 2 for i in range(100)]
    for i in range(100):
        dic[i][0]=25
        dic[i][1]=25
    dic[50][0]=26
    dic[50][1]=24
    dic[48][0]=23
    dic[48][1]=30
    dic[46][0]=27
    dic[46][1]=25
    dic[45][0]=21
    dic[45][1]=30

    plt.rcParams.update({'figure.max_open_warning': 0}) #fix the memory error
    img = cv2.fastNlMeansDenoisingColored(data, None, 31, 31 ,7 ,21)
    height1, width1, channels1 = img.shape #get img height and width
    plt.figure(figsize=(width1, height1), dpi=100)
    plt.axis('off')
    plt.imshow(img)
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)#去白邊
    plt.savefig(path+str(num)+'.jpg',dpi=10)
    img2 = np.array(Image.open(path+str(num)+'.jpg'))
    plt.close()


    ret,thresh = cv2.threshold(img2,127,255,cv2.THRESH_BINARY_INV)#黑白化
#     plt.imshow(thresh)
#     plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)#去白邊
    #get img height and width
    height, width, channels = thresh.shape
    imgarr = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    imgarr[:,100:width-40] = 0
    imagedata = np.where(imgarr == 255) #find where are white
    X = np.array([imagedata[1]])
    Y = height - imagedata[0]

    poly_reg= PolynomialFeatures(degree = 2)
    X_ = poly_reg.fit_transform(X.T)
    regr = LinearRegression()
    regr.fit(X_, Y)

    X2 = np.array([[g for g in range(0,width)]])
    X2_ = poly_reg.fit_transform(X2.T)

    for ele in np.column_stack([regr.predict(X2_).round(0),X2[0],] ):
        pos = height - int(ele[0])
        thresh[pos-int(dic[height1][0]):pos+int(dic[height1][1]), int(ele[1])] = 255 - thresh[pos-int(dic[height1][0]):pos+int(dic[height1][1]),int(ele[1])] #這裡可以更改回歸線條上下範圍

    newdst=np.array(Image.fromarray(thresh).resize((140,48))) #resize (h,w)
    print("now is dealing with:",num)

    return newdst