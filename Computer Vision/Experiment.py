import cv2
import numpy as np
import sys
import argparse
import imutils
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PIL import Image

import matplotlib.pyplot as plt

fname=''
processImg = cv2.imread('')
originImg = cv2.imread('')
checkImg = cv2.imread('')
originGrayImg = cv2.imread('')
grayImg = cv2.imread('')
function=''
Direction=''
parameter= 0.0

Tran=0.0
Scale=1.0
Rota=0.0

meanAll = 0
varAll = 0

from  UI import *  #导入之前新生成的窗口模块

class MyWindow(QMainWindow, Ui_ImageProcessor):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
# 通用
    def openImg(self):
        global fname
        global originImg
        global processImg
        global originGrayImg
        global grayImg
        global checkImg

        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*;;*.png;;All Files(*)")
        image1 = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), -1)
        image2 = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), 0)
        if image1 is None:
            print("未选择图片")
        else:
            if image1.shape[0]>521 or image1.shape[1] > 601:
                num = int(max(image1.shape[0]/521 , image1.shape[1] / 601))+1
                image1 = cv2.resize(image1, dsize=None, fx=num, fy=num, interpolation=cv2.INTER_AREA)

            img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            #img2 = cv2.cvtColor(image, cv2.COLOR_BGR2GRY)

            self.QtImg = QtGui.QImage(img.data,
                                      img.shape[1],
                                      img.shape[0],
                                      img.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label1.setScaledContents(True)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label2.setScaledContents(True)

            fname = imgName
            originImg = image1
            processImg = image1
            originGrayImg = image2
            grayImg = image2
            checkImg = originImg

# 通用
    def saveImg(self):
        global fname
        global processImg
        if originImg is None:  # ^^^^^^^^^^^
            print("未选择图片")
        else:
            # image = cv2.imread(fname)
            fileName = QFileDialog.getSaveFileName(filter="JPG(*.jpg);;PNG(*.png);;TIFF(*.tiff);;BMP(*.bmp)")[0]
            cv2.imwrite(fileName, processImg)
            print('Image saved as:', self.fileName)


    def originImg(self):
        global fname
        global processImg
        global originImg
        global originGrayImg
        global checkImg
        global Tran
        global Rota
        global Scale
        global function
        global parameter
        global direction
        global meanAll
        global varAll

        if originImg == '':
            print("未选择图片")
        else:
            function = ''
            direction = ''
            parameter = ''
            Tran = 0.0
            Rota = 0.0
            Scale = 1.0
            meanAll = 0
            varAll = 0

            self.le3.setText('')
            self.le4.setText('')
            self.le5.setText('')
            self.le6.setText('')
            self.le7.setText('')
            self.le8.setText('')
            self.le9.setText('')
            self.le10.setText('')
            self.le11.setText('')
            self.le12.setText('')
            self.le13.setText('')
            self.le14.setText('')
            self.le15.setText('')
            self.le16.setText('')

            originImg = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), -1)
            originGrayImg = cv2.imdecode(np.fromfile(fname, dtype=np.uint8), 0)
            processImg = originImg
            checkImg = originImg
            grayImg = originGrayImg

            img = cv2.cvtColor(originImg, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(img.data,
                                      img.shape[1],
                                      img.shape[0],
                                      img.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label1.setScaledContents(True)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label2.setScaledContents(True)
            self.label3.setPixmap(QPixmap(""))
            self.label4.setPixmap(QPixmap(""))
            self.label5.setText('')
            self.label6.setText('')
            self.label7.setText('')
            self.label8.setText('')
            self.label9.setPixmap(QPixmap(""))
            self.label10.setPixmap(QPixmap(""))
########################################################################################################################
# 实验一 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def selectFunc(self):
        global function
        global parameter
        global direction
        # 设置弹出框下拉列表内容（平移、旋转、放缩）
        items = ('Translation','Rotation','Scale')
        # 弹框
        item,ok = QInputDialog.getItem(self,"选择功能","选择功能",items,0,False)
        # 存储选择信息至全局变量，并将信息输出到LineText中可视化显示
        if ok and item:
            self.le3.setText(item)
            function = str(item)
            direction = ''
            parameter = ''
            self.le4.setText(direction)
            self.le5.setText(parameter)
# 实验一
    def selectDire(self):
        global function
        global parameter
        global direction
        if function == '':
            print("未选择功能")
        else:
            # 判断已选择的功能，根据不同的功能设置弹出框中不同的下拉列表选项
            if function == 'Translation':
                items = ("Up", "Down", "Left", 'Right')
            if function == 'Rotation':
                items = ("Clockwise", "Anticlockwise")
            if function == 'Scale':
                items = ("ZoomIn", "ZoomOut")
            # 弹框
            item, ok = QInputDialog.getItem(self, "选择方向", "选择方向", items, 0, False)
            # 存储选择信息至全局变量，并将信息输出到LineText中可视化显示
            if ok and item:
                self.le4.setText(item)
                direction = str(item)

# 实验一
    def selectPara(self):
        global function
        global parameter
        global direction
        if function == '':
            print("未选择功能")
        else:
            # 弹框
            num,ok = QInputDialog.getText(self,"输入参数",'输入参数')
            # 存储选择信息至全局变量，并将信息输出到LineText中可视化显示
            if ok:
                self.le5.setText(str(num))
                # 参数值（浮点型）在各功能中通用。平移中作为像素值，旋转中作为旋转角度，放缩中作为放缩倍数。
                parameter = float(num)

# 实验一
    def processImg(self):
        global function
        global parameter
        global direction
        global Tran
        global Rota
        global Scale
        global fname
        global processImg

        if originImg is None:  # ^^^^^^^^^^^
            print("未选择图片")
        else:
            # 旋转
            if function == 'Rotation':
                # 判断方向是否为顺时针，根据方向转换设置参数的正负符号，用以函数的输入
                if direction == 'Clockwise':
                    Rota = -abs(parameter)
                else:
                    Rota = abs(parameter)
                # 核心部分
                h, w = processImg.shape[:2]
                # 旋转  通过getRotationMatrix2D得到图像旋转后的矩阵（其中Rota参数：正数为逆时针、负数为顺时针）
                M = cv2.getRotationMatrix2D((w / 2, h / 2), Rota, 1)
                # 通过仿射变换函数warpAffine将矩阵转化为图像
                image1 = cv2.warpAffine(processImg, M, (w, h))
                processImg = image1
                img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                self.QtImg = QtGui.QImage(img.data,
                                          img.shape[1],
                                          img.shape[0],
                                          img.shape[1] * 3,
                                          QtGui.QImage.Format_RGB888)
                self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.label2.setScaledContents(True)

            # 平移
            if function == 'Translation':
                if direction == 'Left' or direction == 'Up':
                    Tran = -abs(parameter)
                else:
                    Tran = abs(parameter)
                if direction == 'Left' or direction == 'Right':

                    M = np.array([[1, 0, Tran], [0, 1, 0]], dtype=np.float32)
                    image1 = cv2.warpAffine(processImg, M, (processImg.shape[1], processImg.shape[0])) # 原本processImg是image
                    processImg = image1  # 原来没有这行
                    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

                    self.QtImg = QtGui.QImage(img.data,
                                              img.shape[1],
                                              img.shape[0],
                                              img.shape[1] * 3,
                                              QtGui.QImage.Format_RGB888)
                    self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.label2.setScaledContents(True)

                if direction == 'Up' or direction == 'Down':

                    M = np.array([[1, 0, 0], [0, 1, Tran]], dtype=np.float32)
                    image1 = cv2.warpAffine(processImg, M, (processImg.shape[1], processImg.shape[0])) # ^^^^^^^^^^^^^^^^^^
                    processImg = image1  # 原来没有这行
                    img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

                    self.QtImg = QtGui.QImage(img.data,
                                              img.shape[1],
                                              img.shape[0],
                                              img.shape[1] * 3,
                                              QtGui.QImage.Format_RGB888)
                    self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                    self.label2.setScaledContents(True)


            # 放缩
            if function == 'Scale':
                if direction == 'ZoomIn':
                    Scale = 1.0/parameter
                else:
                    Scale = parameter
                h, w = processImg.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, Scale)  # 旋转  通过getRotationMatrix2D得到图像旋转后的矩阵
                image1 = cv2.warpAffine(processImg, M, (w, h))  # 通过仿射变换函数warpAffine将矩阵转化为图像             #^^^^^^^^^^^
                processImg = image1  # ^^^^^^^^^^^
                img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
                self.QtImg = QtGui.QImage(img.data,
                                          img.shape[1],
                                          img.shape[0],
                                          img.shape[1] * 3,
                                          QtGui.QImage.Format_RGB888)
                self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.label2.setScaledContents(True)

# 实验一 点击选像素
    def checkPixel(self):
        global originImg
        global checkImg
        if originImg is None:  # ^^^^^^^^^^^^^^^^^
            print("未选择图片")
        else:
            # 打开图像窗口
            cv2.namedWindow("img")
            cv2.setMouseCallback("img", self.mouse_click)
            cv2.imshow('img', checkImg)
            cv2.waitKey(0)
# 实验一
    def mouse_click(self,event, x, y, flags, para):
        global originImg
        global checkImg
        # 判断左边鼠标点击事件
        if event == cv2.EVENT_LBUTTONDOWN:
            print('PIX:', x, y)
            print("BGR:", checkImg[y, x])
            #print("GRAY:", gray[y, x])
            #print("HSV:", hsv[y, x])

            # 将像素坐标和对应RGB值在LineText中可视化显示
            self.le6.setText('['+str(x)+','+str(y)+']'+str(checkImg[y, x]))
# 实验一 移动选像素
    def mouseTrack(self):
        self.setMouseTracking(True)
        self.mouseMoveEvent
# 实验一
    def mouseMoveEvent(self, event):
        global originImg
        global processImg
        self.setMouseTracking(True)
        s = event.pos()
        if (s.x()>30 and s.x()<631 and s.y()>10 and s.y()<531):
            x = s.x() - 30
            y = s.y() - 10
            print(x,y)
            self.le6.setText('[' + str(x) + ',' + str(y) + ']'+str(originImg[y, x]))
        if (s.x() > 650 and s.x() < 1251 and s.y() > 10 and s.y() < 531):
            x = s.x() - 650
            y = s.y() - 10
            print(x,y)
            self.le6.setText('[' + str(x) + ',' + str(y) + ']' + str(processImg[y, x]))
        else:
            print('不在范围内')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 实验二 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def showAllHist(self):
        global fname
        global originImg
        global processImg
        global originGrayImg
        global grayImg
        global meanAll
        global varAll

        if originImg is None:
            print("未选择图片")
        else:
            processImg = originImg
            image1 = cv2.cvtColor(originGrayImg, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label2.setScaledContents(True)
            # 统计全局直方图
            All_Hist = cv2.calcHist([originGrayImg], [0], None, [256], [0, 256])

            plt.plot(All_Hist/(originGrayImg.shape[0]*originGrayImg.shape[1]))
            plt.savefig('1_All_Hist.jpg')
            plt.clf()
            # 直方图可视化
            img = cv2.imread('1_All_Hist.jpg')
            image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label3.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label3.setScaledContents(True)
            # 计算方差均值
            meanAll = np.mean(originGrayImg)
            varAll = np.var(originGrayImg)
            # 数值结果可视化
            self.label5.setText(str(meanAll))
            self.label6.setText(str(varAll))

    def equalizeHist(self):
        global fname
        global originImg
        global processImg
        global originGrayImg
        global grayImg

        if originImg is None:
            print("未选择图片")
        else:
            # 图像全局均衡化
            All_Equal_Image = cv2.equalizeHist(originGrayImg)
            processImg = All_Equal_Image
            # 处理后图像可视化
            image1 = cv2.cvtColor(All_Equal_Image, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label2.setScaledContents(True)
            # 计算均衡化后图像的方差均值
            meanAll = np.mean(All_Equal_Image)
            varAll = np.var(All_Equal_Image)
            self.label5.setText(str(meanAll))
            self.label6.setText(str(varAll))
            # 统计均衡化后的直方图
            All_Equal_Hist = cv2.calcHist([All_Equal_Image], [0], None, [256], [0, 256])
            plt.plot(All_Equal_Hist/(All_Equal_Image.shape[0]*All_Equal_Image.shape[1]))
            plt.savefig('1_All_Equal_Hist.jpg')
            plt.clf()
            # 直方图可视化
            img = cv2.imread('1_All_Equal_Hist.jpg')
            image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label3.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label3.setScaledContents(True)

    def showPartHist(self):
        global fname
        global originImg
        global processImg
        global originGrayImg
        global grayImg

        x = self.le7.text()
        y = self.le8.text()
        s = self.le9.text()
        if x == '' or y == '' or s == '':
            print('参数不全')
            print(originGrayImg.shape[:2])
        else:
            x = int(x)
            y = int(y)
            s = int(s)
            border = (s-1)/2
            originX = x - border
            originY = y - border
            endX = x + border
            endY = y + border
            print(originX)
            # 判定所选窗口的范围
            if originX < 0 or originY < 0 or endX > originGrayImg.shape[0] or endY > originGrayImg.shape[1]:
                print('超出范围')
            else:
                # 设置掩膜，并在原图上只显示对应窗口
                mask = np.zeros(originGrayImg.shape[:2], np.uint8)
                print(mask)
                mask[int(originX):int(endX), int(originY):int(endY)] = 255
                Part_Image = cv2.bitwise_and(originGrayImg, originGrayImg, mask=mask)
                # 计算选取窗口的图像灰度方差均值
                meanPart = np.mean(Part_Image)
                varPart = np.var(Part_Image)
                # 数值结果可视化
                self.label7.setText(str(meanPart))
                self.label8.setText(str(varPart))
                # 处理后图像可视化
                image1 = cv2.cvtColor(Part_Image, cv2.COLOR_BGR2RGB)

                self.QtImg = QtGui.QImage(image1.data,
                                          image1.shape[1],
                                          image1.shape[0],
                                          image1.shape[1] * 3,
                                          QtGui.QImage.Format_RGB888)
                self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.label2.setScaledContents(True)
                # 统计局部直方图（添加mask参数）
                Part_Hist = cv2.calcHist([originGrayImg], [0], mask, [256], [0, 256])
                plt.plot(Part_Hist/(originGrayImg.shape[0]*originGrayImg.shape[1]))
                plt.savefig('1_Part_Hist.jpg')
                plt.clf()
                # 直方图可视化
                img = cv2.imread('1_Part_Hist.jpg')
                image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.QtImg = QtGui.QImage(image1.data,
                                          image1.shape[1],
                                          image1.shape[0],
                                          image1.shape[1] * 3,
                                          QtGui.QImage.Format_RGB888)
                self.label4.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
                self.label4.setScaledContents(True)

    def equalizePartHist(self):
        global fname
        global originImg
        global processImg
        global originGrayImg
        global grayImg
        global meanAll
        global varAll
        # 读取参数
        E = self.le10.text()
        Sxy = self.le11.text()
        K0 = self.le12.text()
        K1 = self.le13.text()
        K2 = self.le14.text()
        # 判断参数设置是否完整
        if E == '' or Sxy == '' or K0 == '' or K1 == '' or K2 == '' or meanAll == 0 or varAll == 0:
            print('参数不全')
        else:
            E = float(E)
            Sxy = float(Sxy)
            K0 = float(K0)
            K1 = float(K1)
            K2 = float(K2)
            border = int((Sxy - 1)/2)
            pixelList = np.array([])
            # 为原图像添加边框（padding），边框灰度值设为0
            # img = cv2.copyMakeBorder(originGrayImg, int(border), int(border), int(border), int(border),
            #                          cv2.BORDER_CONSTANT, value=[0])
            # # 遍历像素点，计算每一像素点邻域窗口内的均值和方差
            # for i in range(1, img.shape[1] - 1):
            #     for j in range(1, img.shape[0] - 1):
            #         for m in range(-border, border + 1):
            #             for n in range(-border, border + 1):
            #                 x = i + m
            #                 y = j + n
            #                 if x >= 0 and x <= originGrayImg.shape[1] - 1 and y >= 0 and y <= originGrayImg.shape[
            #                     0] - 1 and
            #                     pixelList = np.append(pixelList, img[i + x][j + y])
            #         meanPart = np.mean(pixelList)
            #         varPart = np.var(pixelList)
            #         # 判断是否满足增强条件，若满足，则增强、反之则保持不变
            #         if meanPart <= K0 * meanAll and K1 * varAll <= varPart and varPart <= K2 * varAll:
            #             img[i][j] = E * img[i][j]
            #         grayImg[i - 1][j - 1] = img[i][j]
            #         pixelList = np.array([])
            
            # 遍历像素点，计算每一像素点邻域窗口内的均值和方差
            for i in range(originGrayImg.shape[0] ):
                for j in range(originGrayImg.shape[1] ):
                    for m in range(-border, border + 1):
                        for n in range(-border, border + 1):
                            x = i+m
                            y = j+n
                            if x >= 0 and x <= originGrayImg.shape[0] - 1 and y >= 0 and y <= originGrayImg.shape[1] - 1:
                                pixelList = np.append(pixelList,originGrayImg[x][y])
                    meanPart = np.mean(pixelList)
                    varPart = np.var(pixelList)
                    # 判断是否满足增强条件，若满足，则增强、反之则保持不变
                    if meanPart <= K0 * meanAll and K1 * varAll <= varPart and varPart <= K2 * varAll:
                        processImg[i][j] = E * processImg[i][j]

                    pixelList = np.array([])

            cv2.imwrite('test.jpg',grayImg)
            # 将局部增强处理后的图像结果可视化
            image1 = cv2.cvtColor(processImg, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label2.setScaledContents(True)
            # 计算局部增强处理后图像的方差均值
            meanPart = np.mean(processImg)
            varPart = np.var(processImg)
            # 数值结果可视化
            self.label5.setText(str(meanPart))
            self.label6.setText(str(varPart))
            # 统计处理后图像的直方图
            Part_Hist = cv2.calcHist([processImg], [0], None, [256], [0, 255])
            plt.plot(Part_Hist/(processImg.shape[0]*processImg.shape[1]))
            plt.savefig('1_Part_Equal_Hist.jpg')
            plt.clf()
            # 直方图可视化
            img = cv2.imread('1_Part_Equal_Hist.jpg')
            image1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(image1.data,
                                      image1.shape[1],
                                      image1.shape[0],
                                      image1.shape[1] * 3,
                                      QtGui.QImage.Format_RGB888)
            self.label3.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label3.setScaledContents(True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 实验三 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    def checkPixel_spectrum(self):
        global originImg
        global processImg
        global checkImg

        Height_original, Width_original = np.shape(originImg)  # 获取宽高信息
        #   DFT

        dft = np.fft.fft2(originImg)
        dft_shift = np.fft.fftshift(dft)
        # 频谱图像双通道复数转换为0-255区间
        magitude_spectrum = 9 * np.log(np.abs(dft_shift))
        magitude_spectrum_int = np.zeros((Height_original, Width_original), dtype='uint8')
        for x in range(Height_original):
            for y in range(Width_original):
                index = np.int(magitude_spectrum[x][y])
                magitude_spectrum_int[x][y] = index
                pass
            pass
        print(np.shape(magitude_spectrum_int))
        img = magitude_spectrum_int

        checkImg = img
        self.checkPixel()
        checkImg = originImg

    def notch_filter(self):
        global fname
        global originImg
        global processImg
        global checkImg
        # get image information

        flag = self.le17.text()

        Height_original, Width_original = np.shape(originImg)  # 获取宽高信息

        #   DFT
        img_float32 = np.float32(originImg)
        # dft = cv2.dft(img_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft = np.fft.fft2(originImg)
        dft_shift = np.fft.fftshift(dft)

        # 频谱图像双通道复数转换为0-255区间
        # magitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
        magitude_spectrum = 8 * np.log(np.abs(dft_shift)+0.0000000001)
        magitude_spectrum_int = np.zeros((Height_original, Width_original), dtype='uint8')
        for x in range(Height_original):
            for y in range(Width_original):
                index = np.int(magitude_spectrum[x][y])
                # if index > 128:
                #             #     index = 128
                #             # elif index < 0:
                #             #     index = 0
                #             #     pass
                magitude_spectrum_int[x][y] = index
                pass
            pass

        # # 标记图片亮点坐标
        # def mouse(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         xy = "%d,%d" % (x, y)
        #         cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=-1)
        #         cv2.putText(img, xy, (x - 25, y - 25), cv2.FONT_HERSHEY_PLAIN,
        #                     1.0, (255, 255, 255), thickness=1)
        #         cv2.imshow("image", img)


        # cv2.namedWindow("image")
        # cv2.imshow("image", img)
        # # cv2.resizeWindow("image", 800, 600)
        # cv2.setMouseCallback("image", self.mouse_click())
        # #  显示频谱图
        # cv2.waitKey(0)
        # # cv2.destroyAllWindows()

        '''
        亮点坐标
        亮点坐标（Y，X）
        (54, 42)   |    (112, 41)
        (54, 84)   |    (112, 81)
        (56, 165)   |    (114, 162) 
        (56, 205)   |    (114, 204)
    
        '''

        # 四个高通滤波器中心点与图像中心的相对值
        p1 = (81, 30)
        p2 = (39, 30)
        p3 = (42, -28)
        p4 = (82, -28)

        centerX = np.floor(Height_original / 2)  # 123.0 246
        centerY = np.floor(Width_original / 2)  # 84.0 168
        print(centerX)
        # 构建滤波器传递函数H_NR

        H_NR = np.ones((Height_original, Width_original))
        Dk = np.zeros((2, 4))
        #   parameter
        N = self.le15.text()
        D0k = self.le16.text()

        if flag == '1':
            if N == '' or D0k == '':
                print('参数不全')
            else:
                N = int(N)
                D0k = float(D0k)
                for u in range(Height_original):
                    for v in range(Width_original):
                        Dk[0][0] = np.sqrt((u - centerX - p1[0]) ** 2 + (v - centerY - p1[1]) ** 2)
                        Dk[0][1] = np.sqrt((u - centerX - p2[0]) ** 2 + (v - centerY - p2[1]) ** 2)
                        Dk[0][2] = np.sqrt((u - centerX - p3[0]) ** 2 + (v - centerY - p3[1]) ** 2)
                        Dk[0][3] = np.sqrt((u - centerX - p4[0]) ** 2 + (v - centerY - p4[1]) ** 2)

                        Dk[1][0] = np.sqrt((u - centerX + p1[0]) ** 2 + (v - centerY + p1[1]) ** 2)
                        Dk[1][1] = np.sqrt((u - centerX + p2[0]) ** 2 + (v - centerY + p2[1]) ** 2)
                        Dk[1][2] = np.sqrt((u - centerX + p3[0]) ** 2 + (v - centerY + p3[1]) ** 2)
                        Dk[1][3] = np.sqrt((u - centerX + p4[0]) ** 2 + (v - centerY + p4[1]) ** 2)

                        for i in Dk[0]:
                            H_NR[u][v] = H_NR[u][v] * (1 / (1 + (D0k / i + 0.0001) ** (2 * N)) + 0.0001)
                            pass
                        for i in Dk[1]:
                            H_NR[u][v] = H_NR[u][v] * (1 / (1 + (D0k / i + 0.0001) ** (2 * N)) + 0.0001)
                            pass
                        pass
                    pass

        elif flag == '2':
            for i in range(Height_original):
                for j in range(int(Width_original / 2), int(Width_original / 2) + 2):
                    H_NR[i][j] = 0
            for i in range(int(Height_original / 2), int(Height_original / 2) + 10):
                for j in range(int(Width_original / 2), int(Width_original / 2) + 2):
                    H_NR[i][j] = 1
        else:
            print('请选择模式')
        # IDFT
        # 滤波器图（H_NR）
        # 滤波后频谱图Gs
        Gs = magitude_spectrum * H_NR
        # 频谱对比图
        plt.figure(1)
        plt.title('The spectrum before filtering')
        plt.imshow(magitude_spectrum, cmap='gray')
        plt.savefig('2_spectrum_before.jpg')
        plt.clf()

        plt.figure(2)
        plt.title('The spectrum after filtering')
        plt.imshow(Gs, cmap='gray')
        plt.savefig('2_spectrum_after.jpg')
        plt.clf()

        # 逆变换回图像
        G = dft_shift * H_NR
        iimg = np.fft.ifft2(G)
        iimg = np.abs(iimg)

        # 图像对比图
        plt.figure(3)
        plt.subplot(121)
        plt.title('Before the filtering')
        plt.imshow(img_float32, cmap='gray')
        plt.subplot(122)
        plt.title('After the filtering')
        plt.imshow(iimg, cmap='gray')
        plt.savefig('2_image_after.jpg')
        plt.clf()

        img1 = cv2.imread('2_image_after.jpg')
        image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        processImg = image1
        self.QtImg = QtGui.QImage(image1.data,
                                  image1.shape[1],
                                  image1.shape[0],
                                  image1.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.label2.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label2.setScaledContents(True)

        img2 = cv2.imread('2_spectrum_before.jpg')
        image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(image2.data,
                                  image2.shape[1],
                                  image2.shape[0],
                                  image2.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.label9.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label9.setScaledContents(True)

        img3 = cv2.imread('2_spectrum_after.jpg')
        image3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
        self.QtImg = QtGui.QImage(image3.data,
                                  image3.shape[1],
                                  image3.shape[0],
                                  image3.shape[1] * 3,
                                  QtGui.QImage.Format_RGB888)
        self.label10.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
        self.label10.setScaledContents(True)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())