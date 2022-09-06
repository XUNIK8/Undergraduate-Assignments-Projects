import numpy as np
import cv2
import sys
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from  UI6 import *  #导入之前新生成的窗口模块

originImg = cv2.imread('', 0)
processImgSeed = cv2.imread('')
processImgSeedEr = cv2.imread('')
originSeeds = []


class MyWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)


    def openImg(self):
        global originImg
        global processImgSeed
        global processImgSeedEr
        global originSeeds
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*;;*.png;;All Files(*)")
        image1 = cv2.imdecode(np.fromfile(imgName, dtype=np.uint8), -1)
        if image1 is None:
            print("未选择图片")
        else:
            originImg = image1
            fname = imgName
            img = cv2.resize(image1, (271,271))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.QtImg = QtGui.QImage(img.data,
                                      img.shape[1],
                                      img.shape[0],
                                      img.shape[1]*3,
                                      QtGui.QImage.Format_RGB888)
            self.label1.setPixmap(QtGui.QPixmap.fromImage(self.QtImg))
            self.label1.setScaledContents(True)

            # 原图直方图
            plt.hist(image1.ravel(), 256, (0, 255), density=True)
            plt.savefig('6.2_Hist.jpg')
            plt.clf()
            # 直方图可视化
            image1 = cv2.imread('6.2_Hist.jpg')
            img = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(img.data,
                                 img.shape[1],
                                 img.shape[0],
                                 QtGui.QImage.Format_RGB888)
            self.label2.setPixmap(QPixmap.fromImage(QtImg))
            self.label2.setScaledContents(True)


    def findSeed(self):
        global originImg
        global processImgSeed
        global processImgSeedEr
        global originSeeds
        global fname
        seedList = []

        # 种子阈值处理
        ret, thresh_A = cv2.threshold(originImg, 254, 255, cv2.THRESH_BINARY)  # 灰度图转换成二值图像
        # 保存种子图像为processImgSeed
        cv2.imwrite('6.3_seedBefore.jpg', thresh_A)
        processImgSeed = thresh_A
        img = cv2.cvtColor(thresh_A, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img.data,
                             img.shape[1],
                             img.shape[0],
                             img.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label3.setPixmap(QPixmap.fromImage(QtImg))
        self.label3.setScaledContents(True)

        thresh_A_copy = thresh_A.copy()  # 复制thresh_A到thresh_A_copy
        thresh_B = np.zeros(originImg.shape, np.uint8)  # thresh_B大小与A相同，像素值为0
        th2 = np.zeros(originImg.shape, np.uint8) # 用来存储最终种子图像

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # 3×3结构元

        count = []  # 为了记录连通分量中的像素个数
        num = 0
        # 循环，直到thresh_A_copy中的像素值全部为0
        while thresh_A_copy.any():
            num+=1
            Xa_copy, Ya_copy = np.where(thresh_A_copy > 0)  # thresh_A_copy中值为255的像素的坐标
            thresh_B[Xa_copy[0]][Ya_copy[0]] = 255  # 选取第一个点，并将thresh_B中对应像素值改为255

            # 连通分量算法，先对thresh_B进行膨胀，再和thresh_A执行and操作（取交集）
            for i in range(200):
                dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
                thresh_B = cv2.bitwise_and(thresh_A, dilation_B)

            # 取thresh_B值为255的像素坐标，并将thresh_A_copy中对应坐标像素值变为0
            Xb, Yb = np.where(thresh_B > 0)
            thresh_A_copy[Xb, Yb] = 0

            # 显示连通分量及其包含像素数量
            count.append(len(Xb))
            if len(count) == 0:
                print("无连通分量")
            if len(count) == 1:
                print("第1个连通分量为{}".format(count[0]))
            if len(count) >= 2:
                print("第{}个连通分量为{}".format(len(count), count[-1]))
            # 连通量内各取一点作为种子
            flag = False
            h,w = np.shape(thresh_B)
            for i in range(h):
                for j in range(w):
                    if thresh_B[i][j] > 0:
                        seedList.append([i, j])
                        th2[i][j] = 255
                        flag = True
                        break
                if flag == True:
                    break

            thresh_B = np.zeros(originImg.shape, np.uint8)
        # 获得种子list
        originSeeds = seedList
        print(originSeeds)
        print(np.shape(originSeeds))
        # 保存腐蚀种子图像为processImgSeedEr
        cv2.imwrite('6.4_seedAfter.jpg', th2)
        processImgSeedEr = th2

        img = cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img.data,
                             img.shape[1],
                             img.shape[0],
                             img.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label4.setPixmap(QPixmap.fromImage(QtImg))
        self.label4.setScaledContents(True)


    def OtsuThreshold(self):
        global originImg
        global processImgSeed
        global processImgSeedEr
        global originSeeds
        global fname

        # 差值图像
        img1 =  abs(originImg - processImgSeed)
        img = abs(255 - originImg)

        cv2.imwrite('6.5_DvalueImg.jpg',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img.data,
                             img.shape[1],
                             img.shape[0],
                             img.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label5.setPixmap(QPixmap.fromImage(QtImg))
        self.label5.setScaledContents(True)

        # 差值直方图
        plt.hist(img.ravel(), 256, (0, 255), density=True)
        plt.savefig('6.6_Hist.jpg')
        plt.clf()
        # 直方图可视化
        image1 = cv2.imread('6.6_Hist.jpg')
        img0 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img0.data,
                             img0.shape[1],
                             img0.shape[0],
                             img0.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label6.setPixmap(QPixmap.fromImage(QtImg))
        self.label6.setScaledContents(True)


        # 差值图像双阈值处理
        imgpath = "6.5_DvalueImg.jpg"
        src = cv2.imread(imgpath, flags=0)
        th2 = self.Otsu2Threshold(src)
        cv2.imwrite('6.8_twoThreshold.jpg', th2)
        img2 = cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)
        QtImg = QtGui.QImage(img2.data,
                             img2.shape[1],
                             img2.shape[0],
                             img2.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label7.setPixmap(QPixmap.fromImage(QtImg))
        self.label7.setScaledContents(True)

        # 差值图像单阈值处理
        ret1, th1 = cv2.threshold(img, 68, 255, cv2.THRESH_BINARY)
        cv2.imwrite('6.8_oneThreshold.jpg',th1)
        img1 = cv2.cvtColor(th1, cv2.COLOR_BGR2RGB)
        print('单阈值Threshold：', ret1)
        QtImg = QtGui.QImage(img1.data,
                             img1.shape[1],
                             img1.shape[0],
                             img1.shape[1] * 3,
                             QtGui.QImage.Format_RGB888)
        self.label8.setPixmap(QPixmap.fromImage(QtImg))
        self.label8.setScaledContents(True)

    # 双阈值处理函数
    def Otsu2Threshold(self, src):
        global originImg
        global processImgSeed
        global processImgSeedEr
        global originSeeds
        global fname

        Threshold1 = 0
        Threshold2 = 0

        width, height = np.shape(src)
        hest = np.zeros([256], dtype=np.int32)
        for row in range(width):
            for col in range(height):
                pv = src[row, col]
                hest[pv] += 1
        tempg = -1
        N_blackground = 0
        N_object = 0
        N_all = width * height
        for i in range(256):
            N_object += hest[i]
            for k in range(i, 256, 1):
                N_blackground += hest[k]
            for j in range(i, 256, 1):
                gSum_object = 0
                gSum_middle = 0
                gSum_blackground = 0
                theta = 0
                N_middle = N_all - N_object - N_blackground
                w0 = N_object / N_all
                w2 = N_blackground / N_all
                w1 = 1 - w0 - w2
                for k in range(i):
                    gSum_object += k * hest[k]
                u0 = gSum_object / N_object
                for k in range(i + 1, j, 1):
                    gSum_middle += k * hest[k]
                u1 = gSum_middle / (N_middle + theta)

                for k in range(j + 1, 256, 1):
                    gSum_blackground += k * hest[k]
                u2 = gSum_blackground / (N_blackground + theta)

                u = w0 * u0 + w1 * u1 + w2 * u2
                print(u)
                g = w0 * (u - u0) * (u - u0) + w1 * (u - u1) * (u - u1) + w2 * (u - u2) * (u - u2)
                if tempg < g:
                    tempg = g
                    Threshold1 = i
                    Threshold2 = j
                N_blackground -= hest[j]

        h, w = np.shape(src)
        img = np.zeros([h, w], np.uint8)
        print('双阈值Threshold1:', Threshold1)
        print('双阈值Threshold2:', Threshold2)
        for row in range(h):
            for col in range(w):
                if src[row, col] > Threshold2:
                    img[row, col] = 255
                elif src[row, col] <= Threshold1:
                    img[row, col] = 0
                else:
                    img[row, col] = 126
        BlackgroundNum = 0
        AllNum = width * height
        for i in range(width):
            for j in range(height):
                if img[i, j] == 0:
                    BlackgroundNum += 1
        BlackgroundRatio = BlackgroundNum / AllNum
        if BlackgroundRatio < 0.4:  # 背景占比过少时，做一个反向操作
            w, h = np.shape(src)
            for i in range(w):
                for j in range(h):
                    img[i, j] = 255 - img[i, j]
        return img

    def regionGrow(self):
        global originImg
        global processImgSeed
        global processImgSeedEr
        global originSeeds
        global fname

        thresh = self.le1.text()
        if thresh == '':
            print('参数不全')
        else:
            img = originImg
            thresh = int(self.le1.text())
            img = self.regionGrowProcess(img,originSeeds,thresh,1)
            cv2.imwrite('6.9_regionGrow.jpg',img)
            img = cv2.imread('6.9_regionGrow.jpg',0)
            img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            QtImg = QtGui.QImage(img1.data,
                                 img1.shape[1],
                                 img1.shape[0],
                                 img1.shape[1] * 3,
                                 QtGui.QImage.Format_RGB888)
            self.label9.setPixmap(QPixmap.fromImage(QtImg))
            self.label9.setScaledContents(True)

    # 区域生长部分
    def getGrayDiff(self,img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint[0]][ currentPoint[1]]) - int(img[tmpPoint[0]][tmpPoint[1]]))

    def selectConnects(self,p):
        if p != 0:
            connects = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], \
                        [0, 1], [-1, 1], [-1, 0]]
        else:
            connects = [[0, -1], [1, 0], [0, 1], [-1, 0]]
        return connects

    # 区域生长函数
    def regionGrowProcess(self,img, seeds, thresh, p=1):

        height, width = img.shape

        seedMark = np.zeros(img.shape)
        seedList = seeds #[(228, 427), (229, 424), (232, 436), (233, 500), (236, 366), (236, 531), (239, 378), (239, 452), (239, 465), (241, 248), (244, 203), (245, 291), (248, 131), (248, 290), (250, 121), (255, 248), (262, 165), (273, 93), (292, 72)]

        label = 255
        num = 0
        connects = self.selectConnects(p)
        while (len(seedList) > 0):
            num += 1
            currentPoint = seedList.pop(0)
            seedMark[currentPoint[0], currentPoint[1]] = label
            for i in range(8):
                tmpX = currentPoint[0] + connects[i][0]
                tmpY = currentPoint[1] + connects[i][1]
                if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                    continue
                tmpPoint = [tmpX,tmpY]
                grayDiff = self.getGrayDiff(img, currentPoint, tmpPoint)
                if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                    num +=1
                    seedMark[tmpX, tmpY] = 255
                    img[tmpX,tmpY] = 255
                    seedList.append([tmpX, tmpY])

        return seedMark

if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = MyWindow()
    MainWindow.show()
    sys.exit(app.exec_())
