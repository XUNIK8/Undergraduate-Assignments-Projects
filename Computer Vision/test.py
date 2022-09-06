import numpy as np
import cv2

path = "6.1.tif"
img_A = cv2.imread(path)
gray_A = cv2.cvtColor(img_A, cv2.COLOR_BGR2GRAY) #转换成灰度图
ret, thresh_A = cv2.threshold(gray_A, 254, 255, cv2.THRESH_BINARY) #灰度图转换成二值图像

thresh_A_copy = thresh_A.copy() #复制thresh_A到thresh_A_copy
thresh_B = np.zeros(gray_A.shape, np.uint8) #thresh_B大小与A相同，像素值为0

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))#3×3结构元

count = [ ] #为了记录连通分量中的像素个数

#循环，直到thresh_A_copy中的像素值全部为0
while thresh_A_copy.any():

    Xa_copy, Ya_copy = np.where(thresh_A_copy > 0) #thresh_A_copy中值为255的像素的坐标
    thresh_B[Xa_copy[0]][Ya_copy[0]] = 255 #选取第一个点，并将thresh_B中对应像素值改为255

    #连通分量算法，先对thresh_B进行膨胀，再和thresh_A执行and操作（取交集）
    for i in range(200):
        dilation_B = cv2.dilate(thresh_B, kernel, iterations=1)
        thresh_B = cv2.bitwise_and(thresh_A, dilation_B)

    #取thresh_B值为255的像素坐标，并将thresh_A_copy中对应坐标像素值变为0
    Xb, Yb = np.where(thresh_B > 0)
    thresh_A_copy[Xb, Yb] = 0

    #显示连通分量及其包含像素数量
    count.append(len(Xb))
    if len(count) == 0:
        print("无连通分量")
    if len(count) == 1:
        print("第1个连通分量为{}".format(count[0]))
    if len(count) >= 2:
        print("第{}个连通分量为{}".format(len(count), count[-1] - count[-2]))

cv2.imshow("A", thresh_A)
cv2.imshow("A_copy", thresh_A_copy)
cv2.imshow("B", thresh_B)
cv2.waitKey(0)