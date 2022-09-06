import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def notch_filter(filename = '实验3图2.tif'):

    img = Image.open(filename)
    Height_original, Width_original = np.shape(img)    # 获取宽高信息

    # 参数
    N = 4
    D0k = 15

    #   DFT
    img_float32 = np.float32(img)
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)

    # 频谱图像双通道复数转换为0-255区间
    magitude_spectrum = 8*np.log(np.abs(dft_shift))
    magitude_spectrum_int = np.zeros((Height_original, Width_original), dtype='uint8')
    for x in range(Height_original):
        for y in range(Width_original):
            index = np.int(magitude_spectrum[x][y])
            magitude_spectrum_int[x][y] = index
            pass
        pass

    # 标记图片亮点坐标
    def mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(img, (x, y), 1, (255, 255, 255), thickness=-1)
            cv2.putText(img, xy, (x-25, y-25), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (255, 255, 255), thickness=1)
            cv2.imshow("image", img)

    print(np.shape(magitude_spectrum_int))
    img = magitude_spectrum_int
    cv2.namedWindow("image")
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mouse)
    #  显示频谱图
    cv2.waitKey(0)
    #cv2.destroyAllWindows()


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
    centerY = np.floor(Width_original / 2)   # 84.0 168
    print(centerX)
    # 构建滤波器传递函数H_NR

    H_NR = np.ones((Height_original, Width_original))
    Dk = np.zeros((2, 4))
    if filename == '实验3图1.tif':
        for u in range(Height_original):
            for v in range(Width_original):
                Dk[0][0] = np.sqrt((u - centerX - p1[0]) ** 2 + (v - centerY - p1[1])**2)
                Dk[0][1] = np.sqrt((u - centerX - p2[0]) ** 2 + (v - centerY - p2[1])**2)
                Dk[0][2] = np.sqrt((u - centerX - p3[0]) ** 2 + (v - centerY - p3[1])**2)
                Dk[0][3] = np.sqrt((u - centerX - p4[0]) ** 2 + (v - centerY - p4[1])**2)

                Dk[1][0] = np.sqrt((u - centerX + p1[0])**2 + (v - centerY + p1[1])**2)
                Dk[1][1] = np.sqrt((u - centerX + p2[0]) ** 2 + (v - centerY + p2[1])**2)
                Dk[1][2] = np.sqrt((u - centerX + p3[0]) ** 2 + (v - centerY + p3[1])**2)
                Dk[1][3] = np.sqrt((u - centerX + p4[0]) ** 2 + (v - centerY + p4[1])**2)

                for i in Dk[0]:
                    H_NR[u][v] = H_NR[u][v] * (1/(1 + (D0k/i+0.0001)**(2*N))+0.0001)
                    pass
                for i in Dk[1]:
                    H_NR[u][v] = H_NR[u][v] * (1/(1 + (D0k/i+0.0001)**(2*N))+0.0001)
                    pass
                pass
            pass
    else:
        for i in range(Height_original):
            for j in range(int(Width_original / 2), int(Width_original / 2) + 2):
                H_NR[i][j] = 0
        for i in range(int(Height_original / 2), int(Height_original / 2) + 10):
            for j in range(int(Width_original / 2), int(Width_original / 2) + 2):
                H_NR[i][j] = 1
    # IDFT
    # 滤波器图（H_NR）
    # 滤波后频谱图Gs
    Gs = magitude_spectrum * H_NR
    # 频谱对比图（Gs为滤波后，magitude_spectrum为滤波前）
    plt.figure(1)
    plt.subplot(121)
    plt.title('The spectrum before filtering')
    plt.imshow(magitude_spectrum, cmap='gray')
    plt.subplot(122)
    plt.title('The spectrum after filtering')
    plt.imshow(Gs, cmap = 'gray')

    # 逆变换回图像
    G = dft_shift * H_NR
    iimg = np.fft.ifft2(G)
    iimg = np.abs(iimg)

    # 图像对比图(iimg为滤波后，img_float32为滤波前）
    plt.figure(2)
    plt.subplot(121)
    plt.title('Before the filtering')
    plt.imshow(img_float32, cmap = 'gray')
    plt.subplot(122)
    plt.title('After the filtering')
    plt.imshow(iimg, cmap = 'gray')
    plt.show()


if __name__ == '__main__':
    notch_filter()