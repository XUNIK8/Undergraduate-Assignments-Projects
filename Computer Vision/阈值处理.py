import numpy as np
import cv2





# 区域生长部分
def getGrayDiff(img, currentPoint, tmpPoint):
    return abs(int(img[currentPoint[0], currentPoint[1]]) - int(img[tmpPoint[0], tmpPoint[1]]))


def selectConnects( p):
    if p != 0:
        # connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
        #             Point(0, 1), Point(-1, 1), Point(-1, 0)]

        # connects = [self.Point(-1, -1), self.Point(0, -1), self.Point(1, -1), self.Point(1, 0), self.Point(1, 1), \
        #             self. Point(0, 1), self.Point(-1, 1), self.Point(-1, 0)]

        connects = [[-1, -1], [0, -1], [1, -1], [1, 0], [1, 1], \
                    [0, 1], [-1, 1], [-1, 0]]
    # else:
    #     connects = [self.Point(0, -1), self.Point(1, 0), self.Point(0, 1), self.Point(-1, 0)]
    return connects


# 区域生长函数
def regionGrowProcess(img, seeds, thresh, p=1):
    height, width = np.shape(img)

    seedMark = np.zeros(np.shape(img))
    seedList = seeds

    # for seed in seeds:
    #     seedList.append(seed)

    print(seedList)
    label = 1
    num = 0
    connects = selectConnects(p)
    while (len(seedList) > 0):
        num += 1
        currentPoint = seedList.pop(0)

        ## 没改完
        seedMark[currentPoint[0], currentPoint[1]] = label

        for i in range(8):
            tmpX = currentPoint[0] + connects[i][0]
            tmpY = currentPoint[1] + connects[i][1]
            # print('666')
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= width:
                continue
            tmpPoint = [tmpX, tmpY]
            # print('777')
            # print(tmpPoint)
            grayDiff = getGrayDiff(img, currentPoint, tmpPoint)
            # print('888')
            # print(grayDiff)
            if grayDiff < thresh and seedMark[tmpX, tmpY] == 0:
                num += 1
                seedMark[tmpX, tmpY] = label
                seedList.append([tmpX, tmpY])


    print('000')
    print('num', num)
    print(seedMark)
    return seedMark


img = cv2.imread('6.2_chazhi.jpg', 0)
seeds = [[100, 100]]
binaryImg = regionGrowProcess(img, seeds, 10,1)
print(binaryImg)
cv2.imshow(' ', binaryImg)
cv2.waitKey(0)


