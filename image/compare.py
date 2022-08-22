import cv2

# 读取图片
baseImg = cv2.imread("compare1_1.png")
curImg = cv2.imread("compare1_2.png")
baseImg = cv2.resize(baseImg, (236, 1128))
curImg = cv2.flip(curImg, 1)
# 转灰度图
gray_base = cv2.cvtColor(baseImg, cv2.COLOR_BGR2GRAY)
gray_cur = cv2.cvtColor(curImg, cv2.COLOR_BGR2GRAY)

resImg = cv2.absdiff(gray_base, gray_cur)
cv2.imshow('1', curImg)
cv2.imshow("resImg ", resImg)
cv2.imwrite('result.jpg', resImg)
cv2.waitKey(0)
