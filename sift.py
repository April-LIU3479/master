import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import cv2
ratio = 0.85
str_picture ="003"
image1 = mpimg.imread("./raw/"+str_picture+'.jpg')
#image2 = mpimg.imread("./after/004.jpg")
image2 =cv2.rotate(image1,cv2.ROTATE_90_CLOCKWISE)
'''
第一个参数：旋转中心点
第二个参数：旋转角度
第三个参数：缩放比例
'''
image1=cv2.resize(image1, (640,640))
image2=cv2.resize(image2, (640,640))


plt.figure()
plt.imshow(image1)
plt.savefig('image1.png', dpi = 300)

plt.figure()
plt.imshow(image2)
plt.savefig('image2.png', dpi = 300)
#  计算特征点提取&生成描述时间
start = time.time()
sift = cv2.SIFT_create()
#  使用SIFT查找关键点key points和描述符descriptors
kp1, des1 = sift.detectAndCompute(image1, None)
kp2, des2 = sift.detectAndCompute(image2, None)
end = time.time()
kp_image1 = cv2.drawKeypoints(image1, kp1, None)
kp_image2 = cv2.drawKeypoints(image2, kp2, None)

plt.figure()
plt.imshow(kp_image1)
plt.savefig('kp_image1.png', dpi = 300)

plt.figure()
plt.imshow(kp_image2)
plt.savefig('kp_image2.png', dpi = 300)
#  K近邻算法求取在空间中距离最近的K个数据点，并将这些数据点归为一类
matcher = cv2.BFMatcher()
raw_matches = matcher.knnMatch(des1, des2, k = 2)
good_matches = []
for m1, m2 in raw_matches:
    #  如果最接近和次接近的比值大于一个既定的值，那么我们保留这个最接近的值，认为它和其匹配的点为good_match
    if m1.distance < ratio * m2.distance:
        good_matches.append([m1])
tem_good_matches=[]
for i in range(len(good_matches)):
    if i%20 ==0:
        tem_good_matches.append(good_matches[i])
end = time.time()
print("匹配点匹配运行时间:%.2f秒"%(end-start))

matches = cv2.drawMatchesKnn(kp_image1, kp1, kp_image2, kp2, tem_good_matches, None, flags = 2)

plt.figure()
plt.axis('off') # 去坐标轴
plt.xticks([]) # 去刻度
plt.imshow(matches)
plt.savefig(str_picture+'.jpeg',bbox_inches='tight', pad_inches = -0.1, dpi = 300)