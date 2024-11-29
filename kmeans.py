from sklearn.cluster import KMeans
import numpy as np
import cv2

# 读取图像
image = cv2.imread('/home/zjj/xjd/datasets/geoeye-1/geo_test/img/3_3_6.tif')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 将图像数据转换为二维数组
pixel_values = image.reshape((-1, 3))
pixel_values = np.float32(pixel_values)

# 定义停止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01)

# 应用kmeans函数
k = 3
_, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将结果转换回8位
centers = np.uint8(centers)
print(centers)

segmented_image = centers[labels.flatten()]

# 将结果重塑为原始图像
segmented_image = segmented_image.reshape(image.shape)

# 显示图像
cv2.imwrite('Segmented Image.png', segmented_image)


