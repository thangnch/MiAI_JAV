import cv2
import os

image = cv2.imread("jav1.jpg")

# Tính toán phần cao và rộng của ảnh từng khuôn mặt
W = image.shape[1]//10
H = image.shape[0]//5

cv2.imshow("A", image)

# Đọc tên các em từ trái sang phải, trên xuống dưới
namelist = []
infile = open('name.txt','r')
for line in infile:
    namelist.append(line.strip())
infile.close()

# Bắt đầu loop và cắt
c = 0
for i in range(1,6):
    for j in range(1,11):
        c +=1
        # Trừ đi 35 pixel tên bên dưới
        imageij = image[(i-1)*H:i*H-35,(j-1)*W:j*W]

        # Ghi vào thư mục
        os.mkdir("data/raw/" + str(namelist[c-1]))
        cv2.imwrite("data/raw/" + str(namelist[c-1]) + "/" + str(namelist[c-1])  + ".png",imageij)
