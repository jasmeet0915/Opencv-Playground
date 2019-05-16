import cv2
import math
img = cv2.imread("bhailog.JPG", 1)
threshold = 220
img = cv2.resize(img, (600, 550))

def color_dist(b1, g1, r1):
    dist = (int(255-b1) * int(255-b1)) + (int(g1) * int(g1)) + (int(r1) * int(r1)) #calculate similarity to pure blue color
    return math.sqrt(dist)


for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        b = img[y, x][0]
        g = img[y, x][1]
        r = img[y, x][2]
        dist_color = color_dist(b, g, r)
        if dist_color <= threshold:
            img[y, x] = [0, 0, 0]
            print(dist_color)


cv2.imshow("color detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
