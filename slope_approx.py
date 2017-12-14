import collections as col
import numpy as np
from matplotlib import pyplot as plt
import cv2


CALIBRATION_AREA_SIZE = 40

Point = col.namedtuple('Point', ['x', 'y'])


def underscore_pos(img):
    pt1 = None
    for x in range(CALIBRATION_AREA_SIZE):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != 255:
                pt1 = Point(x, y)
                break
        if pt1:
            break

    print('Underscore top left: ', pt1)

    tmp = None
    for x in range(pt1.x, CALIBRATION_AREA_SIZE):
        if img[pt1.y, x] == 255:
            break
        tmp = Point(x, pt1.y)

    pt2 = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == 255:
            break
        pt2 = Point(tmp.x, y)

    print('Underscore bottom right: ', pt2)

    return pt1, pt2


def roof_pos(img, upos1, upos2):
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = upos2.x - upos1.x + 1
    for x in range(upos2.x + 1, upos2.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != 255 and y < roof.y:
                roof = Point(x, y)

    print('Roof pos: ', roof)

    # cv2.line(img, (roof.x, roof.y), (img.shape[1], roof.y), (0, 0, 0), 1)

    return roof


def separator_high(img, upos1, upos2):
    roof = Point(0, CALIBRATION_AREA_SIZE)

    width = upos2.x - upos1.x + 1
    for x in range(upos1.x, upos1.x + width):
        for y in range(upos2.y + 1, CALIBRATION_AREA_SIZE):

            # print x, y
            # print y - upos2.y
            if img[y, x] != 255 and y < roof.y:
                # print y
                roof = Point(x, y)

            # if img[y, x] != 255:
                # break

            # img[y, x] = 128

    # print roof

    # cv2.line(img, (roof.x, roof.y), (img.shape.y, roof.y), (0, 0, 0), 1)
    # cv2.line(img, (upos2.x, upos2.y), (img.shape.y, upos2.y), (0, 0, 0), 1)
    # cv2.line(img, (upos2.x, upos2.y), (img.shape.y, upos2.y), (0, 0, 0), 1)

    print('Second roof pos: ', roof)
    high = roof.y - upos2.y

    # cv2.line(img, (upos1.x, upos2.y + high), (img.shape[1], upos2.y + high), (0, 0, 0), 1)

    print('Separator high: ', high)
    return high

# Create a black image
# img = np.zeros((512,512,3), np.uint8)

def debug_fill_cell(img, pt, width, high):
    for x in range(pt.x, pt.x + width):
        for y in range(pt.y, pt.y + high):
            img[y, x] ^= 58


def debug_cross_net(img, center, width, high):
    for x in range(center.x, img.shape[1], width):
        cv2.line(img, (x, center.y), (x, img.shape[0]), (0, 0, 0), 1)

    for y in range(center.y, img.shape[0], high):
        cv2.line(img, (center.x, y), (img.shape[1], y), (0, 0, 0), 1)


def cell_size(img):
    upos1, upos2 = underscore_pos(img)

    roof = roof_pos(img, upos1, upos2)
    sep_high = separator_high(img, upos1, upos2)

    width = width = upos2.x - upos1.x + 1
    high = upos2.y - roof.y + sep_high

    print('Cell size: ', (width, high))

    center_pt = Point(upos1.x, upos2.y - high)
    # debug_fill_cell(img, center_pt, width, high)
    # debug_cross_net(img, center_pt, width, high)

    return center_pt, width, high


def erase_calibration_area(img):
    pass


def convexity_defects(img):
    """
    https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_contours/py_contours_more_functions/py_contours_more_functions.html#contours-more-functions
    """
    ret, thresh = cv2.threshold(img, 127, 255,0)
    _, contours,hierarchy = cv2.findContours(thresh,2,1)
    cnt = contours[0]

    hull = cv2.convexHull(cnt,returnPoints = False)
    defects = cv2.convexityDefects(cnt,hull)

    for i in range(defects.shape[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        cv2.line(img,start,end,[0,255,0],2)
        cv2.circle(img,far,5,[0,0,255],-1)


def morphological_transformations(img):
    kernel = np.ones((5,5),np.uint8)
    kernel = np.ones((55,55),np.uint8)


    erosion = cv2.erode(img,kernel,iterations = 1)
    # dilation = cv2.dilate(img,kernel,iterations = 1)
    # pass

def canny_edge_detection(img):
    """
    https://docs.opencv.org/3.1.0/da/d22/tutorial_py_canny.html
    """
    # img = cv2.imread('messi5.jpg',0)
    edges = cv2.Canny(img,100,200)

    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

    plt.show()


def find_if_close(cnt1, cnt2):
    print cnt1
    row1,row2 = cnt1.shape[0], cnt2.shape[0]
    for i in xrange(row1):
        for j in xrange(row2):
            # print cnt1[i]-cnt2[j]
            dist = np.linalg.norm(cnt1[i]-cnt2[j])
            # print dist
            # import sys
            # sys.exit()

            if abs(dist) < 15:
                return True
            elif i==row1-1 and j==row2-1:
                return False

def connect_nearby_contours(gray_img):
    """
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    """
    # gray = cv2.bitwise_not(img)

    _, thresh = cv2.threshold(src=gray_img, thresh=7, maxval=255, type=cv2.THRESH_BINARY)
    # cv2.imshow('debug1', thresh)

    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cv2.imshow('debug', im2)

    LENGTH = len(contours)
    status = np.zeros((LENGTH,1))

    chain = [0]

    # for i, cnt1 in enumerate(contours):
    #     x = i
    #     if i != LENGTH-1:
    #         for j, cnt2 in enumerate(contours[i+1:]):
    #             x = x+1
    #             # print x, j+1
    #             dist = find_if_close(cnt1,cnt2)
    #             if dist == True:
    #                 val = min(status[i],status[x])
    #                 status[x] = status[i] = val
    #             else:
    #                 if status[x]==status[i]:
    #                     status[x] = i+1

    for i, cnt1 in enumerate(contours):
        for j, cnt2 in enumerate(contours[i+1:]):
            dist = find_if_close(cnt1, cnt2)
            if dist == True:
                # status[x] = status[i] = val
                chain.append(j)
            # else:
            #     if status[x]==status[i]:
            #         status[x] = i+1



    # print contours[0]
    print len(contours)

    unified = []
    # maximum = int(status.max())+1
    # for i in xrange(maximum):
    #     pos = np.where(status==i)[0]
    #     if pos.size != 0:
    #         cont = np.vstack(contours[i] for i in pos)
    #         hull = cv2.convexHull(cont)
    #         unified.append(hull)

    # for c in contours:
    # for i in range(4):

    # cont = np.vstack(contours[i] for i in range(4))
    # hull = cv2.convexHull(cont)
    # unified.append(hull)
    # cv2.drawContours(img,unified,-1,(0,255,0),2)

    cont = np.vstack(contours[i] for i in range(len(contours)))
    # hull = cv2.convexHull(cont)
    unified.append(cont)
    cv2.drawContours(img,unified,-1,(0,255,0),2)

    # cont = np.vstack(contours[i] for i in range(len(contours)))
    # hull = cv2.convexHull(cont)
    # unified.append(hull)
    # cv2.polylines(img, cont, True, (0,255,255), 2)

    # cv2.drawContours(thresh,unified,-1,255,-1)

    # print len(contours)
    # print cv2.CHAIN_APPROX_SIMPLE


    # img = thresh


# -----------------------

img = cv2.imread('ascii_fig.png', cv2.IMREAD_GRAYSCALE)


# gray_img = cv2.dilate(src=gray_img, kernel=np.ones((5, 15)), iterations=1)

# Draw a diagonal blue line with thickness of 5 px
# cv2.line(img, (0, 0), (40, 40), (0, 0, 0), 1)

# cv2.putText(img, "Hello world!", )
# x = 0
# y = 10
# cv2.putText(img,"Hello  World!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

print('Image shape:', img.shape)

cell_size(img)

cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), 255, cv2.FILLED)

gray_img = cv2.bitwise_not(img)

# erase_calibration_area(img)

# convexity_defects(img)
# morphological_transformations(img)
# canny_edge_detection(img)
connect_nearby_contours(gray_img)

# img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)




cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()