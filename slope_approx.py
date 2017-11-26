import numpy as np
import cv2

MAX = 40

def underscore_pos(img):
    pt1 = None
    for x in range(MAX):
        for y in range(MAX):
            if img[y, x] != 255:
                pt1 = (x, y)
                break
        if pt1:
            break

    print('Underscore top left: ', pt1)

    tmp = None
    for x in range(pt1[0], MAX):
        if img[pt1[1], x] == 255:
            break
        tmp = (x, pt1[1])

    pt2 = None
    for y in range(tmp[1], MAX):
        if img[y, tmp[0]] == 255:
            break
        pt2 = (tmp[0], y)

    print('Underscore bottom right: ', pt2)

    return pt1, pt2


def roof_pos(img, upos1, upos2):
    roof = (0, MAX)
    width = upos2[0] - upos1[0] + 1
    for x in range(upos2[0] + 1, upos2[0] + width):
        for y in range(MAX):
            if img[y, x] != 255 and y < roof[1]:
                roof = (x, y)

    print('Roof pos: ', roof)

    return roof


def separator_high(img, upos1, upos2):
    roof = (0, MAX)

    width = upos2[0] - upos1[0] + 1
    for x in range(upos1[0], upos1[0] + width):
        for y in range(upos2[1] + 1, MAX):

            # print x, y
            # print y - upos2[1]
            if img[y, x] != 255 and y < roof[1]:
                # print y
                roof = (x, y)

            # if img[y, x] != 255:
                # break

            # img[y, x] = 128

    # print roof

    print('Second roof pos: ', roof)
    high = roof[1] - upos2[1] + 1
    print('Separator high: ', high)
    return high

# Create a black image
# img = np.zeros((512,512,3), np.uint8)

def debug_fill(img, pt, width, high):
    for x in range(pt[0], pt[0] + width):
        for y in range(pt[1], pt[1] + high):
            img[y, x] ^= 128


def debug_cross_net(img, center, width, high):
    for x in range(center[0], img.shape[1], width):
        cv2.line(img, (x, center[1]), (x, img.shape[0]), (0, 0, 0), 1)

    for y in range(center[1], img.shape[0], high):
        cv2.line(img, (center[0], y), (img.shape[1], y), (0, 0, 0), 1)


def cell_size(img):
    upos1, upos2 = underscore_pos(img)

    roof = roof_pos(img, upos1, upos2)
    sep_high = separator_high(img, upos1, upos2)

    width = width = upos2[0] - upos1[0] + 1
    high = upos2[1] - roof[1] + 1 + sep_high

    print('Cell size: ', (width, high))

    center_pt = (upos2[0] + 1, upos2[1] + 1)
    debug_fill(img, center_pt, width, high)
    debug_cross_net(img, center_pt, width, high)

    return center_pt, width, high





img = cv2.imread('ascii_fig.png', cv2.IMREAD_GRAYSCALE)

# Draw a diagonal blue line with thickness of 5 px
# cv2.line(img, (0, 0), (40, 40), (0, 0, 0), 1)

# cv2.putText(img, "Hello world!", )
# x = 0
# y = 10
# cv2.putText(img,"Hello  World!!!", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

print img.shape

cell_size(img)






cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()