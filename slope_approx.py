import collections as col
import numpy as np
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
    debug_fill_cell(img, center_pt, width, high)
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