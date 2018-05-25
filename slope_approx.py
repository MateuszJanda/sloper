#! /usr/bin/env python2
# -*- coding: utf-8 -*-

import collections as co
import itertools as it
import numpy as np
import cv2


CALIBRATION_AREA_SIZE = 40
BLACK = 0
WHITE = 255

Point = co.namedtuple('Point', ['x', 'y'])


def calibration_area(img):
    """
    Calculate calibration area.
    Return top-left point where cell start, cell width, cell hight
    """
    under_pos1, under_pos2 = underscore_pos(img)
    roof = roof_pos(img, under_pos1, under_pos2)
    sep_high = separator_high(img, under_pos1, under_pos2)

    width = width = under_pos2.x - under_pos1.x + 1
    high = under_pos2.y - roof.y + sep_high
    start_pt = Point(under_pos1.x, under_pos2.y - high)

    print('Cell size: ', start_pt, (width, high))
    return start_pt, width, high


def underscore_pos(img):
    """ Calculate underscore "_" position [top-left point, bottom-right point]"""
    pt1 = None
    for x, y in it.product(range(CALIBRATION_AREA_SIZE), range(CALIBRATION_AREA_SIZE)):
        if img[y, x] != BLACK:
            pt1 = Point(x, y)
            break

    tmp = None
    for x in range(pt1.x, CALIBRATION_AREA_SIZE):
        if img[pt1.y, x] == BLACK:
            break
        tmp = Point(x, pt1.y)

    pt2 = None
    for y in range(tmp.y, CALIBRATION_AREA_SIZE):
        if img[y, tmp.x] == BLACK:
            break
        pt2 = Point(tmp.x, y)

    print('Underscore top left: ', pt1)
    print('Underscore bottom right: ', pt2)
    return pt1, pt2


def roof_pos(img, under_pos1, under_pos2):
    """ Calculate roof sign "^" position - only the pick """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_pos2.x - under_pos1.x + 1

    for x in range(under_pos2.x + 1, under_pos2.x + width):
        for y in range(CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    print('Roof pos: ', roof)
    return roof


def separator_high(img, under_pos1, under_pos2):
    """ Calculate separator area between underscore "_" and bottom roof sign "^" """
    roof = Point(0, CALIBRATION_AREA_SIZE)
    width = under_pos2.x - under_pos1.x + 1

    for x in range(under_pos1.x, under_pos1.x + width):
        for y in range(under_pos2.y + 1, CALIBRATION_AREA_SIZE):
            if img[y, x] != BLACK and y < roof.y:
                roof = Point(x, y)

    high = roof.y - under_pos2.y

    print('Separator high: ', high)
    return high


def draw_filled_cell(img, start_pt, width, high):
    """ Just for debug purpose, will cell with color """
    for x in range(start_pt.x, start_pt.x + width):
        for y in range(start_pt.y, start_pt.y + high):
            img[y, x] ^= 158


def draw_net(img, start_pt, width, high):
    """ Just for debug purpose draw net """
    # black = (0, 0, 0)
    # white = (255, 255, 255)
    for x in range(start_pt.x, img.shape[1], width):
        cv2.line(img, (x, start_pt.y), (x, img.shape[0]), WHITE, 1)

    for y in range(start_pt.y, img.shape[0], high):
        cv2.line(img, (start_pt.x, y), (img.shape[1], y), WHITE, 1)


def erase_calibration_area(img):
    """ Erase calibration are from image """
    cv2.rectangle(img, (0, 0), (CALIBRATION_AREA_SIZE, CALIBRATION_AREA_SIZE), BLACK, cv2.FILLED)


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
    kernel = np.ones((5, 5), np.uint8)
    kernel = np.ones((55, 55), np.uint8)
    kernel = np.ones((8, 8), np.uint8)
    kernel = np.ones((10, 8), np.uint8)
    kernel2 = np.ones((8, 5), np.uint8)

    dilation_img = cv2.dilate(img, kernel, iterations=1)
    erosion_img = cv2.erode(dilation_img, kernel2, iterations=1)
    cv2.imshow('erosion_img', erosion_img)


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


def find_closest(head_cnt, contours, dbg=False):
    minimal_dist = 15
    best_cnt = None
    for cnt in contours:
        for head_pos, cnt_pos in it.product(head_cnt, cnt):
            dist = np.linalg.norm(head_pos-cnt_pos)

            if abs(dist) < minimal_dist:
                minimal_dist = abs(dist)
                best_cnt = cnt

    return best_cnt

def connect_nearby_contours(gray_img):
    """
    https://dsp.stackexchange.com/questions/2564/opencv-c-connect-nearby-contours-based-on-distance-between-them
    http://answers.opencv.org/question/169492/accessing-all-points-of-a-contour/
    """
    im2, contours, hierarchy = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    last = contours.pop(0)
    chain = [last]

    while len(contours) > 0:
        cnt = find_closest(last, contours)

        if cnt is None:
            print 'len', len(contours)
            print 'Error'
            exit()

        chain.append(cnt)
        for i in range(len(contours)):
            if np.all(cnt == contours[i]):
                contours.pop(i)
                break
        last = cnt


    cont = np.vstack(chain[i] for i in range(len(chain[:2])))
    # cont = np.vstack(c for c in contours[:2])
    # cont = np.vstack(contours[i] for i in len(contours))
    cv2.drawContours(gray_img, cont, -1, WHITE, 2)


def main():
    orig_img = cv2.imread('ascii_fig.png', cv2.IMREAD_GRAYSCALE)
    # Image should have white characters and black background
    # gray_img = cv2.bitwise_not(orig_img)
    _, gray_img= cv2.threshold(src=orig_img, thresh=30, maxval=255, type=cv2.THRESH_BINARY)
    # cv2.imshow('asdf', gray_img)

    start_pt, width, high = calibration_area(gray_img)
    erase_calibration_area(gray_img)
    # draw_filled_cell(orig_img, start_pt, width, high)
    # draw_net(orig_img, start_pt, width, high)

    # convexity_defects(orig_img)
    # morphological_transformations(gray_img)
    # canny_edge_detection(orig_img)
    connect_nearby_contours(gray_img)

    cv2.imshow('orig_img', orig_img)
    cv2.imshow('gray_img', gray_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
