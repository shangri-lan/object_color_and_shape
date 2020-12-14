# coding=utf-8
# python detect_color_single.py -i ./imgs/1.png

from shapedetector import ShapeDetector
from colorlabeler import ColorLabeler, ColorLabeler2
from imutils import contours
import numpy as np
import argparse
import imutils
import glob
import cv2
import os

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the input image")
    args = vars(ap.parse_args())

    if 1:
        image = cv2.imread(args["image"])
        image1 = image.copy()
        resized = cv2.resize(image, (image.shape[1], image.shape[0]))
        image_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        ratio = image.shape[0] / float(resized.shape[0])
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
        lab = cv2.cvtColor(blurred, cv2.COLOR_BGR2LAB)
        thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
        # thresh = cv2.Canny(gray, 50, 100)
        # thresh = cv2.dilate(thresh, None, iterations=2)
        # thresh = cv2.erode(thresh, None, iterations=2)
        cv2.imshow("Thresh", thresh)
        cv2.waitKey(0)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (cnts, _) = contours.sort_contours(cnts)

        sd = ShapeDetector()
        cl = ColorLabeler()

        # for c in cnts:
        if 1:
            if len(cnts) > 1:
                c = cnts[1]
            else:
                c = cnts[0]
            M = cv2.moments(c)
            cX = int((M["m10"] / M["m00"]) * ratio)
            cY = int((M["m01"] / M["m00"]) * ratio)

            shape, approx = sd.detect(c)
            color, color_value = cl.label(lab, c)
            # mean = cl2.get_mean(lab, c)
            # color = cl2.getColorName(mean)

            c = c.astype("float")
            c *= ratio
            c = c.astype("int")
            text = "{} {}".format(color, shape)
            print(text)
            with open('result.txt', 'w') as f:
                f.write(text)
                f.write('\n')

            cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
            cv2.putText(image, text, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.fillConvexPoly(image1, np.array(approx, np.int32), color_value)

            cv2.imwrite('result.png', image)
            cv2.imwrite('mask.png', image)
            cv2.imshow("Image", image)
            cv2.waitKey(0)