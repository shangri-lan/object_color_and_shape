# coding=utf-8
from scipy.spatial import distance as dist
from collections import OrderedDict
from six.moves import xrange
import pandas as pd
import numpy as np
import cv2


class ColorLabeler:
    def __init__(self):
        self.colors = OrderedDict({
            "Red": (255, 0, 0),
            "Green": (0, 255, 0),
            "Blue": (0, 0, 255),
            "White": (255, 255, 255),
            "Black": (0, 0, 0),
            "Yellow": (255, 255, 0),
            "Cyan": (0, 255, 255),
            "Fuchsia": (255, 0, 255),
    
            "Silver": (192, 192, 192),
            "Gray": (128, 128, 128),
            "Maroon": (128, 0, 0),
            "Olive": (128, 128, 0),
            "Lime": (0, 128, 0),
            "Purple": (128, 0, 128),
            "Teal": (0, 128, 128),
            "Navy": (0, 0, 128)})

        self.lab = np.zeros((len(self.colors), 1, 3), dtype="uint8")
        self.colorNames = []

        for (i, (name, rgb)) in enumerate(self.colors.items()):
            self.lab[i] = rgb
            self.colorNames.append(name)

        self.lab = cv2.cvtColor(self.lab, cv2.COLOR_BGR2LAB)

    def label(self, image, c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        mean = cv2.mean(image, mask=mask)[:3]

        minDist = (np.inf, None)

        for (i, row) in enumerate(self.lab):
            d = dist.euclidean(row[0], mean)

            if d < minDist[0]:
                minDist = (d, i)

        return self.colorNames[minDist[1]], self.colors[self.colorNames[minDist[1]]]


class ColorLabeler2:
    def __init__(self):
        self.index = ["color", "color_name", "hex", "R", "G", "B"]
        self.csv = pd.read_csv('colors.csv', names=self.index, header=None)

    def get_mean(self, image, c):
        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        mask = cv2.dilate(mask, None, iterations=2)
        mask = cv2.erode(mask, None, iterations=2)
        new_mask = cv2.add(image, np.zeros(np.shape(image), dtype=np.uint8), mask=mask)
        print(new_mask.shape)
        max_rgb = self.compute_mask(new_mask, c)
        return max_rgb
        
    def judge_list(self, lists1, list2):
        for list1 in lists1:
            if (str(list2[:]) == str(list1[:])):
                return True
        return False
        
    def compute_mask(self, mask, contours):
        rgb_dicts = {}
        rgb_lists = []
        img_w, img_h = mask.shape[1], mask.shape[0]
        for i in xrange(img_h):
            for j in xrange(img_w):
                if (cv2.pointPolygonTest(contours, (i, j), True) > 0):
                    rgb = mask[i, j, :]
                    rgb_str = str(mask[i, j, 0]) + '_' + str(mask[i, j, 1]) + '_' + str(mask[i, j, 2])
                    # print(rgb)
                    # print(cv2.pointPolygonTest(contours, (i, j), True))
                        
                    if not self.judge_list(rgb_lists, rgb):
                        rgb_lists.append(rgb)
                        rgb_dicts[rgb_str] = 0
                    else:
                        rgb_dicts[rgb_str] += 1
        max_rgb = max(rgb_dicts, key=rgb_dicts.get)
        R, G, B = int(max_rgb.split('_')[0]), int(max_rgb.split('_')[1]), int(max_rgb.split('_')[2])
        print(max_rgb)
        return [R, G, B]

    def getColorName(self, rgbs):
        R, G, B = rgbs[0], rgbs[1], rgbs[2]
        minimum = 10000
        for i in range(len(self.csv)):
            d = abs(R - int(self.csv.loc[i, "R"])) + abs(G - int(self.csv.loc[i, "G"])) + abs(B - int(self.csv.loc[i, "B"]))
            if (d <= minimum):
                minimum = d
                cname = self.csv.loc[i, "color_name"]
        return cname
    
    def draw_function(self, event, img, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            global b, g, r, xpos, ypos, clicked
            # clicked = True
            xpos = x
            ypos = y
            b, g, r = img[y, x]
            b = int(b)
            g = int(g)
            r = int(r)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_function)