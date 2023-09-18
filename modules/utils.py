import sys
import os

import numpy as np
import cv2
import time

import ast

from .filters import KalmanFilter

# MACROS
# methods
RET_EXT = cv2.RETR_EXTERNAL
RET_TREE = cv2.RETR_TREE
RET_LIST = cv2.RETR_LIST
RET_CCOMP = cv2.RETR_CCOMP


def Ticks():
    return int(time.time() * 1000)


class Contours:
    def __init__(self):
        # initial_state = [0, 0, 0, 0]  # [x, y, vx, vy]
        # initial_covariance = np.eye(4) * 10
        # measurement_noise = 1
        # process_noise = 0.1
        self.kf = KalmanFilter([0, 0, 0, 0], np.eye(4) * 10, 1, 0.1)

    def getContours(self, filtered_frame, method=cv2.RETR_EXTERNAL):
        return cv2.findContours(filtered_frame, method,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    def fill(self, size_frame, contours):
        hull = []
        drawing = np.zeros(
            (size_frame.shape[0], size_frame.shape[1], 1), np.uint8)
        for i in range(len(contours)):
            hull.append(cv2.convexHull(contours[i], False))
            cv2.drawContours(drawing, hull, i, (255, 255, 255), -1, 8)
        cv2.dilate(drawing, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2, 2)))  # 7
        drawing = cv2.dilate(
            drawing, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))  # 7
        drawing = cv2.morphologyEx(drawing, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # 11
        return drawing

    def circlePos(self, cnts):
        pos = {}
        if len(cnts) > 0:
            sortedKontur = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = max(sortedKontur, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
            try:
                # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                prediction = self.kf.update(np.array([x, y], np.float32))
                pos = {"x_pos": float(prediction[0]),
                       "y_pos": float(prediction[1]),
                       "size": radius * 2}
            except ZeroDivisionError as e:
                pass
        return pos

    def rectPos(self, cnts):
        pos = {}
        if len(cnts) > 0:
            sortedKontur = sorted(cnts, key=cv2.contourArea, reverse=True)
            c = max(sortedKontur, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            try:
                pos = {"x_pos": x,
                       "y_pos": y,
                       "width": w,
                       "height": h}
            except ZeroDivisionError as e:
                pass
        return pos


# MACROS
# element
DRAW_RECT = 0
DRAW_LINE = 1
DRAW_CIRCLE = 2
DRAW_POLYGON = 3


class Drawing:
    # def __init__(self) -> None:
    #     pass
    def drawPoints(self, frame, pos, shape, label="obj", disp_coordinates=False):
        if shape is DRAW_CIRCLE:
            try:
                if pos["size"] > 0:
                    cv2.circle(frame, (int(pos["x_pos"]), int(pos["y_pos"])), int(pos["size"] / 2),
                               (0, 255, 0), 2)
                    cv2.circle(frame, (int(pos["x_pos"]), int(pos["y_pos"])),
                               3, (0, 0, 255), -1)
                    if disp_coordinates:
                        cv2.putText(frame, label, (int(pos["x_pos"]) + 10, int(pos["y_pos"])),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                        cv2.putText(frame, "(" + str(int(pos["x_pos"])) + "," + str(int(pos["y_pos"])) + ")",
                                    (int(pos["x_pos"]) +
                                     10, int(pos["y_pos"]) + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            except KeyError as e:
                # print("Wrong pos argument")
                # print(e)
                pass
        elif shape is DRAW_RECT:
            pass


# MACROS
# element
ELE_NORM = 0
ELE_RECT = cv2.MORPH_RECT
ELE_CROSS = cv2.MORPH_CROSS
ELE_ELLIPSE = cv2.MORPH_ELLIPSE

# color space
TO_HSV = cv2.COLOR_BGR2HSV
TO_YUV = cv2.COLOR_BGR2YUV
TO_LAB = cv2.COLOR_BGR2LAB
TO_LUV = cv2.COLOR_BGR2LUV

# morph
MORPH_OPEN = cv2.MORPH_OPEN
MORPH_CLOSE = cv2.MORPH_CLOSE
MORPH_GRADIENT = cv2.MORPH_GRADIENT
MORPH_TOPHAT = cv2.MORPH_TOPHAT
MORPH_BLACKHAT = cv2.MORPH_BLACKHAT


class Filter(Contours, Drawing):
    # def __init__(self):
    #     pass

    def getElement(self, type, size):
        if not type:
            return np.ones((size, size), np.uint8)
        else:
            return cv2.getStructuringElement(type, (size, size))

    def color2(self, frame, type):
        return cv2.cvtColor(frame, type)

    def morph(self, frame, type, kernel, iterate=1):
        return cv2.morphologyEx(frame, type, kernel, iterations=iterate)


class Bitwise:
    # def __init__(self):
    #     pass

    def And(self, src1, src2, mask=None):
        return cv2.bitwise_and(src1, src2, mask=mask)

    def Or(self, src1, src2, mask=None):
        return cv2.bitwise_or(src1, src2, mask=mask)

    def Xor(self, src1, src2, mask=None):
        return cv2.bitwise_xor(src1, src2, mask=mask)

    def Not(self, src, mask=None):
        return cv2.bitwise_not(src, mask=mask)


class Blob:
    def __init__(self):
        self.params = None

    def blobSetParams(self, params=None):
        if params is not None:
            self.params = params
        else:
            self.params = cv2.SimpleBlobDetector_Params()
            self.params.minThreshold = 0
            self.params.maxThreshold = 100
            self.params.filterByArea = True
            self.params.minArea = 200
            self.params.maxArea = 40000
            self.params.filterByCircularity = True
            self.params.minCircularity = 0.1
            self.params.filterByConvexity = True
            self.params.minConvexity = 0.5
            self.params.filterByInertia = True
            self.params.minInertiaRatio = 0.5

    def blobGetParams(self):
        return self.params

    def blob(self, filtered_frame):
        detector = cv2.SimpleBlobDetector_create(self.params)
        return detector.detect(filtered_frame)

    def blobPos(self, keypoints):
        pos = {}
        if keypoints != [] or keypoints != {}:
            for keypoint in keypoints:
                pos = {"x_pos": keypoint.pt[0],
                       "y_pos": keypoint.pt[1],
                       "size": keypoint.size}
        return pos


class ColorBased(Filter, Bitwise, Blob):
    def __init__(self):
        Blob.__init__(self)
        Contours.__init__(self)
        self.__isUsingTrackbar = False
        self.__windowName = "tracking"
        self.__trackName = ["L", "L", "L",
                            "U", "U", "U"]

    def setWinName(self, name):
        self.__windowName = name

    def createTrackbar(self, winName=None, trackName=None):
        if (winName is not None and trackName is not None):
            try:
                self.__windowName = winName
                if (len(trackName) == 3):
                    for i, dat in enumerate(trackName):
                        self.__trackName[i] += dat
                        self.__trackName[i + 3] += dat
                else:
                    raise NameError("Name length must be 3")
            except NameError as e:
                print(e)
                exit()

        cv2.namedWindow(self.__windowName)
        try:
            for index, data in enumerate(self.__trackName):
                if index < (len(self.__trackName) / 2):
                    cv2.createTrackbar(
                        data, self.__windowName, 0, 255, self.nothing)
                else:
                    cv2.createTrackbar(data, self.__windowName,
                                       255, 255, self.nothing)
        except TypeError as e:
            pass
        self.__isUsingTrackbar = True

    def calibrate(self, filetered_frame, path="value"):
        if (self.__isUsingTrackbar):
            lower = []
            upper = []
            for index, data in enumerate(self.__trackName):
                if index < len(self.__trackName) / 2:
                    lower.append(cv2.getTrackbarPos(
                        data, self.__windowName))
                else:
                    upper.append(cv2.getTrackbarPos(
                        data, self.__windowName))
            val = {path + "_lower_val": lower, path +
                                               "_upper_val": upper}  # not used
            with open(path + ".txt", "w") as f:
                f.write(str(val))
            return cv2.inRange(
                filetered_frame, np.array(lower), np.array(upper))

    def load(self, path):
        low_up = []
        with open(path + ".txt", "r") as file:
            file_content = file.read()
            dict = ast.literal_eval(file_content)
            low_up.append(dict[path + "_lower_val"])
            low_up.append(dict[path + "_upper_val"])
        return low_up

    def mask(self, filetered_frame, value):
        return cv2.inRange(filetered_frame, np.array(value[0]), np.array(value[1]))

    @staticmethod
    def enableReferenceLine(frame):
        # TODO :: GARIS REFERENSI - TILT
        cv2.line(frame, (0, (frame.shape[0] // 2) - 20),
                 (frame.shape[1], (frame.shape[0] // 2) - 20), (255, 0, 255), 2)
        cv2.line(frame, (0, (frame.shape[0] // 2)),
                 (frame.shape[1], (frame.shape[0] // 2)), (0, 0, 255), 2)
        cv2.line(frame, (0, (frame.shape[0] // 2) + 20),
                 (frame.shape[1], (frame.shape[0] // 2) + 20), (255, 0, 255), 2)

        cv2.line(frame, (0, (frame.shape[0] - 85)),
                 (frame.shape[1], (frame.shape[0] - 85)), (255, 0, 0), 2)
        cv2.line(frame, (0, 85), (frame.shape[1], 85), (255, 0, 0), 2)

        # TODO :: GARIS REFERENSI - PAN
        cv2.line(frame, (100, 0), (100, frame.shape[0]), (255, 120, 120), 2)
        cv2.line(frame, (frame.shape[1] - 100, 0),
                 (frame.shape[1] - 100, frame.shape[0]), (255, 120, 120), 2)

        cv2.line(frame, ((frame.shape[1] // 2) - 20, 0),
                 ((frame.shape[1] // 2) - 20, frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, ((frame.shape[1] // 2), 0),
                 ((frame.shape[1] // 2), frame.shape[0]), (0, 0, 255), 2)
        cv2.line(frame, ((frame.shape[1] // 2) + 20, 0),
                 ((frame.shape[1] // 2) + 20, frame.shape[0]), (0, 0, 255), 2)

    def nothing(self, x):
        pass
