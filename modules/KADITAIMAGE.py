import sys
import os

import numpy as np
import cv2
import urllib.request
import time


class KADITACVVision:
    def __init__(self, isUsingCam=None, addr=None, index=0):
        # write configuration
        self.frame_count = 0
        self.filenames = None
        self.fourcc = None
        self.out = None

        # get address
        self.cap = None
        self.success = False
        self.index = index
        if isUsingCam:
            while not self.success:
                try:
                    print(f"[INFO] Initialize Camera with Index {self.index}")
                    self.cap = cv2.VideoCapture(self.index)
                    if not self.cap.isOpened():
                        raise Exception(f"Cannot Open Camera by Index {self.index}")
                    ret, frame = self.cap.read()
                    if not ret:
                        raise Exception(f"Failed to Capture Frame by Index {self.index}")
                    self.success = True
                except Exception as err:
                    print(f"[ERROR] Camera Initialization Failed: {err}")
                    time.sleep(1.5)
                    self.index += 1
            print(f"[INFO] Camera Initialization Success")
        else:
            self.cap = cv2.VideoCapture(addr)

        # fps
        self._prev_time = 0
        self._new_time = 0

    def writeConfig(self, name="output.mp4", types="mp4v"):  # XVID -> avi
        self.filenames = name
        self.fourcc = cv2.VideoWriter_fourcc(*types)  # format video
        # filename, format, FPS, frame size
        frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.out = cv2.VideoWriter(self.filenames, self.fourcc, fps, (frame_width, frame_height))

    def write(self, frame):
        self.out.write(frame)

    def writeImg(self, frame, path="cats-output.png"):
        filename = path
        cv2.imwrite(filename, frame)
        with open(filename, 'ab') as f:
            f.flush()
            os.fsync(f.fileno())

    def resize(self, image, width=None, height=None,
               interpolation=cv2.INTER_AREA):
        dim = None
        w = image.shape[1]
        h = image.shape[0]
        if width is None and height is None:
            return image
        if width is None:
            r = height / float(h)
            dim = (int(w * r), height)
        else:
            r = width / float(w)
            dim = (width, int(h * r))
        resized = cv2.resize(image, dim, interpolation=interpolation)
        return resized

    def __get_fps(self):
        fps = 0.0
        try:
            self._new_time = time.time()
            fps = 1 / (self._new_time - self._prev_time)
            self._prev_time = self._new_time
            fps = 30 if fps > 30 else 0 if fps < 0 else fps
        except ZeroDivisionError as e:
            pass
        return int(fps)

    def blur(self, frame=None, sigma=11):
        return cv2.GaussianBlur(frame, (sigma, sigma), 0)

    def autoContrast(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        min_val, max_val, _, _ = cv2.minMaxLoc(gray)
        adjusted_image = cv2.convertScaleAbs(frame, alpha=255.0 / (max_val - min_val),
                                             beta=-min_val * 255.0 / (max_val - min_val))
        return adjusted_image

    def adaptiveContrast(self, frame, clip_limit=2.0, tile_grid_size=(8, 8)):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        enhanced_gray = clahe.apply(gray)
        enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
        return enhanced_image

    def equalizeHistogram(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized_gray = cv2.equalizeHist(gray)
        equalized_image = cv2.cvtColor(equalized_gray, cv2.COLOR_GRAY2BGR)
        return equalized_image

    def enhanceColors(self, frame, saturation_factor):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = hsv[..., 1] * saturation_factor
        hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
        enhanced_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced_image

    def sharpenImage(self, frame, sharp):
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, 1 + sharp, blurred, -sharp, 0)
        return sharpened

    def sharpen(self, frame):
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
        return sharpened

    def setBrightness(self, frame, value):
        h, s, v = cv2.split(
            cv2.cvtColor(frame, cv2.COLOR_BGR2HSV))
        v = np.clip(v.astype(int) + value, 0, 255).astype(np.uint8)
        return cv2.cvtColor(
            cv2.merge((h, s, v)), cv2.COLOR_HSV2BGR)

    def setContrast(self, frame, value):
        alpha = float(131 * (value + 127)) / (127 * (131 - value))
        gamma = 127 * (1 - alpha)
        return cv2.addWeighted(
            frame, alpha, frame, 0, gamma)

    def setBrightnessNcontrast(self, frame, bright=0.0, contr=0.0, beta=0.0):
        return cv2.addWeighted(frame, 1 + float(contr)
                               / 100.0, frame, beta, float(bright))

    def read(self, frame_size=0, show_fps=False):
        try:
            success, frame = self.cap.read()
            if not success:
                raise RuntimeError
            if show_fps:
                try:  # put fps
                    cv2.putText(frame, str(self.__get_fps()) + " fps", (20, 40), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                except RuntimeError as e:
                    print(e)
            if frame_size != 0:
                return self.resize(frame, frame_size)
            return frame
        except RuntimeError as e:
            print("[INFO] Failed to capture the Frame")

    def readFromUrl(self, url="http://192.168.200.24/cam-hi.jpg", frame_size=480, show_fps=False):
        try:
            img_resp = urllib.request.urlopen(url)
            imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
            frame = cv2.imdecode(imgnp, -1)
            if show_fps:
                try:  # put fps
                    cv2.putText(frame, str(self.__get_fps()) + " fps", (20, 40), 0, 1,
                                [225, 255, 255], thickness=2, lineType=cv2.LINE_AA)
                except RuntimeError as e:
                    print(e)
            frame = self.resize(frame, frame_size)
            return frame
        except RuntimeError as e:
            print("[INFO] Failed to capture the Frame")

    def show(self, frame, winName="frame"):
        cv2.imshow(winName, frame)

    def wait(self, delay):
        return cv2.waitKey(delay)

    def release(self):
        self.cap.release()

    def destroy(self):
        cv2.destroyAllWindows()
