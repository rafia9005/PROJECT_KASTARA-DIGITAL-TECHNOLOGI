import random
import sys
import os
import glob

import pandas as pd
import numpy as np
import torch
import cv2
import time
import urllib.error
import serial
import adafruit_fingerprint
from PIL import Image


def Ticks():
    return int(time.time() * 1000)


class ImgRex:  # YoloV3 dan V4
    def __init__(self):
        self.output_layers = None
        self.net = None
        self.colors = None
        self.classes = None
        self.classes_path = None

    def __map(self, x, inMin, inMax, outMin, outMax):
        return (x - inMin) * (outMax - outMin) // (inMax - inMin) + outMin

    def load(self, weight_path, cfg, classes):
        self.classes = None
        self.classes_path = classes
        with open(classes, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.colors = np.random.uniform(
            0, 255, size=(len(self.classes), 3))  # optional
        self.net = cv2.dnn.readNet(weight_path, cfg)
        # self.net = cv2.dnn.readNetFromDarknet(cfg, weight_path)
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1]
                              for i in self.net.getUnconnectedOutLayers()]
        # self.output_layers = self.net.getUnconnectedOutLayersNames()

    @staticmethod
    def draw(frame, detection):
        if detection is not []:
            for idx in detection:
                color = idx["color"]
                cv2.rectangle(
                    frame, (idx["x"], idx["y"]), (idx["x"] + idx["width"], idx["y"] + idx["height"]), color, 2)
                tl = round(0.002 * (frame.shape[0] + frame.shape[1]) / 2) + 1
                c1, c2 = (int(idx["x"]), int(idx["y"])), (int(
                    idx["width"]), int(idx["height"]))

                tf = int(max(tl - 1, 1))  # font thickness
                t_size = cv2.getTextSize(
                    idx["class"], 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3

                cv2.rectangle(frame, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, idx["class"] + " " + str(int(idx["confidence"] * 100)) + "%", (c1[0], c1[1] - 2), 0,
                            tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                # cv2.putText(frame, idx["class"] + " " + str(int(random.randint(65, 75))) + "%", (c1[0], c1[1] - 2), 0,tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
                cv2.circle(frame, (
                    int(idx["x"] + int(idx["width"] / 2)), int(idx["y"] + int(idx["height"] / 2))),
                           4, color, -1)
                cv2.putText(frame, str(int(idx["x"] + int(idx["width"] / 2))) + ", " + str(
                    int(idx["y"] + int(idx["height"] / 2))), (
                                int(idx["x"] + int(idx["width"] / 2) + 10),
                                int(idx["y"] + int(idx["height"] / 2) + 10)), cv2.FONT_HERSHEY_PLAIN, tl / 2,
                            [255, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
        return frame

    def predict(self, frame):
        blob = cv2.dnn.blobFromImage(
            frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        height, width, ch = frame.shape
        self.net.setInput(blob)
        outs = self.net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []
        center = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.1:
                    # object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    center.append([center_x, center_y])
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        values = []
        indexes = cv2.dnn.NMSBoxes(
            boxes, confidences, 0.5, 0.4)  # 0.4 changeable
        font = cv2.FONT_HERSHEY_PLAIN
        for i in range(len(boxes)):
            if i in indexes:
                label = str(self.classes[class_ids[i]])
                x, y, w, h = boxes[i]
                temp = {
                    "class": label,
                    "confidence": confidences[i],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center": center[i],
                    "color": self.colors[class_ids[i]]
                }
                values.append(temp)
        return values

    def getClassMapping(self):
        class_mapping = {}
        try:
            with open(self.classes_path, 'r') as file:
                lines = file.readlines()
                for index, line in enumerate(lines):
                    class_name = line.strip()
                    class_mapping[class_name] = index
        except FileNotFoundError:
            print(f"Error: File '{self.classes_path}' not found.")
        except Exception as e:
            print(f"Error: An unexpected error occurred - {e}")
        return class_mapping


class ImgBuzz(ImgRex):  # 8
    def __init__(self):
        self.classes = None
        self.colors = None
        self.model = None
        self.count = 0

    def load(self, names, weight):
        name = open(names, "r")
        self.classes = name.read().split("\n")
        self.model = None
        try:
            # self.model = ul(weight)
            pass
        except ModuleNotFoundError:
            pass
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def predict(self, frame):
        values = []
        # height, width, ch = frame.shape
        results = self.model.predict(frame)
        res = results[0].boxes.boxes
        px = pd.DataFrame(res).astype("float")
        boxes = []
        center = []
        class_ids = []
        confidences = []
        for index, row in px.iterrows():
            confidence = row[4]
            if confidence > 0.5:
                x1, x2 = int(row[0]), int(row[2])
                y1, y2 = int(row[1]), int(row[3])
                # indexes, conf = index, row[4]
                # labels = self.classes[int(row[5])]
                boxes.append([x1, y1, x2 - x1, y2 - y1])
                center.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
                confidences.append(round(row[4], 2))
                class_ids.append(int(row[5]))
        for i in range(len(boxes)):
            x, y, w, h = boxes[i]
            temp = {
                "class": str(self.classes[class_ids[i]]),
                "confidence": confidences[i],
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center": center[i],
                "color": self.colors[class_ids[i]]
            }
            values.append(temp)
        return values


class KADITAYOLOV5(ImgRex):  # 5
    def __init__(self):
        self.classes = None
        self.colors = None
        self.model = None
        self.count = 0

    def load(self, names, weight):
        with open(names, "r") as name:
            self.classes = name.read().split("\n")
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # print(f"self.colors[0] = {self.colors[0]}, type = {type(self.colors)}")
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', weight, force_reload=True)
        # self.model = torch.hub.load('yolov5', 'custom', weight, source='local')

        count = 0
        success = False
        max_count = 100
        while not success:
            print(f"[INFO] connecting {count} ...")
            try:
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 'custom', weight)
                success = True
            except urllib.error.URLError as e:
                print(f"[ERROR] {e}")
                time.sleep(10.0)
                count += 1
        if not success:
            print(f"[ERROR] Connection not stable error code: {max_count}!!")

    def predict(self, frame):
        width = frame.shape[1]
        height = frame.shape[0]

        values = []
        results = self.model(frame)
        pred = results.pred[0]
        boxes_t = pred[:, :4].cpu().numpy()
        labels_t = pred[:, -1].cpu().numpy()
        confidences_t = pred[:, 4].cpu().numpy()

        boxes = []
        center = []
        class_ids = []
        confidences = []
        try:
            # frame = np.squeeze(results.render())
            for box, label, confidence in zip(boxes_t, labels_t, confidences_t):
                if confidence > 0.1:
                    x1, y1, x2, y2 = box
                    boxes.append([int(x1), int(y1), int(
                        x2) - int(x1), int(y2) - int(y1)])
                    center.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
                    confidences.append(round(confidence, 2))
                    class_ids.append(int(label))

            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                temp = {
                    "class": str(self.classes[class_ids[i]]),
                    "confidence": confidences[i],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center": center[i],
                    "color": self.colors[class_ids[i]]
                }
                values.append(temp)
        except TypeError:
            pass

        return values


class HogDescriptor:
    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def predict(self, frame):
        values = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        boxes, weights = self.hog.detectMultiScale(
            gray, winStride=(8, 8), padding=(32, 32), scale=1.05)
        for (x, y, w, h) in boxes:
            temp = {
                "class": "person",
                "confidence": 0.5,
                "x": x,
                "y": y,
                "width": w,
                "height": h,
                "center": [(x + w) // 2, (y + h) // 2],
                "color": np.array([255.12, 10.22, 20.3])
            }
            values.append(temp)
        return values


class FingerprintController:

    def __init__(self):
        self.uart = None
        self.finger = None
        self.id_name_map = {}

    # def initialize(self, uart_port="COM7", baud_rate=57600):
    #     self.uart = serial.Serial(uart_port, baudrate=baud_rate, timeout=1)
    #     self.finger = adafruit_fingerprint.Adafruit_Fingerprint(self.uart)

    # def get_fingerprint(self):
    #     print("Waiting for image...")
    #     while self.finger.get_image() != adafruit_fingerprint.OK:
    #         pass
    #     print("Templating...")
    #     if self.finger.image_2_tz(1) != adafruit_fingerprint.OK:
    #         return False
    #     print("Searching...")
    #     if self.finger.finger_search() != adafruit_fingerprint.OK:
    #         return False
    #     return True
    def get_fingerprint(self):
        print("Waiting for image...")
        while self.finger.get_image() != adafruit_fingerprint.OK:
            pass
        print("Templating...")
        if self.finger.image_2_tz(1) != adafruit_fingerprint.OK:
            return False
        print("Searching...")
        if self.finger.finger_search() != adafruit_fingerprint.OK:
            return False
        # Inisialisasi
        detected_id = self.finger.finger_id  # Get the detected fingerprint's ID
        if detected_id in self.id_name_map:
            detected_name = self.id_name_map[detected_id]  # Get the name associated with the ID
        else:
            detected_name = "Nama Tidak Teregister"  # Use "Unknown" as the name if the ID is not registered
        # Save the fingerprint image as a PNG file with the format "ID_Name.png"
        filename = f"{detected_name}_{detected_id}.png"
        while self.finger.get_image():
            pass
        image_saved = self.save_fingerprint_image(filename)
        if image_saved:
            print(f"Fingerprint image saved as {filename}")
        else:
            print("Failed to save fingerprint image")

        return True

    def get_fingerprint_detail(self):
        finger_condition = 0  # Inisialisasi kondisi sidik jari

        print("Getting image...", end="")
        i = self.finger.get_image()
        if i == adafruit_fingerprint.OK:
            print("Image taken")
        else:
            if i == adafruit_fingerprint.NOFINGER:
                print("No finger detected")
            elif i == adafruit_fingerprint.IMAGEFAIL:
                print("Imaging error")
            else:
                print("Other error")
            return False

        print("Templating...", end="")
        i = self.finger.image_2_tz(1)
        if i == adafruit_fingerprint.OK:
            print("Templated")
        else:
            if i == adafruit_fingerprint.IMAGEMESS:
                print("Image too messy")
            elif i == adafruit_fingerprint.FEATUREFAIL:
                print("Could not identify features")
            elif i == adafruit_fingerprint.INVALIDIMAGE:
                print("Image invalid")
            else:
                print("Other error")
            return False

        print("Searching...", end="")
        i = self.finger.finger_fast_search()
        if i == adafruit_fingerprint.OK:
            print("Found fingerprint!")
            finger_condition = 1
        elif i == adafruit_fingerprint.NOTFOUND:
            print("No match found")
            finger_condition = 0
        else:
            print("Other error")

        return finger_condition

    def enroll_finger(self, location):
        for fingerimg in range(1, 3):
            if fingerimg == 1:
                print("Place finger on sensor...", end="")
            else:
                print("Place same finger again...", end="")

            while True:
                i = self.finger.get_image()
                if i == adafruit_fingerprint.OK:
                    print("Image taken")
                    break
                if i == adafruit_fingerprint.NOFINGER:
                    print(".", end="")
                elif i == adafruit_fingerprint.IMAGEFAIL:
                    print("Imaging error")
                    return False
                else:
                    print("Other error")
                    return False

            print("Templating...", end="")
            i = self.finger.image_2_tz(fingerimg)
            if i == adafruit_fingerprint.OK:
                print("Templated")
            else:
                if i == adafruit_fingerprint.IMAGEMESS:
                    print("Image too messy")
                elif i == adafruit_fingerprint.FEATUREFAIL:
                    print("Could not identify features")
                elif i == adafruit_fingerprint.INVALIDIMAGE:
                    print("Image invalid")
                else:
                    print("Other error")
                return False

            if fingerimg == 1:
                print("Remove finger")
                time.sleep(1)
                while i != adafruit_fingerprint.NOFINGER:
                    i = self.finger.get_image()

        print("Creating model...", end="")
        i = self.finger.create_model()
        if i == adafruit_fingerprint.OK:
            print("Created")
        else:
            if i == adafruit_fingerprint.ENROLLMISMATCH:
                print("Prints did not match")
            else:
                print("Other error")
            return False

        print("Storing model #%d..." % location, end="")
        i = self.finger.store_model(location)
        if i == adafruit_fingerprint.OK:
            print("Stored")
        else:
            if i == adafruit_fingerprint.BADLOCATION:
                print("Bad storage location")
            elif i == adafruit_fingerprint.FLASHERR:
                print("Flash storage error")
            else:
                print("Other error")
            return False

        return True

    def save_fingerprint_image(self, filename):
        while self.finger.get_image():
            pass
        from PIL import Image

        img = Image.new("L", (256, 288), "white")
        pixeldata = img.load()
        mask = 0b00001111
        result = self.finger.get_fpdata(sensorbuffer="image")
        x = 0
        y = 0
        for i in range(len(result)):
            pixeldata[x, y] = (int(result[i]) >> 4) * 17
            x += 1
            pixeldata[x, y] = (int(result[i]) & mask) * 17
            if x == 255:
                x = 0
                y += 1
            else:
                x += 1

        datasets_path = f"datasets/{filename}"  # Path to the datasets directory
        if not img.save(datasets_path):
            return True
        return False

    def get_num(self, max_number):
        i = -1
        while (i > max_number - 1) or (i < 0):
            try:
                i = int(input("Enter ID # from 0-{}: ".format(max_number - 1)))
                if i in self.id_name_map:
                    print(f"Name for ID #{i}: {self.id_name_map[i]}")
                else:
                    print(f"No name registered for ID #{i}")
                    register_name = input("Enter a name for ID #{}: ".format(i))
                    self.id_name_map[i] = register_name
            except ValueError:
                pass
        return i

    def predictFinger(self):
        finger_detected = False
        while True:
            print("SELAMAT DATANG DI SISTEM KADITA-CV FINGERPRINT")
            print("----------------")
            print("e) Training Fingerprint")
            print("f) Deteksi dengan Fingerprint")
            print("d) Hapus User pada Fingerprint")
            print("s) Simpan Gambar Fingerprint")
            # print("r) Reset Library")
            print("q) Continue to Program Face Recognition")
            print("x) Exit Fingerprint Recognition Program")
            c = input("> ")

            if c == "e":
                self.enroll_finger(self.get_num(self.finger.library_size))
            # elif c == "f":
            #     if self.get_fingerprint():
            #         detected_name = self.finger.finger_id
            #         print("Detected #", self.finger.finger_id, self.id_name_map[detected_name], "with confidence",
            #               self.finger.confidence)
            #         finger_detected = True
            #     else:
            #         print("Finger not found")
            #         finger_detected = False
            elif c == "f":
                if self.get_fingerprint():
                    detected_id = self.finger.finger_id  # Get the detected fingerprint's ID
                    if detected_id in self.id_name_map:
                        detected_name = self.id_name_map[detected_id]  # Get the name associated with the ID
                    else:
                        detected_name = "Name not register"  # Use "Unknown" as the name if the ID is not registered

                    print("Detected #", detected_id, detected_name, "with confidence", self.finger.confidence)
                    finger_detected = True
                else:
                    print("Finger not found")
                    finger_detected = False
            elif c == "d":
                location = self.get_num(self.finger.library_size)
                if self.finger.delete_model(location) == adafruit_fingerprint.OK:
                    if location in self.id_name_map:
                        deleted_name = self.id_name_map[location]
                        del self.id_name_map[location]
                        print(f"Deleted ID #{location} and Name ID:", deleted_name)
                    print("Deleted!")
                else:
                    print("Failed to delete")
            elif c == "s":
                if self.save_fingerprint_image("fingerprint.png"):
                    print("Fingerprint image saved")
                else:
                    print("Failed to save fingerprint image")
            elif c == "r":
                if self.finger.empty_library() == adafruit_fingerprint.OK:
                    print("Library empty!")
                else:
                    print("Failed to empty library")
            elif c == "q":
                print("Exiting fingerprint program")
                break
            elif c == "x":
                raise SystemExit
            else:
                print("Invalid option. Please select a valid option.")

        return finger_detected


class FaceRecognitionTraining:
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.detector = cv2.CascadeClassifier(
            "assets/cascades/data/haarcascades/haarcascade_frontalface_default.xml")
        self.is_train = False

    def getImagesWithLabels(self, path):
        imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
        faceSamples = []
        Ids = []
        for imagePath in imagePaths:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces = self.detector.detectMultiScale(imageNp)
            for (x, y, w, h) in faces:
                faceSamples.append(imageNp[y:y + h, x:x + w])
                Ids.append(Id)
        return faceSamples, Ids

    def process(self):
        print(f"[INFO] Started Training")
        faces, Ids = self.getImagesWithLabels('assets/cascades/datasets')
        self.recognizer.train(faces, np.array(Ids))
        self.recognizer.save('assets/cascades/training/training.xml')
        print(f"[INFO] Training Finished")


class FaceRecognition:
    def __init__(self):
        self.gray = None
        self.frame = None
        self.temp_names = None
        self.frame_detect = None
        self.detect = cv2.CascadeClassifier(
            "assets/cascades/data/haarcascades/haarcascade_frontalface_default.xml")

        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("assets/cascades/training/training.xml")
        self.name_to_save = ""

        self.read_name = []
        self.names = ""
        with open("assets/cascades/names/name.names") as f:
            self.names = f.readlines()

        self.read_name = [x.strip() for x in self.names]
        self.result_name = [x.split(" ") for x in self.read_name]

        self.coordinate = [0, 0, 0, 0]
        self.capture_save = False
        self.max_samples = 0
        self.stop = False
        self.u_time = 0
        self.name = ""
        self.index = 0
        self.id = 0

    def setMaxNumSamples(self, samples=500):
        self.max_samples = samples

    def showUser(self, name_path="assets/cascades/names/name.names"):
        with open(name_path, 'r') as file:
            lines = file.readlines()
        id_user = []
        name_user = []
        for line in lines:
            data = line.strip().split(' ')
            id_user.append(data[0])
            name_user.append(' '.join(data[1:]))
        df = pd.DataFrame({'ID': id_user, 'Nama': name_user})
        print(df)

    def deleteAllUser(self, path="assets/cascades/datasets", name_path="assets/cascades/names/name.names"):
        extension = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        image_list = []
        for ext in extension:
            image_list.extend(glob.glob(os.path.join(path, f"*{ext}")))
        for image in image_list:
            try:
                os.remove(image)
                print(f"[INFO] Image Deleted {image}")
            except Exception as e:
                print(f"[INFO] Failed to Delete {image}: {e}")
        with open(name_path, 'w') as file:
            file.write('')
        print(f"[INFO] Success Delete All User")

    def deleteUser(self, user_number=-1, path="assets/cascades/datasets", name_path="assets/cascades/names/name.names"):
        extension = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
        pattern = os.path.join(path, f"user.{user_number}.*")
        image_list = []
        for ext in extension:
            image_list.extend(glob.glob(f"{pattern}{ext}"))
        for image in image_list:
            try:
                os.remove(image)
                print(f"[INFO] Image Deleted {image}")
            except Exception as e:
                print(f"[INFO] Failed to Delete {image}: {e}")
        with open(name_path, 'r') as file:
            lines = file.readlines()
        lines = [line for line in lines if not line.startswith(f"{user_number} ")]
        with open(name_path, 'w') as file:
            file.writelines(lines)
        print(f"[INFO] Success Delete User: {user_number}")

    def predict(self, frame):
        try:
            self.gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame_detect = self.detect.detectMultiScale(self.gray, 1.3, 5)
            if self.capture_save and len(self.frame_detect) > 0:
                self.index += 1
                print(f"[INFO] User {self.id} Image Saved {self.index}")
            values = []
            for (x, y, w, h) in self.frame_detect:
                id, conf = self.recognizer.predict(self.gray[y:y + h, x:x + w])
                for i in range(len(self.result_name)):
                    for j in range(len(self.result_name[i])):
                        if self.result_name[i][0] == str(id):
                            # self.temp_names = f"{self.result_name[i][0]}|{self.result_name[i][1]}"
                            self.temp_names = f"{self.result_name[i][1]}"
                if self.temp_names is not None:
                    temp = {
                        "class": self.temp_names,
                        "confidence": conf / 100,
                        "x": x,
                        "y": y,
                        "width": w,
                        "height": h,
                        "center": [(x + w) // 2, (y + h) // 2],
                        "color": np.array([255.12, 10.22, 20.3])
                    }
                    values.append(temp)
                if self.capture_save:
                    cv2.imwrite("assets/cascades/datasets/user." +
                                str(self.id) + "." + str(self.index) + ".jpg",
                                self.gray[y:y + h, x: x + w])
                    self.name_to_save = f"{str(self.id)} {self.name}\n"
                    if not self.index > (self.max_samples - 1):
                        self.stop = False
                    else:
                        self.stop = True
                        with open("assets/cascades/names/name.names", "a+") as f:
                            f.write(self.name_to_save)
            return values
        except TypeError as e:
            print(e)

    def isStop(self):
        return self.stop

    def isTraining(self):
        return self.capture_save

    def getCoordinate(self):
        return self.coordinate

    def Capture(self, state=False):
        self.capture_save = state
        if state:
            while True:
                try:
                    self.id = int(input("[INFO] Input your ID's   : "))
                    with open("assets/cascades/names/name.names", 'r') as file:
                        lines = file.readlines()
                    for line in lines:
                        if line.startswith(f"{self.id} "):
                            raise Exception()
                except ValueError as e:
                    print(f"\033[91m[ERROR] ID's must be an Number \n{e}.\033[0m")
                    continue
                except Exception as e:
                    print(f"\033[91m[ERROR] ID User '{self.id}' is Used, Pls Use Other ID.\033[0m")
                    continue
                try:
                    self.name = str(input("[INFO] Input your Name's : "))
                    if self.isContainsNum(self.name) or self.name == "":
                        raise ValueError()
                    break
                except ValueError as e:
                    print(f"\033[91m[ERROR] Name's must be an Character.\033[0m")
            print("\033[92m[INFO] Input ID's Success..\033[0m")
            print("[INFO] Face the camera")
            count = 5
            while True:
                print(f"[INFO] Face the camera [ {count} ]")
                if count <= 1:
                    break
                else:
                    count -= 1
                time.sleep(1)

    def getIDs(self):
        return self.id

    def isContainsNum(self, str_in):
        return any(char.isdigit() for char in str_in)


class HOGDescriptor(ImgRex):  # 5
    def __init__(self):
        self.classes = None
        self.colors = None
        self.model = None
        self.count = 0

    def load(self, names="assets/class/coco.txt", weight="assets/models/coco-yolo5.pt"):
        with open(names, "r") as name:
            self.classes = name.read().split("\n")
        self.colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        # print(f"self.colors[0] = {self.colors[0]}, type = {type(self.colors)}")
        # self.model = torch.hub.load('ultralytics/yolov5', 'custom', weight, force_reload=True)
        # self.model = torch.hub.load('yolov5', 'custom', weight, source='local')

        count = 0
        success = False
        max_count = 100
        while not success:
            print(f"[INFO] connecting {count} ...")
            try:
                self.model = torch.hub.load(
                    'ultralytics/yolov5', 'custom', weight)
                success = True
            except urllib.error.URLError as e:
                print(f"[ERROR] {e}")
                time.sleep(10.0)
                count += 1
        if not success:
            print(f"[ERROR] Connection not stable error code: {max_count}!!")

    def predict(self, frame):
        width = frame.shape[1]
        height = frame.shape[0]

        values = []
        results = self.model(frame)
        pred = results.pred[0]
        boxes_t = pred[:, :4].cpu().numpy()
        labels_t = pred[:, -1].cpu().numpy()
        confidences_t = pred[:, 4].cpu().numpy()

        boxes = []
        center = []
        class_ids = []
        confidences = []
        try:
            # frame = np.squeeze(results.render())
            for box, label, confidence in zip(boxes_t, labels_t, confidences_t):
                if confidence > 0.1:
                    x1, y1, x2, y2 = box
                    boxes.append([int(x1), int(y1), int(
                        x2) - int(x1), int(y2) - int(y1)])
                    center.append([int((x1 + x2) / 2), int((y1 + y2) / 2)])
                    confidences.append(round(confidence, 2))
                    class_ids.append(int(label))

            for i in range(len(boxes)):
                x, y, w, h = boxes[i]
                temp = {
                    "class": str(self.classes[class_ids[i]]),
                    "confidence": confidences[i],
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "center": center[i],
                    "color": self.colors[class_ids[i]]
                }
                values.append(temp)
        except TypeError:
            pass

        return values
