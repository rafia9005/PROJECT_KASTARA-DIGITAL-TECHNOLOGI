import argparse
import time
import serial
import adafruit_fingerprint

from modules.KADITAIMAGE import KADITACVVision
from modules.KADITAVISION import FaceRecognition
from modules.KADITAVISION import FaceRecognitionTraining
from modules.KADITAVISION import FingerprintController
from modules.KADITAVISION import KADITAYOLOV5 as Yolo
from modules.KADITAVISION import ImgRex as YoloV8
from utility.data import YAMLDataHandler

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Argument')
    parser.add_argument('--sample', type=bool, default=False,
                        help='Apakah Ingin Mengambil sample \033[93m(True / False)\033[0m')
    parser.add_argument('--num', type=int, default=100, help='Number of Samples \033[93m(0 - 1000)\033[0m')
    parser.add_argument('--delete', type=int, default=None, help='Menghapus User \033[93m(0 - 100)\033[0m')
    parser.add_argument('--del_all', type=bool, default=None,
                        help='Menghapus Semua User \033[93m(True / False)\033[0m \033[91m Hati-hati!! \033[0m')
    parser.add_argument('--show', type=bool, default=None, help='Menampilkan Semua User \033[93m(True / False)')
    args = parser.parse_args()

    print("[INFO] Main Initialize")
    face = FaceRecognition()
    if args.del_all is not None:
        face.deleteAllUser()
        exit()
    if args.show is not None:
        face.showUser()
        exit()
    if args.delete is not None:
        face.deleteUser(args.delete)
        t = FaceRecognitionTraining()
        t.process()
    else:
        fingerprint_controller = FingerprintController()
        # fingerprint_controller.initialize()
        # fingerprint_training = fingerprint_controller.predictFinger()
        face.Capture(args.sample)
        face.setMaxNumSamples(args.num)
        cam = KADITACVVision(isUsingCam=True, addr="data/person/men.mp4")
        yolo = YoloV8()
        yolo.load("assets/models/yolov4-tiny-custom_final.weights", "assets/config/yolov4-tiny-custom.cfg",
                  "assets/class/person.names")
        # yolo.load("assets/class/apd.txt", "assets/models/best-int8.tflite")
        data = YAMLDataHandler("out/person-output-data.yaml")
        try:
            while True:
                try:
                    frame = cam.read(640, True)
                    face_detect = face.predict(frame)
                    # fingerprint_detect = fingerprint_controller.get_fingerprint_detail()
                    detect = []
                    if not face.isTraining():
                        detect = yolo.predict(frame)
                        yolo.draw(frame, detect)
                        yolo.draw(frame, face_detect)
                        face_condition = 1 if face_detect else 0
                        person = any(data['class'] == 'person' for data in detect)
                        condition = 1 if (person and face_condition) else 0
                        if condition == 1:
                            print("SELAMAT ANDA HADIR")
                        else:
                            print("MOHON MAAF ANDA TIDAK HADIR")

                    # finger_condition = 1 if fingerprint_detect else 0
                    face_condition = 1 if face_detect else 0
                    person = any(data['class'] == 'person' for data in detect)
                    condition = 1 if (person and face_condition) else 0
                    data.update("face-finger-voice-condition", str(condition))
                    cam.show(frame, "frame")
                    cam.writeImg(frame, "out/person-output.png")
                    if cam.wait(1) == ord('q') or face.isStop():
                        if face.isTraining():
                            t = FaceRecognitionTraining()
                            t.process()
                        break
                except Exception as err:
                    print(err)
            cam.release()
            cam.destroy()
        except Exception as e:
            print(f"[INFO] {time.time()} Main Initialize Failed: \n{e}")
