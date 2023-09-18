import os
import cv2
import time


def main():
    start_time = time.time()
    sample = cv2.imread("test-fingerprint/1 (419).jpg")
    best_score = 0
    filename = None
    image = None
    kp1, kp2, mp = None, None, None

    for file in [file for file in os.listdir("test-fingerprint/")][:1000]:
        fingerprint_image = cv2.imread(os.path.join("test-fingerprint", file))
        sift = cv2.SIFT_create()

        keypoints_1, descriptor_1 = sift.detectAndCompute(sample, None)
        keypoints_2, descriptor_2 = sift.detectAndCompute(fingerprint_image, None)

        bf = cv2.BFMatcher(cv2.NORM_L2)
        matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
        match_points = []

        for p, q in matches:
            if p.distance < 0.1 * q.distance:
                match_points.append(p)

        keypoints = min(len(keypoints_1), len(keypoints_2))

        if keypoints > 0 and len(match_points) / keypoints * 100 > best_score:
            best_score = len(match_points) / keypoints * 100
            filename = file
            image = fingerprint_image
            kp1, kp2, mp = keypoints_1, keypoints_2, match_points

    if filename is not None:
        print("BEST MATCH: " + filename)
    else:
        print("Tidak Ditemukan")
    print("HASIL DETEKSI: " + str(best_score))
    print("TIME DELAY: ", time.time() - start_time)

    if sample is not None and image is not None:
        result = cv2.drawMatches(sample, kp1, image, kp2, mp, None)
        result = cv2.resize(result, (result.shape[1] * 2, result.shape[0] * 2))
        cv2.imshow("Hasil", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Gambar sample atau image tidak valid.")


if __name__ == "__main__":
    main()
