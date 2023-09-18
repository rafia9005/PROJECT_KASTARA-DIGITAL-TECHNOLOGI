import cv2
import numpy as np


class SimpleKalmanFilter:
    """
    CARA PENGGUNAAN:
        while True:
        estimate = kf.update_estimate(new_measurement)
        print(f"Estimasi baru: {estimate}")

        # Mendapatkan gain Kalman saat ini
        gain = kf.get_kalman_gain()
        print(f"Gain Kalman saat ini: {gain}")
        
        # Mengubah kesalahan pengukuran
        kf.set_measurement_error(0.5)
        
        # Mengubah kesalahan estimasi
        kf.set_estimate_error(0.2)
        
        # Mengubah noise proses
        kf.set_process_noise(0.05)
        
        # Mendapatkan kesalahan estimasi setelah perubahan
        err_estimate = kf.get_estimate_error()
        print(f"Kesalahan estimasi setelah perubahan: {err_estimate}")
    """

    def __init__(self, mea_e, est_e, q):
        self._err_measure = mea_e
        self._err_estimate = est_e
        self._q = q
        self._current_estimate = 0
        self._last_estimate = 0
        self._kalman_gain = 0

    def update_estimate(self, mea):
        self._kalman_gain = self._err_estimate / (self._err_estimate + self._err_measure)
        self._current_estimate = self._last_estimate + self._kalman_gain * (mea - self._last_estimate)
        self._err_estimate = (1.0 - self._kalman_gain) * self._err_estimate + abs(
            self._last_estimate - self._current_estimate) * self._q
        self._last_estimate = self._current_estimate

        return self._current_estimate

    def set_measurement_error(self, mea_e):
        self._err_measure = mea_e

    def set_estimate_error(self, est_e):
        self._err_estimate = est_e

    def set_process_noise(self, q):
        self._q = q

    def get_kalman_gain(self):
        return self._kalman_gain

    def get_estimate_error(self):
        return self._err_estimate


class KalmanFilter:
    def __init__(self, initial_state, initial_covariance, measurement_noise, process_noise):
        # Inisialisasi matriks sistem dan ketidakpastian pada sensor dan model sistem
        self.kalman = cv2.KalmanFilter(
            len(initial_state), (measurement_noise))
        self.kalman.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, 0], [
            0, 0, 0, 2]], np.float32) * process_noise
        self.kalman.measurementNoiseCov = np.array(
            [[1, 0], [0, 1]], np.float32) * measurement_noise
        self.kalman.errorCovPost = np.array(initial_covariance, np.float32)
        self.kalman.statePost = np.array(initial_state, np.float32)

    def update(self, measurement):  # np.array([cx, cy], np.float32)
        # Perbarui state dan covariance dengan pengukuran terbaru
        self.kalman.correct(measurement)
        prediction = self.kalman.predict()
        return prediction


class KalmanFilter1D(object):
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.x = np.matrix([[0], [0]])
        self.A = np.matrix([[1, dt],
                            [0, 1]])
        self.B = np.matrix([[(dt ** 2) / 2],
                            [dt]])
        self.H = np.matrix([[1, 0]])
        self.u = np.matrix([[u]])
        self.Q = np.matrix([[(dt ** 4) / 4, (dt ** 3) / 2],
                            [(dt ** 3) / 2, dt ** 2]]) * std_acc ** 2
        self.R = std_meas ** 2
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(
            np.dot(np.dot(self.H, self.P), self.H.T) + self.R))
        self.z = np.matrix([[z]])
        self.x = self.x + np.dot(self.K, (self.z - np.dot(self.H, self.x)))
        self.P = self.P - np.dot(np.dot(self.K, self.H), self.P)

    def get(self):
        return self.x[0, 0], self.x[1, 0]


class KalmanFilter2D(object):
    def __init__(self, dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
        """
        KALMAN FILTER 2Dimensi v 3.2
        :param dt: sampling time (time for 1 cycle)
        :param u_x: acceleration in x-direction
        :param u_y: acceleration in y-direction
        :param std_acc: process noise magnitude
        :param x_std_meas: standard deviation of the measurement in x-direction
        :param y_std_meas: standard deviation of the measurement in y-direction
        """
        # Define sampling time
        self.dt = dt
        # Define the  control input variables
        self.u = np.matrix([[u_x], [u_y]])
        # Intial State
        self.x = np.matrix([[0], [0], [0], [0]])
        # Define the State Transition Matrix A
        self.A = np.matrix([[1, 0, self.dt, 0],
                            [0, 1, 0, self.dt],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        # Define the Control Input Matrix B
        self.B = np.matrix([[(self.dt ** 2) / 2, 0],
                            [0, (self.dt ** 2) / 2],
                            [self.dt, 0],
                            [0, self.dt]])
        # Define Measurement Mapping Matrix
        self.H = np.matrix([[1, 0, 0, 0],
                            [0, 1, 0, 0]])

        self.Q = np.matrix([[(self.dt ** 4) / 4, 0, (self.dt ** 3) / 2, 0],
                            [0, (self.dt ** 4) / 4, 0, (self.dt ** 3) / 2],
                            [(self.dt ** 3) / 2, 0, self.dt ** 2, 0],
                            [0, (self.dt ** 3) / 2, 0, self.dt ** 2]]) * std_acc ** 2
        # Initial Measurement Noise Covariance
        self.R = np.matrix([[x_std_meas ** 2, 0],
                            [0, y_std_meas ** 2]])
        # Initial Covariance Matrix
        self.P = np.eye(self.A.shape[1])

    def predict(self):
        # Refer to :Eq.(9) and Eq.(10)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # Update time state
        # x_k =Ax_(k-1) + Bu_(k-1)     Eq.(9)
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        # Calculate error covariance
        # P= A*P*A' + Q               Eq.(10)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[0:2]

    def update(self, z):
        # Refer to :Eq.(11), Eq.(12) and Eq.(13)  in https://machinelearningspace.com/object-tracking-simple-implementation-of-kalman-filter-in-python/?preview_id=1364&preview_nonce=52f6f1262e&preview=true&_thumbnail_id=1795
        # S = H*P*H'+R
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)
        self.x = np.round(
            self.x + np.dot(K, (z - np.dot(self.H, self.x))))  # Eq.(12)
        I = np.eye(self.H.shape[1])
        # Update error covariance matrix
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)
        return self.x[0:2]
