import cv2
import numpy as np


class KalmanFilter:
    def __init__(self) -> None:
        self.kf = cv2.KalmanFilter(4, 2)  # State: [x, y, dx, dy], Measurement: [x, y]
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], 
                                              [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                             [0, 1, 0, 1],
                                             [0, 0, 1, 0],
                                             [0, 0, 0, 1]], np.float32)
        self.kf.processNoiseCov = np.array([[1, 0, 0, 0],
                                            [0, 1, 0, 0],
                                            [0, 0, 1, 0],
                                            [0, 0, 0, 1]], np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-2  # Example measurement noise
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1  # Initial error estimate    kf = cv2.KalmanFilter(4, 2)


    def predict(self, coordX, coordY):
        measured = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measured)
        predicted = self.kf.predict()
        x, y = int(predicted[0]), int(predicted[1])
        return x, y
    
    def update(self, coordX, coordY):
        # Incorporates a new measurement
        measurement = np.array([[np.float32(coordX)], [np.float32(coordY)]])
        self.kf.correct(measurement)
