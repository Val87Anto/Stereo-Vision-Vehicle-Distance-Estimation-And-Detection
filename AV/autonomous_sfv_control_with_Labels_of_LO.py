"""
====================================================
PROJECT: Stereo + Ultrasonic Sensor Fusion for AV
COURSES:
1) Sensor Fusion (SFV)
2) Autonomous Vehicle (AV)
====================================================
"""

import zmq
import time
import numpy as np
from collections import deque
import serial
import threading

# ==================================================
# 🔵 SFV LO1: Sensor Fusion Principles & Components
# - Stereo camera (vision-based distance)
# - Ultrasonic sensor (range-based distance)
# ==================================================

STEREO_GATE = 1.5      # Validation threshold (meters)
ULTRA_MAX   = 5.0      # Ultrasonic max range (meters)
SERIAL_PORT = "COM5"
BAUD_RATE   = 115200

# ==================================================
# 🔵 SFV LO1: Sensor Data Acquisition (Ultrasonic)
# ==================================================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

latest_ultra_value = None
serial_lock = threading.Lock()

def read_ultrasonic():
    global latest_ultra_value
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode().strip()
                if line and line[0].isdigit():
                    latest_ultra_value = float(line)
        except:
            pass

threading.Thread(target=read_ultrasonic, daemon=True).start()

# ==================================================
# 🔵 SFV LO1: Stereo Input via ZMQ
# ==================================================
context = zmq.Context()
stereo_socket = context.socket(zmq.PULL)
stereo_socket.bind("tcp://*:7000")

poller = zmq.Poller()
poller.register(stereo_socket, zmq.POLLIN)

print("[SYSTEM] Sensor Fusion Online")

# ==================================================
# 🔵 SFV LO2: Linear / Probabilistic Model Evaluation
# - Online variance estimation
# ==================================================
class OnlineVariance:
    def __init__(self, window=20):
        self.buf = deque(maxlen=window)

    def update(self, v):
        if np.isfinite(v):
            self.buf.append(v)

    def variance(self):
        if len(self.buf) < 5:
            return 1.0
        return float(np.var(self.buf, ddof=1))

# ==================================================
# 🔵 SFV LO3 + LO4:
# - State-space model
# - Kalman Filter for sensor fusion
# ==================================================
class KalmanDV:
    def __init__(self, dt=0.05):
        # State: [distance, velocity]
        self.x = np.zeros((2,1))
        self.P = np.eye(2)

        self.F = np.array([[1, dt],
                           [0, 1]])
        self.H = np.array([[1, 0]])

        self.Q = np.array([[0.05, 0],
                           [0, 0.2]])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, z, R):
        z = np.array([[z]])
        R = np.array([[R]])

        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def state(self):
        return self.x[0,0], self.x[1,0]

ultra_var_est = OnlineVariance()
kf = KalmanDV()

# ==================================================
# 🟢 AV LO1: Sensing → Perception → Control Pipeline
# ==================================================
class AutonomousController:
    def __init__(self):
        self.kp = 0.8
        self.ki = 0.05
        self.kd = 0.5

        self.prev_error = 0
        self.integral = 0

        self.target_speed = 20.0     # km/h
        self.safe_distance = 5.0     # meters

    # ==================================================
    # 🟢 AV LO4: Feedback Control & Motion Decision
    # ==================================================
    def calculate_control(self, speed, distance):
        throttle = 0.0
        brake = 0.0

        if distance < self.safe_distance:
            return 0.0, 100.0  # Emergency brake

        error = self.target_speed - speed
        self.integral += error
        derivative = error - self.prev_error

        output = (self.kp * error +
                  self.ki * self.integral +
                  self.kd * derivative)

        self.prev_error = error

        if output > 0:
            throttle = min(100.0, output)
        else:
            brake = min(100.0, abs(output))

        return throttle, brake

controller = AutonomousController()

# ==================================================
# 🔵 SFV LO2 + LO3:
# Sensor Validation + Fusion Logic
# ==================================================
latest_stereo = None

while True:
    socks = dict(poller.poll(timeout=100))
    if stereo_socket in socks:
        latest_stereo = stereo_socket.recv_json()

    if latest_stereo is None:
        continue

    stereo_d = latest_stereo["stereo"]["distance"]
    stereo_var = latest_stereo["stereo"]["variance"]

    with serial_lock:
        ultra_d = latest_ultra_value if latest_ultra_value else ULTRA_MAX

    ultra_var_est.update(ultra_d)
    ultra_var = ultra_var_est.variance()

    # -------------------------------
    # 🔵 SFV LO4: Kalman Prediction
    # -------------------------------
    kf.predict()
    pred_dist, _ = kf.state()

    # -------------------------------
    # 🔵 SFV LO2: Sensor Selection
    # -------------------------------
    if not np.isfinite(stereo_d) or abs(stereo_d - pred_dist) > STEREO_GATE:
        z = ultra_d
        R = ultra_var
        sensor = "ULTRASONIC"
    else:
        z = stereo_d
        R = stereo_var
        sensor = "STEREO"

    # -------------------------------
    # 🔵 SFV LO4: Kalman Update
    # -------------------------------
    kf.update(z, R)
    fused_dist, fused_vel = kf.state()

    # ==================================================
    # 🟢 AV LO4 + LO5:
    # Using fused perception for control decision
    # ==================================================
    throttle, brake = controller.calculate_control(
        speed=15.0,  # dummy speed
        distance=fused_dist
    )

    print("="*60)
    print(f"Sensor Used   : {sensor}")
    print(f"Stereo Dist   : {stereo_d:.2f} m")
    print(f"Ultrasonic   : {ultra_d:.2f} m")
    print(f"Fused Dist   : {fused_dist:.2f} m")
    print(f"Decision     : Throttle={throttle:.1f}% | Brake={brake:.1f}%")
