import time
import zmq
import numpy as np
import serial
import threading
from collections import deque
import os
import matplotlib.pyplot as plt

# =====================================================
# PARAMETERS
# =====================================================
STEREO_GATE = 1.5
ULTRA_MAX = 5.0
SERIAL_PORT = "COM5"
BAUD_RATE = 115200
CALIB_FILE = "stereo_params_debug.npz"

# =====================================================
# LOAD STEREO CALIBRATION (OPTIONAL)
# =====================================================
if os.path.exists(CALIB_FILE):
    calib = np.load(CALIB_FILE)
    baseline = calib.get("baseline", None)
    focal_length = calib.get("focal_length", None)
    print("[CALIB] Stereo calibration loaded")
    print(f"[CALIB] Baseline: {baseline}")
    print(f"[CALIB] Focal length: {focal_length}")
else:
    print("[CALIB] No calibration file found, running without calibration")

# =====================================================
# AUTONOMOUS CONTROLLER
# =====================================================
class AutonomousController:
    def __init__(self):
        self.kp = 0.8
        self.ki = 0.05
        self.kd = 0.5

        self.prev_error = 0
        self.integral = 0

        self.target_speed = 20.0
        self.safe_distance = 5.0
        self.simulated_speed = 0.0

    def calculate_control_command(self, speed, distance, emergency):
        throttle = 0.0
        brake = 0.0
        target = self.target_speed

        if 5.0 <= distance < 15.0:
            target = 10.0

        if distance < self.safe_distance or emergency:
            self.integral = 0
            self.prev_error = 0
            return 0.0, 100.0

        error = target - speed
        self.integral += error
        derivative = error - self.prev_error

        output = self.kp*error + self.ki*self.integral + self.kd*derivative
        self.prev_error = error

        if output > 0:
            throttle = min(100, output)
        else:
            brake = min(100, abs(output))

        return throttle, brake

    def update_physics(self, throttle, brake):
        if brake > 50:
            self.simulated_speed -= 5
        elif throttle > 0:
            self.simulated_speed += 2
        else:
            self.simulated_speed -= 0.5

        self.simulated_speed = max(0, self.simulated_speed)

# =====================================================
# SENSOR FUSION
# =====================================================
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

class KalmanDV:
    def __init__(self, dt=0.05):
        self.x = np.zeros((2,1))
        self.P = np.eye(2)
        self.F = np.array([[1, dt],[0, 1]])
        self.H = np.array([[1, 0]])
        self.Q = np.array([[0.05, 0],[0, 0.2]])

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

# =====================================================
# ULTRASONIC SERIAL THREAD
# =====================================================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)

latest_ultra = None
lock = threading.Lock()

def read_ultrasonic():
    global latest_ultra
    while True:
        try:
            if ser.in_waiting:
                line = ser.readline().decode().strip()
                if line and line[0].isdigit():
                    with lock:
                        latest_ultra = float(line)
        except:
            pass

threading.Thread(target=read_ultrasonic, daemon=True).start()

# =====================================================
# ZMQ STEREO INPUT
# =====================================================
context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind("tcp://*:7000")

poller = zmq.Poller()
poller.register(socket, zmq.POLLIN)

# =====================================================
# GRAPH SETUP
# =====================================================
plt.ion()
fig, ax = plt.subplots()
ax.set_title("Sensor Fusion & Vehicle Control")
ax.set_xlabel("Time")
ax.set_ylabel("Value")

t_buf = deque(maxlen=200)
stereo_buf = deque(maxlen=200)
ultra_buf = deque(maxlen=200)
kalman_buf = deque(maxlen=200)
speed_buf = deque(maxlen=200)

line_stereo, = ax.plot([], [], label="Stereo Distance")
line_ultra, = ax.plot([], [], label="Ultrasonic Distance")
line_kalman, = ax.plot([], [], label="Kalman Distance")
line_speed, = ax.plot([], [], label="Vehicle Speed")

ax.legend()

# =====================================================
# INIT
# =====================================================
av = AutonomousController()
kf = KalmanDV()
ultra_var = OnlineVariance()

latest_stereo = None
t = 0

print("[SYSTEM] SFV + AV System Running")

# =====================================================
# MAIN LOOP
# =====================================================
while True:
    socks = dict(poller.poll(timeout=50))
    if socket in socks:
        latest_stereo = socket.recv_json()

    if latest_stereo is None:
        continue

    stereo_d = latest_stereo["stereo"]["distance"]
    stereo_var = latest_stereo["stereo"]["variance"]

    with lock:
        ultra_d = latest_ultra if latest_ultra else ULTRA_MAX

    ultra_var.update(ultra_d)
    uv = ultra_var.variance()

    kf.predict()
    pred, _ = kf.state()

    if not np.isfinite(stereo_d) or abs(stereo_d - pred) > STEREO_GATE:
        z, R, src = ultra_d, uv, "ULTRA"
    else:
        z, R, src = stereo_d, stereo_var, "STEREO"

    kf.update(z, R)
    dist, vel = kf.state()

    throttle, brake = av.calculate_control_command(
        av.simulated_speed, dist, dist < av.safe_distance
    )
    av.update_physics(throttle, brake)

    # ---- STORE DATA ----
    t += 1
    t_buf.append(t)
    stereo_buf.append(stereo_d)
    ultra_buf.append(ultra_d)
    kalman_buf.append(dist)
    speed_buf.append(av.simulated_speed)

    # ---- UPDATE GRAPH ----
    line_stereo.set_data(t_buf, stereo_buf)
    line_ultra.set_data(t_buf, ultra_buf)
    line_kalman.set_data(t_buf, kalman_buf)
    line_speed.set_data(t_buf, speed_buf)

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.001)

    # ---- CONSOLE OUTPUT ----
    print("="*60)
    print(f"Sensor Used : {src}")
    print(f"Distance   : {dist:.2f} m | Velocity: {vel:.2f} m/s")
    print(f"Speed      : {av.simulated_speed:.2f} km/h")
    print(f"Control    : Throttle={throttle:.1f}% | Brake={brake:.1f}%")
