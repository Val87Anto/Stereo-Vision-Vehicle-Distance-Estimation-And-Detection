import zmq
import time
import numpy as np
from collections import deque
import serial
import threading
import matplotlib.pyplot as plt

# ======================================
# PARAMETERS
# ======================================
STEREO_GATE = 1.5      # meters (stereo validation threshold)
ULTRA_MAX   = 5.0      # ultrasonic max valid range
SERIAL_PORT = "COM5"   # Change to your ESP32 port
BAUD_RATE   = 115200

# ======================================
# SERIAL SETUP FOR ULTRASONIC
# ======================================
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Wait for ESP32 to initialize

latest_ultra_value = None
serial_lock = threading.Lock()

def read_ultrasonic():
    """Read ultrasonic data from serial port"""
    global latest_ultra_value
    
    while True:
        try:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').strip()
                if line and line[0].isdigit():
                    try:
                        distance = float(line)
                        with serial_lock:
                            latest_ultra_value = distance
                    except ValueError:
                        pass
        except Exception as e:
            print(f"[SERIAL] Error: {e}")
            time.sleep(0.1)

# Start serial reading thread
serial_thread = threading.Thread(target=read_ultrasonic, daemon=True)
serial_thread.start()

# ======================================
# ZMQ SETUP (FOR STEREO VISION)
# ======================================
context = zmq.Context()
stereo_socket = context.socket(zmq.PULL)
stereo_socket.bind("tcp://*:7000")

poller = zmq.Poller()
poller.register(stereo_socket, zmq.POLLIN)

print("[SFV] Online — waiting for stereo + ultrasonic data...")

# ======================================
# ONLINE VARIANCE (Noise Estimation)
# ======================================
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

# ======================================
# KALMAN FILTER (Distance + Velocity)
# ======================================
class KalmanDV:
    def __init__(self, dt=0.05):
        self.x = np.zeros((2,1))   # [distance, velocity]
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

# ======================================
# INITIALIZATION
# ======================================
ultra_var_est = OnlineVariance()
kf = KalmanDV()

latest_stereo = None

# ======================================
# GRAPH SETUP
# ======================================
plt.ion()
fig, ax = plt.subplots()

time_buf   = deque(maxlen=200)
stereo_buf = deque(maxlen=200)
ultra_buf  = deque(maxlen=200)
kalman_buf = deque(maxlen=200)

start_time = time.time()

# ======================================
# MAIN LOOP
# ======================================
while True:
    socks = dict(poller.poll(timeout=100))

    if stereo_socket in socks:
        latest_stereo = stereo_socket.recv_json()

    if latest_stereo is None:
        continue

    stereo_d   = latest_stereo["stereo"]["distance"]
    stereo_var = latest_stereo["stereo"]["variance"]

    with serial_lock:
        ultra_d = latest_ultra_value if latest_ultra_value is not None else ULTRA_MAX

    ultra_var_est.update(ultra_d)
    ultra_var = ultra_var_est.variance()

    # -------------------------------
    # KALMAN PREDICT (LO: State Estimation)
    # -------------------------------
    kf.predict()
    pred_dist, _ = kf.state()

    # -------------------------------
    # SENSOR VALIDATION (LO: Sensor Fusion)
    # -------------------------------
    use_ultra = False

    if not np.isfinite(stereo_d):
        use_ultra = True
    elif abs(stereo_d - pred_dist) > STEREO_GATE:
        use_ultra = True

    if use_ultra and ultra_d <= ULTRA_MAX:
        z = ultra_d
        R = ultra_var
        sensor = "ULTRASONIC"
    else:
        z = stereo_d
        R = stereo_var
        sensor = "STEREO"

    # -------------------------------
    # KALMAN UPDATE (Weighting via Noise R)
    # -------------------------------
    kf.update(z, R)
    dist, vel = kf.state()

    # -------------------------------
    # PRINT STATUS
    # -------------------------------
    print("="*60)
    print(f"Sensor used : {sensor}")
    print(f"Stereo      : {stereo_d:.2f} m")
    print(f"Ultrasonic  : {ultra_d:.2f} m")
    print(f"Kalman dist : {dist:.2f} m | vel={vel:.2f} m/s")

    # -------------------------------
    # GRAPH UPDATE
    # -------------------------------
    t = time.time() - start_time

    time_buf.append(t)
    stereo_buf.append(stereo_d)
    ultra_buf.append(ultra_d)
    kalman_buf.append(dist)

    ax.clear()
    ax.plot(time_buf, stereo_buf, label="Stereo Distance")
    ax.plot(time_buf, ultra_buf, label="Ultrasonic Distance")
    ax.plot(time_buf, kalman_buf, label="Kalman Fused Distance")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Sensor Fusion using Kalman Filter")
    ax.legend()
    ax.grid(True)

    plt.pause(0.01)
# =====================================================