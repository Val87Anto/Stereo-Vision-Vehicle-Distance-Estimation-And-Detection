import zmq
import serial
import time

# =========================
# ZMQ SETUP
# =========================
context = zmq.Context()
socket = context.socket(zmq.PUSH)
socket.connect("tcp://127.0.0.1:7001")

# =========================
# SERIAL SETUP
# =========================
ser = serial.Serial("COM4", 115200, timeout=1)

print("[ULTRA] Bridge online")
print("[ULTRA] Waiting for ESP32 data...")

# =========================
# MAIN LOOP
# =========================
while True:
    line = ser.readline().decode(errors="ignore").strip()

    if not line:
        continue

    try:
        d = float(line)

        # Sanity check (HC-SR04 realistic range)
        if 0.15 <= d <= 5.0:
            packet = {
                "timestamp": time.time(),
                "ultrasonic": {
                    "distance": d
                }
            }

            socket.send_json(packet)

            # ✅ THIS IS THE DISTANCE OUTPUT
            print(f"[ULTRA] Distance = {d:.3f} m")

        else:
            print(f"[ULTRA] Out-of-range: {d}")

    except ValueError:
        print(f"[ULTRA] Invalid data: {line}")
