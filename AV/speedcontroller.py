import time
import random

class AutonomousController:
    def __init__(self):
        # --- Setting Variabel PID (Untuk Maintain Speed) ---
        self.kp = 0.8  # Power gas
        self.ki = 0.05 # Akumulasi error
        self.kd = 0.5  # Peredam kejut
        
        self.prev_error = 0
        self.integral = 0
        
        # --- Setting Keamanan ---
        self.target_speed = 20.0  # Misal: 20 km/h
        self.safe_distance = 5.0  # Jarak aman minimal (meter)
        
        # Simulasi Fisika Kendaraan (Hanya untuk Dummy Data)
        self.simulated_speed = 0.0

   # BAGIAN 1: DUMMY SENSOR DATA (Simulasi Input dari Sensor)
    def get_dummy_sensor_data(self, cycle_count):
        """
        Mengembalikan data pura-pura seolah-olah dari sensor.
        Skenario:
        - Detik 0-10: Jalan kosong (Normal)
        - Detik 11-15: Ada mobil di depan tapi agak jauh (Slow down)
        - Detik 16-20: Ada penyebrang jalan TIB-TIBA (Emergency Brake)
        """
        
        # 1. Data Kecepatan Aktual (Biasanya dari Wheel Encoder/SLAM)
        # Kita simulasi speed naik turun sedikit biar kayak asli
        current_speed = self.simulated_speed + random.uniform(-0.5, 0.5)
        
        # 2. Data Computer Vision (Jarak Halangan & Lampu Merah)
        obstacle_distance = 100.0 # Default jauh (aman)
        emergency_status = False  # True jika CV deteksi 'STOP SIGN' atau 'RED LIGHT'

        if 0 <= cycle_count <= 10:
            print(f"\n[Skenario]: Jalanan Kosong Lancar")
            obstacle_distance = 50.0 
            
        elif 11 <= cycle_count <= 15:
            print(f"\n[Skenario]: Ada kendaraan di depan (Jarak Menipis)")
            obstacle_distance = 10.0 
            
        elif cycle_count > 15:
            print(f"\n[Skenario]: BAHAYA! Objek sangat dekat / Lampu Merah!")
            obstacle_distance = 2.0  # Sangat dekat!
            emergency_status = True  # Anggap CV melihat tanda STOP
            
        return max(0, current_speed), obstacle_distance, emergency_status
    
    # BAGIAN 2: LOGIKA UTAMA KAMU (Maintain Speed & Brake)
   
    def calculate_control_command(self, current_speed, obstacle_dist, is_emergency):
        throttle = 0.0
        brake = 0.0
        
        current_target_speed = self.target_speed 
        
        # 2. Zona Waspada (Jarak antara 5m - 15m): Kurangi kecepatan setengahnya
        if 5.0 <= obstacle_dist < 15.0:
            print("[INFO] Objek di depan, menurunkan kecepatan...")
            current_target_speed = 10.0 # Turun jadi 10 km/h
            
        # 3. Zona Bahaya (Jarak < 5m) ATAU Emergency Signal: BERHENTI
        if obstacle_dist < self.safe_distance or is_emergency:
            print("!!! EMERGENCY BRAKE ACTIVATED !!!")
            # ... (kode rem emergency sama seperti sebelumnya)
            throttle = 0.0
            brake = 100.0
            self.integral = 0
            self.prev_error = 0
            return throttle, brake # Langsung return biar PID tidak jalan

        # --- LOGIKA PID (Sekarang menggunakan current_target_speed yang dinamis) ---
        # Gunakan 'current_target_speed' (bukan self.target_speed yg fix 20)
        error = current_target_speed - current_speed 
        
        self.integral += error
        derivative = error - self.prev_error
        
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error
        
        if output > 0:
            throttle = min(100.0, output)
            brake = 0.0
        else:
            throttle = 0.0
            brake = min(100.0, abs(output))

        return throttle, brake
        
       

    # Fungsi untuk update fisika dummy (biar speed-nya berubah kalau digas)
    def update_physics_sim(self, throttle, brake):
        if brake > 50:
            self.simulated_speed -= 5.0 # Rem pakem cepat berhenti
        elif throttle > 0:
            self.simulated_speed += 2.0 # Akselerasi
        else:
            self.simulated_speed -= 0.5 # Gesekan jalan (melambat sendiri)
            
        self.simulated_speed = max(0, self.simulated_speed) # Speed gak bisa minus


# MAIN LOOP (Jantung Program)

if __name__ == "__main__":
    av_system = AutonomousController()
    
    # Loop simulasi selama 20 siklus (detik)
    for i in range(21):
        # 1. AMBIL DUMMY DATA
        curr_speed, obs_dist, is_emerg = av_system.get_dummy_sensor_data(i)
        
        # 2. JALANKAN ALGORITMA KAMU
        gas_cmd, brake_cmd = av_system.calculate_control_command(curr_speed, obs_dist, is_emerg)
        
        # 3. UPDATE FISIKA (Hanya simulasi)
        av_system.update_physics_sim(gas_cmd, brake_cmd)
        
        # 4. PRINT HASIL UNTUK LAPORAN
        print(f"Cycle {i} | Speed: {curr_speed:.2f} km/h | Jarak: {obs_dist}m")
        print(f"--> KEPUTUSAN: Gas {gas_cmd:.1f}% | Rem {brake_cmd:.1f}%")
        print("-" * 40)
        
        time.sleep(0.5) # Biar enak dilihat log-nya