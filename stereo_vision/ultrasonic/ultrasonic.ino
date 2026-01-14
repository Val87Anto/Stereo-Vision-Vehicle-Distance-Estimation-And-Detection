#include <Arduino.h>

// ===============================
// Ultrasonic Pin Configuration
// ===============================
#define TRIG_PIN 13
#define ECHO_PIN 12

// ===============================
// Sensor Configuration
// ===============================
#define MAX_DISTANCE_M 5.0
#define SOUND_SPEED 0.000343  // meters per microsecond
#define TIMEOUT_US 35000
#define MIN_REALISTIC_DURATION_US 150

// ===============================
// Function Declarations
// ===============================
void setup();
void loop();
float getUltrasonicDistance();

void setup() {
  Serial.begin(115200);

  pinMode(TRIG_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  digitalWrite(TRIG_PIN, LOW);

  Serial.println("ULTRASONIC_READY");
}

float getUltrasonicDistance() {
  // Trigger pulse
  digitalWrite(TRIG_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIG_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIG_PIN, LOW);

  // Measure echo pulse
  long duration = pulseIn(ECHO_PIN, HIGH, TIMEOUT_US);

  if (duration < MIN_REALISTIC_DURATION_US || duration == 0) {
    return MAX_DISTANCE_M;
  }

  float distance = (duration * SOUND_SPEED) / 2.0;
  if (distance > MAX_DISTANCE_M) distance = MAX_DISTANCE_M;

  return distance;
}

void loop() {
  float distance = getUltrasonicDistance();

  // Output ONLY the float value for easy parsing
  Serial.println(distance, 2);
  delay(50);                    // 20 Hz
}
