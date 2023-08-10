#include <Wire.h>               // Library komunikasi I2C
#include <LiquidCrystal_I2C.h>  // Library modul I2C LCD
#include <EEPROM.h>             // Library untuk TDS
#include "GravityTDS.h"         // Library untuk TDS
#include <math.h>               // Library untuk JST

LiquidCrystal_I2C lcd = LiquidCrystal_I2C(0x27, 20, 4);  // Setup LCD I2C

GravityTDS gravityTds;                                 // Setup TDS
#define TDSSensorPin A1                                // Pin TDS
float temperature = 26.6, TDSValue = 0, TDSValueTemp;  // Varibel suhu dan TDS

int pHSensorPin = A0;                             // Pin pH
float pHValue = 0, pHAnalog, pHTegangan, pHTemp;  // Variabel pH, perbedaan pH, nilai analog dan nilai tegangan
int buffer_arr[10], temp;

byte pHUp = 7;      // Pin pompa 1
byte pHDown = 6;    // Pin pompa 2
byte nutrisiA = 5;  // Pin pompa 3
byte nutrisiB = 4;  // Pin pompa 4

float pHLowerThreshold = 6.0;   // Ambang batas bawah pH
float pHUpperThreshold = 7.0;   // Ambang batas atas pH
float TDSLowerThreshold = 560;  // Ambang batas bawah TDS
float TDSUpperThreshold = 840;  // Ambang batas atas TDS

float x1[2] = { pHValue, TDSValue };
float y1[3];

float x1_step1_xoffset[2] = { 4.03, 353 };
float x1_step1_gain[2] = { 0.409836065573771, 0.00412371134020619 };
float x1_step1_ymin = -1;

float b1[10] = { 1.0611451878403826, 1.2292694114405898, 0.61880742064123029, -1.9323893079441505, 2.1419125800123031, -2.5614059873197061, -1.4238327014124188, -1.4282362891286995, 1.6224871740249383, 1.9461525173567351 }; // Nilai Bias Layer 1 - 2
float IW1_1[10][2] = { // Nilai Bobot Layer 1 - 2
  { -1.5797815589156827, -0.0086407960231580823 },
  { 4.4177424602067017, 0.010176133571616678 },
  { 0.029081029354116883, 1.0296110679431814 },
  { -8.8412297702364278, 0.13827244294430893 },
  { -7.8305587277910789, -0.10141310892748259 },
  { -11.369453225554631, 0.25140136995353329 },
  { 0.024093313277825788, 3.7659808352851791 },
  { 5.9681281274232125, 0.060837097504298603 },
  { -0.032445520706524372, -3.7259895754755719 },
  { 2.181064076032901, 0.001222389470232013 }
};

float b2[3] = { 0.39365584754290317, 0.44137658976381822, 0.36517775859020707 }; // Nilai Bias Layer 2-  3
float LW2_1[3][10] = { // Nilai Bobot Layer 2 - 3
  { 0.0062480514440808954, -0.75493610593705829, 0.0089335648594362875, -1.5695902366938788, 0.13298854629289406, 1.2293653799010049, 0.001094905697645714, 0.19540125672953584, 0.0033766496300679277, -1.0468561116295059 },
  { -1.6947005941989961, -0.053804525372577405, 0.010653515127460825, 0.19306526714209141, -0.79662593331702802, -0.13729466462968193, -0.0094566507253065982, -0.91084659753842356, -0.0042745956064945845, -0.055393478231561384 },
  { -0.0055414364322180377, 0.019451328963432012, -1.3922155499377928, 0.031306651381369899, 0.028882408670875513, -0.011214197503107003, -1.1549539155989028, 0.0278263445373217, -1.0783400546822872, 0.0070957532550043568 }
};

float y1_step1_ymin = -1;
float y1_step1_gain[3] = { 0.000322684737011939, 0.000319744204636291, 0.000187318535169055 };
float y1_step1_xoffset[3] = { 0, 0, 0 };

// Helper Functions
float mapminmax_apply(float x, float settings_gain, float settings_xoffset, float settings_ymin) {
  float y = (x - settings_xoffset) * settings_gain + settings_ymin;
  return y;
}

float tansig_apply(float n) {
  float a = 2 / (1 + exp(-2 * n)) - 1;
  return a;
}

float mapminmax_reverse(float y, float settings_gain, float settings_xoffset, float settings_ymin) {
  float x = (y - settings_ymin) / settings_gain + settings_xoffset;
  return x;
}

// Neural Network Function
void myNeuralNetworkFunction(float x1[2], float y1[3]) {
  // Input 1
  float xp1[2] = {
    mapminmax_apply(x1[0], x1_step1_gain[0], x1_step1_xoffset[0], x1_step1_ymin),
    mapminmax_apply(x1[1], x1_step1_gain[1], x1_step1_xoffset[1], x1_step1_ymin)
  };

  // Layer 1
  float a1[10];
  for (int j = 0; j < 10; j++) {
    float sum = b1[j];
    for (int k = 0; k < 2; k++) {
      sum += IW1_1[j][k] * xp1[k];
    }
    a1[j] = tansig_apply(sum);
  }

  // Layer 2
  float a2[3];
  for (int j = 0; j < 3; j++) {
    float sum = b2[j];
    for (int k = 0; k < 10; k++) {
      sum += LW2_1[j][k] * a1[k];
    }
    a2[j] = sum;
  }

  // Output 1
  y1[0] = mapminmax_reverse(a2[0], y1_step1_gain[0], y1_step1_xoffset[0], y1_step1_ymin);
  y1[1] = mapminmax_reverse(a2[1], y1_step1_gain[1], y1_step1_xoffset[1], y1_step1_ymin);
  y1[2] = mapminmax_reverse(a2[2], y1_step1_gain[2], y1_step1_xoffset[2], y1_step1_ymin);
}

void readpH() {
  for (int i = 0; i < 10; i++) {
    buffer_arr[i] = analogRead(pHSensorPin);
    delay(30);
  }
  for (int i = 0; i < 9; i++) {
    for (int j = i + 1; j < 10; j++) {
      if (buffer_arr[i] > buffer_arr[j]) {
        temp = buffer_arr[i];
        buffer_arr[i] = buffer_arr[j];
        buffer_arr[j] = temp;
      }
    }
  }
  pHAnalog = 0;
  for (int i = 2; i < 8; i++) {
    pHAnalog += buffer_arr[i];
  }
  float pHTegangan = (float)pHAnalog * 5 / 1024 / 6;
  pHValue = -4.56 * pHTegangan + 21.34;
}

void readTDS() {
  gravityTds.setTemperature(temperature);  // Mengatur suhu
  gravityTds.update();                     // Update TDS
  TDSValue = gravityTds.getTdsValue();     // Membaca nilai TDS
}

void fixpHUp() {
  digitalWrite(pHUp, LOW);  // Mengaktifkan pompa pH up
  delay(y1[0]);
  digitalWrite(pHUp, HIGH);  // Mematikan pompa pH up

  lcd.setCursor(0, 2);  //Mengatur titik kursor LCD
  lcd.print("pH Up (ms): ");
  lcd.setCursor(13, 2);  //Mengatur titik kursor LCD
  lcd.print(y1[0], 0);
}

void fixpHDown() {
  digitalWrite(pHDown, LOW);  // Mengaktifkan pompa pH Down
  delay(y1[1]);
  digitalWrite(pHDown, HIGH);  // Mematikan pompa pH Down

  lcd.setCursor(0, 2);  //Mengatur titik kursor LCD
  lcd.print("pH Down (ms): ");
  lcd.setCursor(14, 2);  //Mengatur titik kursor LCD
  lcd.print(y1[1], 0);
}

void fixTDS() {
  digitalWrite(nutrisiA, LOW);  // Mengaktifkan pompa nutrisi
  delay(y1[2]);
  digitalWrite(nutrisiA, HIGH);  // Mengaktifkan pompa nutrisi

  lcd.setCursor(0, 3);  //Mengatur titik kursor LCD
  lcd.print("Nutrisi (ms): ");
  lcd.setCursor(14, 3);  //Mengatur titik kursor LCD
  lcd.print(y1[2], 0);
}

void setup() {
  lcd.begin();  // Inisialisasi LCD
  Serial.begin(9600);

  gravityTds.setPin(TDSSensorPin);  // Mengatur pin TDS
  gravityTds.setAref(5.0);          // Nilai Referensi Tegangan TDS
  gravityTds.setAdcRange(1024);     // Nilai bit ADC untuk 10 bit
  gravityTds.begin();               // Inisialisasi TDS

  pinMode(pHSensorPin, INPUT);  //Inisialisasi pH

  pinMode(pHUp, OUTPUT);      // Mengatur pin pompa 1
  pinMode(pHDown, OUTPUT);    // Mengatur pin pompa 1
  pinMode(nutrisiA, OUTPUT);  // Mengatur pin pompa 1
  pinMode(nutrisiB, OUTPUT);  // Mengatur pin pompa 1

  digitalWrite(pHUp, HIGH);
  digitalWrite(pHDown, HIGH);
  digitalWrite(nutrisiA, HIGH);
  digitalWrite(nutrisiB, HIGH);
}

void loop() {
  readpH();
  readTDS();

  lcd.clear();            // Membersihkan tampilan LCD
  lcd.setCursor(0, 0);    // Mengatur titik kursor LCD
  lcd.print("pH: ");      // Menampilkan output
  lcd.setCursor(4, 0);    // Mengatur titik kursor LCD
  lcd.print(pHValue, 2);  // Menampilkan output

  lcd.setCursor(0, 1);     //Mengatur titik kursor LCD
  lcd.print("TDS: ");      // Menampilkan output
  lcd.setCursor(5, 1);     // Mengatur titik kursor LCD
  lcd.print(TDSValue, 0);  // Menampilkan output
  lcd.print("ppm");        // Menampilkan output

  // Initialize inputs
  x1[0] = pHValue;
  x1[1] = TDSValue;

  lcd.setCursor(6, 2);
  lcd.print("-Prediksi-");  // Menampilkan output

  if (pHValue < pHLowerThreshold && TDSValue < TDSLowerThreshold) {
    myNeuralNetworkFunction(x1, y1);
    fixpHUp();
    fixTDS();
  } else if (pHValue > pHUpperThreshold && TDSValue < TDSLowerThreshold) {
    myNeuralNetworkFunction(x1, y1);
    fixpHDown();
    fixTDS();
  } else if (pHValue < pHLowerThreshold) {
    myNeuralNetworkFunction(x1, y1);
    fixpHUp();
  } else if (pHValue > pHUpperThreshold) {
    myNeuralNetworkFunction(x1, y1);
    fixpHDown();
  } else if (TDSValue < TDSLowerThreshold) {
    myNeuralNetworkFunction(x1, y1);
    fixTDS();
  }

  delay(1000);
}