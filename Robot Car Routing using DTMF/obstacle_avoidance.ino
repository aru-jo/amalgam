#include <NewPing.h>

//Initialise Ultrasound Module
#define TRIGGER_PIN_1  2  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_1     A0  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define TRIGGER_PIN_2  3  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_2     A1  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define TRIGGER_PIN_3  4  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_3     A2  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define TRIGGER_PIN_4  A4  // Arduino pin tied to trigger pin on the ultrasonic sensor.
#define ECHO_PIN_4     A3  // Arduino pin tied to echo pin on the ultrasonic sensor.
#define MAX_DISTANCE 200 // Maximum distance we want to ping for (in centimeters). Maximum sensor distance is rated at 400-500cm.

NewPing sonar_FRONT(TRIGGER_PIN_1, ECHO_PIN_1, MAX_DISTANCE); // NewPing setup of pins and maximum distance.
NewPing sonar_BACK(TRIGGER_PIN_2, ECHO_PIN_2, MAX_DISTANCE); // NewPing setup of pins and maximum distance.
NewPing sonar_SIDE_FRONT(TRIGGER_PIN_3, ECHO_PIN_3, MAX_DISTANCE); // NewPing setup of pins and maximum distance.
NewPing sonar_SIDE_BACK(TRIGGER_PIN_4, ECHO_PIN_4, MAX_DISTANCE); // NewPing setup of pins and maximum distance.

int d3,d2,d1,d0,d6;
int SIDE_FRONT_SENSOR,SIDE_BACK_SENSOR,FRONT_SENSOR,BACK_SENSOR;
bool flag_1;
int flag=0;

void forward(float V_MAX, float V_MIN)
{ 
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,V_MAX);
  analogWrite(6,V_MIN);
  //left motor
  analogWrite(9,V_MAX);
  analogWrite(10,V_MIN);
}

void back(float V_MAX, float V_MIN)
{
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,V_MIN);
  analogWrite(6,V_MAX);
  //left motor
  analogWrite(9,V_MIN);
  analogWrite(10,V_MAX);
}

void hard_right(float V_MAX, float V_MIN)
{ 
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,V_MAX);
  analogWrite(6,V_MIN);
  //left motor
  analogWrite(9,V_MIN);
  analogWrite(10,V_MAX);
}

void hard_left(float V_MAX, float V_MIN)
{ 
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,V_MIN);
  analogWrite(6,V_MAX);
  //left motor
  analogWrite(9,V_MAX);
  analogWrite(10,V_MIN);
}

void soft_right(float V_MAX, float V_MIN)
{ 
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,V_MAX);
  analogWrite(6,V_MIN);
  //left motor
  analogWrite(9,0);
  analogWrite(10,0);
}

void soft_left (float V_MAX, float V_MIN)
{ 
  V_MAX = map(V_MAX,0,5,0,255);
  V_MIN = map(V_MIN,0,5,0,255);
  //right motor
  analogWrite(5,0);
  analogWrite(6,0);
  //left motor
  analogWrite(9,V_MAX);
  analogWrite(10,V_MIN);
}

void stop_motor()
{ 
  //right motor
  analogWrite(5,0);
  analogWrite(6,0);
  //left motor
  analogWrite(9,0);
  analogWrite(10,0);
}

/*void get_input()
{
   SIDE_FRONT_SENSOR = sonar_SIDE_FRONT.ping_cm();
   SIDE_BACK_SENSOR = sonar_SIDE_BACK.ping_cm();
   FRONT_SENSOR = sonar_FRONT.ping_cm();
   BACK_SENSOR = sonar_BACK.ping_cm();
  
   Serial.print(FRONT_SENSOR);
   Serial.print("\t");
   Serial.print(BACK_SENSOR);
   Serial.print("\t");
   Serial.print(SIDE_FRONT_SENSOR);
   Serial.print("\t");
   Serial.println(SIDE_BACK_SENSOR);
   
}
*/

void get_dtmf_input()
{  
 d3=digitalRead(7);
 d2=digitalRead(12);
 d1=digitalRead(11);
 d0=digitalRead(8); 
 d6=digitalRead(A5);
}

/*void coll_detect()
{ 
 get_input();
 
  while(FRONT_SENSOR<=5)
  {
    stop_motor();
  }
  delay(2000);
 
  return;
  
} */
void setup()
{
 Serial.begin(9600);
 
  //Initialise Motor connections(PWM)
 pinMode(5,OUTPUT);
 pinMode(6,OUTPUT);
 pinMode(9,OUTPUT);
 pinMode(10,OUTPUT); 
 
 //Initialise DTMF
 pinMode(7,INPUT);//D3
 pinMode(12,INPUT);//D2
 pinMode(11,INPUT);//D1
 pinMode(8,INPUT);//D0
 pinMode(A5,INPUT);
 
 //Initialise Buzzer
 pinMode(4,OUTPUT);

}

void loop()
{
 get_dtmf_input();
if (d3==0&&d2==0&&d1==1&&d0==0)//2
 {
  forward(5,0);
 }
 if (d3==0&&d2==1&&d1==0&&d0==0&&d6==1)//4
 {
  hard_left(5,1);
  delay(1000);
  stop_motor();
  }
 if (d3==0&&d2==1&&d1==1&&d0==0&&d6==1)//6
 {
  hard_right(5,1);
  delay(1000);
  stop_motor();
 }
 if (d3==1&&d2==0&&d1==0&&d0==0)//8
 {
  back(5,0);
 }
 if (d3==0&&d2==1&&d1==0&&d0==1)//5
 {
  stop_motor();
 }
}

