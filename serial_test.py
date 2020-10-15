import time
import serial

output_list = list()
float1 = 1.33333333333
float2 = 2.442535
float3 = 7.369536775

output_list.append(float1)
output_list.append(float2)
output_list.append(float3)
ser = serial.Serial('/dev/ttyAMA0', 9600, timeout = 1)

for values in output_list:
    ser.write(str.encode(str(values) + " "))
    time.sleep(1)
    ser.flush()
