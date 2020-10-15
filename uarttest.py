import time
import serial

output_list = list()
final_midpoint_1 = 0.3333
final_midpoint_2 = 0.2222
final_depth = 1.2

output_list.append(final_midpoint_1)
output_list.append(final_midpoint_2)
output_list.append(final_depth)

ser = serial.Serial('/dev/ttyAMA0', 9600, timeout = 1)

for values in output_list:
    ser.write(str.encode(str(values) + " "))
    time.sleep(1)
    ser.flush()
