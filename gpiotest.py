import RPi.GPIO as GPIO # Import Raspberry Pi GPIO library
import third_face_detection as fd
GPIO.setwarnings(False) # Ignore warning for now
GPIO.setmode(GPIO.BCM) # Use physical pin numbering
GPIO.setup(0, GPIO.IN, pull_up_down=GPIO.PUD_DOWN) # Set pin 10 to be an input pin and set initial value to be pulled low (off)

while True: # Run forever
    if GPIO.input(0) == GPIO.HIGH:
        fd.face_detect()
        break
    else:
        print("Button not pushed!")
