import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import time
print("Environment Ready")

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
    time.sleep(1)
print("reset done")

# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipe.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

current_frame = 0
net = cv2.dnn.readNetFromCaffe("./deploy.prototxt.txt", "./res10_300x300_ssd_iter_140000.caffemodel")
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

depth_list = list()
depth_list_output = list()
depth_intrin = 0
midpoint_1 = 0
midpoint_2 = 0
final_depth = 0
expected= 300
midpoint_array_1 = list()
midpoint_array_2 = list()
try:
	while True:
		# Store next frameset for later processing:
		frames = pipe.wait_for_frames()
		color_frame = frames.get_color_frame()
		depth_frame = frames.get_depth_frame()
		if not depth_frame or not color_frame:
			continue

		print(f"Captured {len(midpoint_array_1)} frames")
		# Convert images to numpy arrays
		depth_image = np.asanyarray(depth_frame.get_data())
		color_image = np.asanyarray(color_frame.get_data())

		#Align colour and depth data together based on XYZ
		align = rs.align(rs.stream.depth)
		frameset = align.process(frames)
		if frameset.size() < 2:
			continue
		depth_frame = frameset.get_depth_frame()
		# color_frame = frameset.get_color_frame()
		depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
		# color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
		depth_image = np.asanyarray(depth_frame.get_data())
		# color_image = np.asanyarray(color_frame.get_data())


		(height, width) = color_image.shape[:2]
		expected = 300
		aspect = width / height
		resized_image = cv2.resize(color_image, (round(expected * aspect), expected))
		crop_start = round(expected * (aspect - 1) / 2)
		blob = cv2.dnn.blobFromImage(cv2.resize(color_image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

		net.setInput(blob, "data")
		detections = net.forward("detection_out")
		print(current_frame)
		if current_frame > 10:

			# loop over the detections
			for i in range(0, detections.shape[2]):
				# extract the confidence
				confidence = detections[0, 0, i, 2]

				if confidence > 0.6:

					# compute the (x, y)-coordinates of the bounding box
					box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
					(startX, startY, endX, endY) = box.astype("int")
					faceROI_color_image = color_image[startY:startY+endY, startX:startX+endX]

					# scale = height / expected
					# xmin_depth = int((startX * expected + crop_start) * scale)
					# ymin_depth = int((startY * expected) * scale)
					# xmax_depth = int((endX * expected + crop_start) * scale)
					# ymax_depth = int((endY * expected) * scale)
					# print(xmin_depth, ymin_depth, xmax_depth, ymax_depth)
					# Crop depth data:
					depth = depth_image[startX:endX, startY:endY].astype(float)
					# print(depth)
					# Get data scale from the device and convert to meters
					depth = depth * depth_scale

					final_depth,_,_,_ = cv2.mean(depth)

					eyes = eyes_cascade.detectMultiScale(faceROI_color_image)
					if(len(eyes) != 2):
						# get XY using midpoint of face
						midpoint_1 = int(startX + (endX/2))
						midpoint_2 = int(startY + (endY/2))

						print("Used midpoint")
					else:
						middle_x_1 = eyes[0][0] + (eyes[0][2]/2)
						middle_y_1 = eyes[0][1] + (eyes[0][3]/2)
						middle_x_2 = eyes[1][0] + (eyes[1][2]/2)
						middle_y_2 = eyes[1][1] + (eyes[1][3]/2)

						midpoint_1 = int((middle_x_1 + middle_x_2) / 2)
						midpoint_2 = int((middle_y_1 + middle_y_2) / 2)
						print("Used Haar")


			other_depth = depth_frame.get_distance(midpoint_2, midpoint_1)
			depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, [midpoint_2, midpoint_1], other_depth) #convert point to meters
			midpoint_array_1.append(depth_point[0])
			midpoint_array_2.append(depth_point[1])
			depth_list_output.append(final_depth)
			print(f"Midpoint of eye is X:{depth_point[0]}, y:{depth_point[1]} and distance to face is {final_depth}")

		if len(midpoint_array_1) >= 5:
			final_depth = np.mean(depth_list_output, axis=0)
			final_midpoint_1 = np.mean(midpoint_array_1, axis=0)
			final_midpoint_2 = np.mean(midpoint_array_2, axis=0)
			print(f"Midpoint of eye (in meters) is at X:{final_midpoint_1}, Y:{final_midpoint_2} and distance to face is {final_depth} meters")
			break

		current_frame += 10

finally:
	pipe.stop()
