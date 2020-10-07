import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import math
import time
import statistics as stats
print("Environment Ready")

ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
    time.sleep(1)
print("reset done")
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eyes_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')


# Setup:
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
profile = pipe.start(cfg)
depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

counter = 0
depth_array = list()
eyes_center_array = list()
midpoint_array_1 = list()
midpoint_array_2 = list()
depth = 0
midpoint = 0

try:
    while True:
        # Wait for a coherent pair of frames: depth and color
        # for x in range(100):
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
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
        color_frame = frameset.get_color_frame()
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # find the human face in the color_image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        # break
        if isinstance(faces, np.ndarray):
            # print(faces)
            for (x, y, w, h) in faces:
                faceROI_color_image = color_image[y:y+h, x:x+w]
                # faceROI_depth_image = depth_image[y:y+h, x:x+w]

                # cols = list(range(x, x+w))
                # # H
                # rows = list(range(y, y+h))
                # for i in rows: #H
                #     for j in cols: #W
                #         depth = (depth_frame.get_distance(j, i)) # W,H
                #         depth_point = rs.rs2_deproject_pixel_to_point(
                #             depth_intrin, [j, i], depth)
                #         counter += 1

                #find center of face
                middle_x = int(x + (w/2))
                middle_y = int(y + (h/2))

                print(middle_x, middle_y)
                depth = depth_image[middle_x, middle_y].astype(float) * depth_scale
                if(depth == 0):
                    break

                eyes = eyes_cascade.detectMultiScale(faceROI_color_image)
                if(len(eyes) != 2):
                    break

                middle_x_1 = int(eyes[0][0] + (eyes[0][2]/2))
                middle_y_1 = int(eyes[0][1] + (eyes[0][3]/2))
                first_eye = (middle_x_1, middle_y_1)
                middle_x_2 = int(eyes[1][0] + (eyes[1][2]/2))
                middle_y_2 = int(eyes[1][1] + (eyes[1][3]/2))
                second_eye = (middle_x_2, middle_y_2)
                # print(eyes[0])
                # for (x2,y2,w2,h2) in eyes:
                #     eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                #     eyes_center_array.append(eye_center)


                midpoint_1 = int((middle_x_1 + middle_x_2) / 2)
                midpoint_2 = int((middle_y_1 + middle_y_2) / 2)
                print(f"Midpoint of eye is {midpoint} and distance to face is {depth}")
                depth_array.append(depth)
                midpoint_array_1.append(midpoint_1)
                midpoint_array_2.append(midpoint_2)
                del eyes_center_array[:]

        if len(depth_array) >= 5:

            final_depth = stats.mean(depth_array)
            final_midpoint_1 = stats.mean(midpoint_array_1)
            final_midpoint_2 = stats.mean(midpoint_array_2)
            print(f"Midpoint of eye is at X:{final_midpoint_1}, Y:{final_midpoint_2} and distance to face is {final_depth} meters")
            break
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', color_image)
        cv2.waitKey(1)
finally:
    pipe.stop()
