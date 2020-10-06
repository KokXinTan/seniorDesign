import pyrealsense2 as rs
import numpy as np
import cv2
# import os

# opencv-haar人脸检测
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipe_profile = pipeline.start(config)

curr_frame = 0

try:
    while True:

        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if frames.size() < 2:
            continue

        if not depth_frame or not color_frame:
            continue

        # Intrinsics & Extrinsics
        # 深度相机内参矩阵
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        # RGB相机内参矩阵
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        # 深度图到彩图的外参RT
        depth_to_color_extrin = depth_frame.profile.get_extrinsics_to(
            color_frame.profile)

        depth_value = 0.5
        depth_pixel = [depth_intrin.ppx, depth_intrin.ppy]
        depth_point = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, depth_value)
        #print(depth_point)


        # print(depth_intrin.ppx, depth_intrin.ppy)

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # 将RGB对齐到深度，获取对应下的XYZ
        #Color->Depth
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

        # 找到人脸
        # find the human face in the color_image
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        left = []

        for (x, y, w, h) in faces:
            # 当前帧大于100
            if curr_frame > 100 and curr_frame % 40 == 10:

                # 取出人脸的深度图和彩色图
                roi_depth_image = depth_image[y:y+h, x:x+w]
                roi_color_image = color_image[y:y+h, x:x+w]

                # W
                cols = list(range(x, x+w))
                # H
                rows = list(range(y, y+h))
                for i in rows: #H
                    for j in cols: #W
                        # 坐标变换一定要注意检查
                        # 此时获取的是真实世界坐标的深度
                        # https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/hpp/rs_frame.hpp#L810
                        depth = depth_frame.get_distance(j, i) # W,H
                        # 给定没有失真或反失真系数的图像中的像素坐标和深度，计算相对于同一相机的3D空间中的对应点
                        # https://github.com/IntelRealSense/librealsense/blob/master/include/librealsense2/rsutil.h#L67
                        depth_point = rs.rs2_deproject_pixel_to_point(
                            depth_intrin, [j, i], depth)
                        text = "%.5lf, %.5lf, %.5lf\n" % (
                             depth_point[0], depth_point[1], depth_point[2])
                        #f.write(text)
                        if i==rows[0]:
                            left.append(depth_point)

                #print("Finish writing the depth img")
                # temp = np.array(left)
                # # 求均值
                # _mean = np.mean(temp, axis=0)
                # # 求方差
                # _var = np.var(temp, axis=0)
                # minmean = _mean - 1 * abs(_mean)
                # maxmean = _mean + 1 * abs(_mean)
                # minvar = _var - 1 * abs(_var)
                # maxvar = _var + 1 * abs(_var)


                def non_zero_mean(np_arr, axis):
                    exist = (np_arr != 0)
                    num = np_arr.sum(axis=axis)
                    den = exist.sum(axis=axis)
                    return num / den


                temp = np.array(left)
                # 求均值
                _mean = non_zero_mean(temp, axis=0)
                # 求方差
                _var = np.var(temp, axis=0)
                minmean = _mean - 1 * abs(_mean)
                maxmean = _mean + 1 * abs(_mean)
                minvar = _var - 1 * abs(_var)
                maxvar = _var + 1 * abs(_var)


                index = []
                i = 0 # H
                for j in range(len(cols)):  # W
                    if temp[j][0] != 0 and temp[j][1] != 0 and temp[j][2] != 0:
                        if temp[j][0]>minmean[0] and temp[j][0]<maxmean[0]:
                            if temp[j][1] > minmean[1] and temp[j][1] < maxmean[1]:
                                if temp[j][2] > minmean[2] and temp[j][2] < maxmean[2]:
                                    index.append(j)


                #dist2 = np.sqrt(np.square(left[index[-1]][0] - left[index[0]][0])+np.square(left[index[-1]][1] - left[index[0]][1])+np.square(left[index[-1]][2] - left[index[0]][2]))
                # // 计算两点之间的欧几里得距离
                # return sqrt(pow(upoint[0] - vpoint[0], 2) +
                #             pow(upoint[1] - vpoint[1], 2) +
                #             pow(upoint[2] - vpoint[2], 2));
                #这里的距离，收到环境的影响，因为我是直接计算框里面最左端到最右端的距离
                #如果把背景框进来，那么你测的是两个背景的宽度
                if len(index) > (len(cols)/2):
                    # 新建
                    # os.system('mkdir -p ./3d_output/%d' % curr_frame)
                    # 保存
                    # cv2.imwrite('./3d_output/%d/depth.jpg' %
                    #             curr_frame, roi_depth_image)
                    # cv2.imwrite('./3d_output/%d/color.jpg' %
                    #             curr_frame, roi_color_image)

                    print("dist","---------------------", str(left[index[-1]][0] - left[index[0]][0]))

                    # 这里要做很多工作，离群噪声点的去除，去除后矩阵的真实大小判断 很多行，哪一行是最真实的距离
                    cv2.putText(color_image, str(left[index[-1]][0] - left[index[0]][0]),
                                (x, y - 30),
                                cv2.FONT_HERSHEY_COMPLEX, 1.5, (255, 255, 255))
            cv2.rectangle(color_image, (x, y), (x+w, y+h), (255, 0, 0), 2)

         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(
            depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Stack both images horizontally
        images = np.hstack((color_image, depth_colormap))

        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        cv2.waitKey(1)

        curr_frame += 1
finally:

    # Stop streaming
    pipeline.stop()
