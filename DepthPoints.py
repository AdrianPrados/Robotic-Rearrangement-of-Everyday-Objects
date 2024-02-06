import pyrealsense2 as rs
import cv2
import numpy as np

def main(point_x, point_y):

    #? ====== REALSENSE CONFIG ======
    ''' Detect the camera RealSense D435i and activate it'''
    realsense_ctx = rs.context()
    connected_devices = [] # List of serial numbers for present cameras
    for i in range(len(realsense_ctx.devices)):
        detected_camera = realsense_ctx.devices[i].get_info(rs.camera_info.serial_number)
        print("Detected_camera")
        connected_devices.append(detected_camera)
    device = connected_devices[0] # For this code only one camera is neccesary
    pipeline = rs.pipeline()
    config = rs.config()
    background_removed_color = 153 # Grey color for the background

    # ====== Enable Streams ======
    ''' Activate the stream caracteristics for the RealSense D435i'''
    config.enable_device(device)

    #For better FPS. but worse resolution:

    stream_res_x = 640
    stream_res_y = 480

    stream_fps = 30

    config.enable_stream(rs.stream.depth, stream_res_x, stream_res_y, rs.format.z16, stream_fps)
    config.enable_stream(rs.stream.color, stream_res_x, stream_res_y, rs.format.bgr8, stream_fps)
    profile = pipeline.start(config)

    align_to = rs.stream.color
    align = rs.align(align_to)

    # ====== Get depth Scale ======
    ''' Obtain the scale of the depth estimated by the depth sensors of the camara''' 
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale for Camera SN",device,"is: ",depth_scale)

    # ====== Set clipping distance ======
    ''' Generate the maximun distance for the camera range to detect'''
    clipping_distance_in_meters = 5
    clipping_distance = clipping_distance_in_meters / depth_scale
    print("Configuration Successful for SN", device)

    # ====== Get and process images ====== 
    print("Starting to capture images on SN:",device)
    #? ================================================================================================


    #? ======= PROCESS FRAME =========
    while True:

        #* Get and align frames from the camera to the point cloud
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        
        if not aligned_depth_frame or not color_frame:
            print("\tNot aligned")
            continue

        #* Process images to work with one color image
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        depth_image_flipped = cv2.flip(depth_image,1)
        #color_image = np.asanyarray(color_frame.get_data())

        """ depth_image_3d = np.dstack((depth_image,depth_image,depth_image)) #Depth image is 1 channel, while color image is 3
        background_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), background_removed_color, color_image) """

        #depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        #images = cv2.flip(background_removed,1)
        """ color_image = cv2.flip(color_image,1)
        color_images_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB) """

        #* Load the intrinsics values of the camera RealSense D435i
        INTR = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        #* Z values for the point
        Z_point = depth_image_flipped[point_y,point_x] * depth_scale # meters
        #Z_point = depth_image[point_x,point_y] * depth_scale # meters

        #* Values of the different studied points in meters
        Point = rs.rs2_deproject_pixel_to_point(INTR,[point_x,point_y],Z_point)
        break

    return Z_point, Point


if __name__=="__main__":
    """ point_x = 250
    point_y = 240 """
    main(point_x, point_y)