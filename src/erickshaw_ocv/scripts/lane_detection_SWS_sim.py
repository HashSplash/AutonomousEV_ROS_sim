#!/usr/bin/env python3


import cv2
import numpy as np
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

import rospy
from std_msgs.msg import Float64,String
from cv_bridge import CvBridge,CvBridgeError
from sensor_msgs.msg import Image

# Meter-to-pixel conversion factors
ym_per_pix = 30 / 720
xm_per_pix = 3.7 / 720

def nothing(x):
    pass

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L - H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L - V", "Trackbars", 200, 255, nothing)
cv2.createTrackbar("U - H", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U - S", "Trackbars", 50, 255, nothing)
cv2.createTrackbar("U - V", "Trackbars", 255, 255, nothing)

###########     applpying filters on the image to get the edges     ##################3333333
def process_image(image,l_h,l_s,l_v,u_h,u_s,u_v):
    # Apply HLS color filtering to filter out white lane lines
    hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    # lower_white = np.array([0, 109, 124])
    # upper_white = np.array([149, 149, 149])
    lower_white = np.array([l_h, l_s, l_v])
    upper_white = np.array([u_h, u_s, u_v])
    white_mask = cv2.inRange(hls, lower_white, upper_white)
    white_lines = cv2.bitwise_and(image, image, mask=white_mask)

    # Convert image to grayscale, apply threshold, blur & extract edges
    gray = cv2.cvtColor(white_lines, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    blur = cv2.GaussianBlur(thresh, (3, 3), 0)
    canny = cv2.Canny(blur, 40, 60)

    return image, white_lines, gray, thresh, blur, canny, white_mask


#########   Transforming normal image to Top-Down view   ###################
def perspective_warp(image):
    # Define perspective transformation points
    # src = np.float32([[600, 540], [685, 540], [200, 720], [805, 720]])
    # dst = np.float32([[200, 0], [1200, 0], [200, 710], [1200, 710]])

    src = np.float32([[450,120], [0,472], [900,120], [1200,472]])
    dst = np.float32([[0,0], [0,720], [1280, 0], [1280, 720]])


    # Get perspective transformation matrix
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    # Apply perspective transformation
    birdseye = cv2.warpPerspective(image, matrix, (image.shape[1], image.shape[0]))

    return birdseye, inverse_matrix


###########     Plotting the Histogram to find the starting of lanes    ####################
def plot_histogram(image):
    histogram = np.sum(image[image.shape[0] // 2:, :], axis=0)
    midpoint = np.int_(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    return histogram, leftx_base, rightx_base


##########      Applying Sliding Window Search to track the Lanes   #######################
def slide_window_search(binary_warped, histogram):
    # Implementation of sliding window search
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = np.int_(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = np.int_(binary_warped.shape[0] / nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    margin = 100
    minpix = 50
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify nonzero pixels in x and y within the window
        good_left_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
        ).nonzero()[0]
        good_right_inds = (
            (nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
            (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
        ).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = np.int_(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int_(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, binary_warped.shape[1])
    plt.ylim(binary_warped.shape[0], 0)

    return ploty, left_fit, right_fit, left_fitx, right_fitx


########    applying general search to further look for the pixels inside the sliding window    ##########
def general_search(binary_warped, left_fit, right_fit):
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy**2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy**2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin, ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin, ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    ret = {}
    ret['leftx'] = leftx
    ret['rightx'] = rightx
    ret['left_fitx'] = left_fitx
    ret['right_fitx'] = right_fitx
    ret['ploty'] = ploty

    return ret


###########     Drawing lane lines back on the original image for viewing   ###############
def draw_lane_lines(original_image, warped_image, minv, draw_info):
    leftx = draw_info['leftx']
    rightx = draw_info['rightx']
    left_fitx = draw_info['left_fitx']
    right_fitx = draw_info['right_fitx']
    ploty = draw_info['ploty']

    warp_zero = np.zeros_like(warped_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    mean_x = np.mean((left_fitx, right_fitx), axis=0)
    pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])
    
    # Draw the lane area on the warped image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

    newwarp = cv2.warpPerspective(color_warp, minv, (original_image.shape[1], original_image.shape[0]))
    result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

    return pts_mean, result


#########   Calculating Lnae Curvature  ###########
def measure_lane_curvature(ploty, leftx, rightx):
    leftx = leftx[::-1]  # Reverse to match top-to-bottom in y
    rightx = rightx[::-1]  # Reverse to match top-to-bottom in y

    y_eval = np.max(ploty)

    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0]
    )
    right_curverad = (
        (1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit_cr[0])

    if leftx[0] - leftx[-1] > 60:
        curve_direction = 'Left Curve'
    elif leftx[-1] - leftx[0] > 60:
        curve_direction = 'Right Curve'
    else:
        curve_direction = 'Straight'

    return (left_curverad + right_curverad) / 2.0, curve_direction


############    Calculating how off center is the lane  ###############
def off_center(mean_pts, inp_frame):

    mean_x_bottom = mean_pts[-1][-1][-2].astype(int)
    pixel_deviation = inp_frame.shape[1] / 2 - abs(mean_x_bottom)
    deviation = pixel_deviation * xm_per_pix
    direction = "left" if deviation < 0 else "right"

    return deviation, direction


###########     adding text for viewing     #############

# video = cv2.VideoCapture()
# ret,frame = video.read()
deviation,direction_dev=0.0,0.0
# direction_dev=0.0

# def cb(data):
#     # global video
#     # global frame
#     # global deviation
#     # global direction_dev
#     bridge = CvBridge()
#     try:
#         cv_image= bridge.imgmsg_to_cv2(data,desired_encoding="bgr8")
#     except:
#         rospy.logerr(CvBridgeError)
#     # video = cv_image
#     # _,frame = cv_image.read()
#     frame = cv_image

    # birdseye, minverse = perspective_warp(frame)
    # processed_image = process_image(birdseye)
    # hist, left_base, right_base = plot_histogram(processed_image[3])
    # ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(processed_image[3], hist)
    # draw_info = general_search(processed_image[3], left_fit, right_fit)
    # # curve_rad, curve_dir = measure_lane_curvature(ploty, left_fitx, right_fitx)
    # mean_pts, result = draw_lane_lines(frame, processed_image[3], minverse, draw_info)
    # deviation, direction_dev = off_center(mean_pts, frame)
    # cv2.imshow("Final", result)
    # cv2.waitkey(1)



class img_rec:
    def __init__(self):
        rospy.Subscriber("/image_raw", Image, self.cb)
        self.deviation=0
        self.direction_dev=0
        

    def cb(self,data):
        bridge=CvBridge()
        try:
            cv_image=bridge.imgmsg_to_cv2(data,desired_encoding="bgr8")
        except CvBridgeError as e:
            rospy.logerr(e)

        frame=cv_image
        # cv2.imshow("Final", frame)
        frame=cv2.resize(frame,(1200,720))

        birdseye, minverse = perspective_warp(frame)
        l_h = cv2.getTrackbarPos("L - H", "Trackbars")
        l_s = cv2.getTrackbarPos("L - S", "Trackbars")
        l_v = cv2.getTrackbarPos("L - V", "Trackbars")
        u_h = cv2.getTrackbarPos("U - H", "Trackbars")
        u_s = cv2.getTrackbarPos("U - S", "Trackbars")
        u_v = cv2.getTrackbarPos("U - V", "Trackbars")
        # processed_image = process_image(birdseye,l_h,l_s,l_v,u_h,u_s,u_v)
        processed_image = process_image(birdseye,0,62,0,95,146,255)
        hist, left_base, right_base = plot_histogram(processed_image[3])
        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(processed_image[3], hist)
        draw_info = general_search(processed_image[3], left_fit, right_fit)
        # curve_rad, curve_dir = measure_lane_curvature(ploty, left_fitx, right_fitx)
        mean_pts, result = draw_lane_lines(frame, processed_image[3], minverse, draw_info)
        self.deviation, self.direction_dev = off_center(mean_pts, frame)
        cv2.imshow("Final", result)
        # cv2.imshow("Final1",birdseye)
        cv2.imshow("Final2",processed_image[3])
        cv2.waitKey(2)
        # status=cv2.imwrite("/home/harshit/ROS/catkin_ws1/src/erickshaw_ocv/scripts/perspective_warp.png",birdseye)
        # cv2.destroyAllWindows()

    def return_val(self):
        return (self.deviation,self.direction_dev)



if __name__=="__main__":
    rospy.init_node("lane_detection_py", anonymous=True)
    pub_dev= rospy.Publisher("steering_controller/value/deviation", Float64, queue_size=10)
    pub_dir= rospy.Publisher("steering_controller/value/direction", String, queue_size=10)

    
    # video_path = "videoplayback.mp4"
    # video = read_video(video_path)
    img=img_rec()
    t=rospy.Rate(10)
    while not rospy.is_shutdown():
        # _, frame = video.read()
        
        
        #final_img = add_text(result, curve_rad, curve_dir, deviation, direction_dev)

        # outputs are deviation, direction_dev, curve_rad, curve_dir
        deviation,direction_dev=img.return_val()
        deviation = Float64(deviation)
        rospy.loginfo("deviation={} and direction={}".format(deviation,direction_dev))
        pub_dev.publish(deviation)
        pub_dir.publish(direction_dev)
        
        # if cv2.waitKey(1) == 13:
        #     break
        t.sleep()

    # video.release()
    cv2.destroyAllWindows()
