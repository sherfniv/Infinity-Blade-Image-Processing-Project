import cv2
from cv2 import GaussianBlur
import numpy as np
from color_detector import ColorDetector
from kalmanfilter import KalmanFilter
import win32gui
import math
import win32con
import time
import subprocess
import platform

def open_file(file_path):
    if platform.system() == "Windows":
        subprocess.call(["start", "", file_path], shell=True)
    elif platform.system() == "Darwin":  # macOS
        subprocess.call(["open", file_path])
    else:  # Linux
        subprocess.call(["xdg-open", file_path])


def send_key_to_game(key):
    if key == None:
        return
    window_title = "Infinity Blade"

    # Find the han``
    game_window = win32gui.FindWindow(None, window_title)

    # Set the game window as foreground
    #win32gui.SetForegroundWindow(game_window)

    # Send key press event to the game window
    win32gui.SendMessage(game_window, win32con.WM_KEYDOWN, key, 0)
    win32gui.SendMessage(game_window, win32con.WM_KEYUP, key, 0)
    time.sleep(0.02)
    
def key_up(key):
    if key == None:
        return

    window_title = "Infinity Blade"

    # Find the handle of the game window
    game_window = win32gui.FindWindow(None, window_title)

    # Set the game window as foreground
    #win32gui.SetForegroundWindow(game_window)

    # Send key press event to the game window
    win32gui.SendMessage(game_window, win32con.WM_KEYUP, key, 0)
    
    return False
def key_down(key):
    if key == None:
        return

    window_title = "Infinity Blade"

    # Find the handle of the game window
    game_window = win32gui.FindWindow(None, window_title)

    # Set the game window as foreground
    #win32gui.SetForegroundWindow(game_window)

    # Send key press event to the game window
    win32gui.SendMessage(game_window, win32con.WM_KEYDOWN, key, 0)
    time.sleep(0.02)    
    return True

def dodge(frame, center_x, smoothed_center_x, position_label, left_thresh, right_thresh, fgbg, num_prev_frames, prev_centers):

    mask = cd.undetect(frame)
    not_sword = cv2.bitwise_and(frame, frame, mask=mask)
    # Apply background subtraction
    fgmask = fgbg.apply(not_sword)

    # Apply morphological operations to remove noise
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel=np.ones((7, 7), np.uint8))

    # Find contours in the foreground mask
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #draw left and right thresholds
    cv2.line(frame, (left_thresh, 0), (left_thresh, frame.shape[0]), (0, 255, 0), 2)
    cv2.line(frame, (right_thresh, 0), (right_thresh, frame.shape[0]), (0, 255, 0), 2)

    # Find the largest contour
    max_contour_area = 0
    max_contour = None
    area_thresh = 5000#7000  # Ignore contours with a small area
    for contour in contours:
        area = cv2.contourArea(contour)
        # global frame_count
        if area > max_contour_area and area > area_thresh:
            max_contour_area = area
            max_contour = contour
            # if frame_count % 15 == 0:
            #    print(f"area: {area}")

    # Calculate the centroid of the largest contour
    if max_contour is not None:
        M = cv2.moments(max_contour)
        center_x = int(M["m10"] / M["m00"])

        # Draw contour to visualize player segmentation
        #cv2.drawContours(frame, [max_contour], -1, (0, 255, 0), 2)
    else:
        center_x = None

    # Initialize new_position_label with a default value
    new_position_label = "Center"

    # Smooth the center_x coordinate using a moving average
    if center_x is not None:
        prev_centers.append(center_x)
        if len(prev_centers) > num_prev_frames:
            prev_centers.pop(0)
        # Weighted average
        smoothed_center_x = int(sum([(i + 1) * prev_centers[i] for i in range(len(prev_centers))]) / sum(range(1, 1 + len(prev_centers))))

        # Update position label based on smoothed center_x
        if smoothed_center_x > left_thresh:
            new_position_label = "Left"
        elif smoothed_center_x < right_thresh:
            new_position_label = "Right"

    # Send key only if position has changed
    if position_label != new_position_label:
        position_label = new_position_label
        if position_label == "Right":
            key_to_press = ord('D')
        elif position_label == "Left":
            key_to_press = ord('A')
        else:
            key_to_press = None
    else:
        key_to_press = None

    # Draw center line
    if smoothed_center_x is not None:
        cv2.line(frame, (smoothed_center_x, 0), (smoothed_center_x, frame.shape[0]), (0, 255, 0), 2)

    # Add text annotation
    cv2.putText(frame, f"Position: {position_label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, key_to_press, position_label


def detect_blocking(frame, block):
    # Define the region of interest (upper part of the frame)
    height = frame.shape[0]
    width = frame.shape[1]
    #fgmask = fgbg.apply(frame)
    hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
    #fgmask = fgbg.apply(frame)
    region_of_interest = hsv_img[:height//3,width//4:width*3//4]
    mask = cv2.inRange(region_of_interest, cd.low_color, cd.high_color)
    res = cv2.bitwise_and(region_of_interest, region_of_interest, mask=mask)
    # Convert to grayscale
    gray = cv2.cvtColor(region_of_interest, cv2.COLOR_BGR2GRAY)
    gray = res
    # Apply Gaussian Blur
    blurred = cv2.medianBlur(gray,3)
    # Edge detection
    edges = cv2.Canny(blurred, 20, 500, apertureSize=3)
    # Detect lines using Hough Line Transform
    #lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=40, maxLineGap=40)
    lines = cv2.HoughLinesP(edges, rho = 1, theta =np.pi/180, threshold =15, minLineLength = 30, maxLineGap = 40)

    blocking = find_lines(lines)
    if blocking:
        block += 3
    else:
        block = block*0.3
        block -= 1
        if block < 0:
            block = 0
    
    if block > 0:
        return True, block
    return False, block

def find_lines(lines):
    # Define the angle threshold for your desired angles (in radians)
    angle_threshold = np.pi / 6  # Approximately equal to 30 degrees in radians
    # Iterate through each line
    if lines is None:
        return 0
    for line in lines:
        x1, y1, x2, y2 = line[0]  # Extract the coordinates of the line
        # Calculate the angle of the line
        angle = np.arctan2(y2 - y1, x2 - x1)
        # Check if the angle is approximately 30 degrees or -30 degrees
        if np.abs(angle - angle_threshold) < angle_threshold or np.abs(angle + angle_threshold) < angle_threshold:
            cv2.line(frame, (x2+ frame.shape[1]//4,y2), (x1 + frame.shape[1]//4,y1), (0, 255, 255), 30)
            return 1
        else:
            return 0

def preprocess_frame(frame):
    # Apply Gaussian filter to the frame
    blurred_frame = GaussianBlur(frame, (9, 9), 0)
    return blurred_frame

def distance(x, y, z ,w):
    d = math.sqrt((x-z)**2+(y-w)**2)
    return d

def slope(x,y,z,w):
    if x == z:
        return math.inf      
    else:
        m = (w-y)/(z-x)
        return m
    
def movement(frame):

    sword = cd.detect(frame)
    if sword is not None:
        M = cv2.moments(sword)
        if M["m00"] != 0:  # Ensure that the contour has nonzero area
            sword_x = int(M["m10"] / M["m00"])
            sword_y = int(M["m01"] / M["m00"])
        else:
            # Set center coordinates to None if the contour has zero area
            sword_x = None
            sword_y = None
    else:
        sword_x = None
        sword_y = None

    if sword_x is not None and sword_y is not None:
        cv2.circle(frame, (sword_x, sword_y), 10, (255, 255, 0), -1)
        cv2.putText(frame, "tracking object", (sword_x + 10, sword_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)    

        px, py = kf.predict(sword_x, sword_y)
        cv2.circle(frame, (px, py), 10, (0, 0, 255), -1)  
    else:
        px,py = None, None

    return sword_x, sword_y, px, py
    
# Create video capture
cap = cv2.VideoCapture(0)

color = int(input("sword color: 1- green , 2 - purple "))
# Load detector
    
cd = ColorDetector(color)

# Load Kalman Filter
kf = KalmanFilter()

# Frame count to keep track of frames
frame_count = 0

# Counter to keep track of how many frames the text has been displayed for
text_counter = 0

# Text to display
text_to_display = ""

# Tracking List
track_list = []

# Region List
region_list = []

# Initialize variables for tracking
center_x = None
smoothed_center_x = None
position_label = "Center"
# Define frame regions
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# Initialize left and right thresholds
right_thresh = int(0.35 * frame_width)
left_thresh = int(0.65 * frame_width)
prev_py = None
prev_px = None
# Create background subtractor object
fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold= 20)

# Define the number of previous frames to consider for smoothing
num_prev_frames = 7
prev_centers = []
block = 0
blocking = False
block_sent = False

cx = 0
cy = 0
px = 0
py = 0
coordX_array = np.array([[np.float32(cx)]])
coordY_array = np.array([[np.float32(cy)]])
key_to_press_block = ord('B')

while True:

    key_to_press_slash = None
    key_to_press_dodge = None

    ret, frame = cap.read()
    if not ret:
        break
    
    # Preprocess the frame
    blurred_frame = preprocess_frame(frame)
    
    # Increment frame count
    frame_count += 1

    # Define the frame dimensions
    frame_height, frame_width, _ = blurred_frame.shape

    temp_cx, temp_cy, temp_px, temp_py = movement(frame)

    if temp_cx is not None:
        cx = temp_cx
        cy = temp_cy
        px = temp_px
        py = temp_py

    d = distance(cx, cy, px, py)
    m = slope(cx, cy, px, py)  
    if text_counter > 0:
        cv2.putText(frame, text_to_display, (100,100), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
        text_counter -= 1 
    else:
        if d > 65:
            if abs(m) > 2: 
                if cy < py:
                    text_to_display = "Up Slash"
                    text_counter = 10
                    key_to_press_slash = ord("1")
                else: 
                    text_to_display = "Down Slash"
                    text_counter = 10
                    key_to_press_slash = ord("2")
            elif 0.5 < m < 2:
                if cy < py:
                    text_to_display = "Up Left Slash"
                    text_counter = 10
                    key_to_press_slash = ord("3")
                else: 
                    text_to_display = "Down Right Slash"
                    text_counter = 10
                    key_to_press_slash = ord("4")
            elif -0.5 < m < 0.5:
                if cx < px:
                    text_to_display = "Right Slash"
                    text_counter = 10
                    key_to_press_slash = ord("5")
                else: 
                    text_to_display = "Left Slash"
                    text_counter = 10
                    key_to_press_slash = ord("6")
            elif -2 < m < -0.5:
                if cy < py:
                    text_to_display = "Up Right Slash"
                    text_counter = 10
                    key_to_press_slash = ord("7")
                else: 
                    text_to_display = "Down Left Slash" 
                    text_counter = 10 
                    key_to_press_slash = ord("8")      
    

    # Call the dodge function
    frame, key_to_press_dodge, position_label = dodge(frame, center_x, smoothed_center_x, position_label, left_thresh, right_thresh, fgbg, num_prev_frames, prev_centers)
    #if key_to_press_dodge is not None:
        #print(key_to_press_dodge)

    ## Blocking implemtation:
    if not blocking:
        if frame_count % 5 == 0:
            blocking, block = detect_blocking(frame, block)
    else:
        blocking, block = detect_blocking(frame, block)


    if blocking:
        if not block_sent:
            print(chr(key_to_press_block))
            block_sent = key_down(key_to_press_block)


    if not blocking and block_sent:
        block_sent = key_up(key_to_press_block)
             
    if not block_sent:
        if key_to_press_slash is not None:
            print(chr(key_to_press_slash))
            send_key_to_game(key_to_press_slash)
        elif key_to_press_dodge is not None:
            if text_counter == 0:
                print(chr(key_to_press_dodge))
                send_key_to_game(key_to_press_dodge)

    #cv2.imshow('Frame1', blurred_frame)
    cv2.imshow('Frame', frame)
    k = cv2.waitKey(1)
    if k == ord("q"):  # Press 'q' to exit
        break
    prev_px = px
    prev_py = py
cap.release()
cv2.destroyAllWindows()
