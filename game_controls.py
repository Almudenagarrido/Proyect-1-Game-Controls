"""
COMP 332 Grid-based game controls.

Name: Almudena Garrido
Email: agarridogarciapita@sandiego.edu
Date: 02/08/2026
"""

import pyautogui

last_position = (None,None)
last_dir = ''

def debugging(isdebugging,action,direction):
    """
    Prints the action that was used and the direction if debugging mode is on.

    Parameters:
    isdebugging (bool): indicates if the debugging mode is on
    action (str): The action that was taken depending on the mode running
    direction (str): The direction expected (e.g., up, down, left, right)

    Return: None
    """
    if isdebugging:
        print(action,"was used to go",direction)


def keypress(is_debugging):
    ''' 
    Controls a grid based game through the use of the keyboard. 
    The keymapping is as follows:
    w = up, s = down, a = left, d = right, q = exit

    Parameters:
    is_debugging (bool): indicates if the debugging mode is on

    Returns: None
    '''

    from pynput import keyboard

    # function that takes action when a key is released
    def on_release(key):
        try:
            if key.char == 'w':
                pyautogui.press('up')
                debugging(is_debugging, 'keyboard', 'up')

            elif key.char == 's':
                pyautogui.press('down')
                debugging(is_debugging, 'keyboard', 'down')

            elif key.char == 'a':
                pyautogui.press('left')
                debugging(is_debugging, 'keyboard', 'left')

            elif key.char == 'd':
                pyautogui.press('right')
                debugging(is_debugging, 'keyboard', 'right')

            elif key.char == 'q':
                print("Exiting keyboard control.")
                return False 

        except AttributeError:
            pass
    
    # listens for a key to be pressed
    with keyboard.Listener(
        on_release=on_release) as listener:
        listener.join()



def trackpad_mouse(is_debugging):
    ''' 
    Controls a grid based game through the use of the mouse/trackpad. 

    Parameters:
    is_debugging (bool): indicates if the debugging mode is on

    Returns: None
    '''

    from pynput import mouse

    # function that takes action when the mouse is moved
    def on_move(x, y):
        global last_position
        global last_dir
        # TODO: fill in your code here and delete the pass
        pass
    
    # listens for a mouse movement
    with mouse.Listener(on_move=on_move) as listener:
        listener.join() 

def color_tracker(is_debugging):
    ''' 
    Controls a grid based game through the use of tracking a x colored object. 

    Parameters:
    is_debugging (bool): indicates if the debugging mode is on

    Returns: None
    '''

    import cv2
    import imutils
    import numpy as np
    from collections import deque
    import time
    import multithreaded_webcam as mw


    # TODO: You need to define HSV colour range (currently green)
    colorLower = (29, 86, 6)
    colorUpper = (64, 255, 255)

    # set the limit for the number of frames to store and the number that have seen direction change
    buffer = 20
    pts = deque(maxlen = buffer)

    # store the direction and number of frames with direction change
    num_frames = 0
    (dX, dY) = (0, 0)
    direction = ''
    global last_dir

    #Sleep for 2 seconds to let camera initialize properly
    time.sleep(2)
    #Start video capture
    vs = mw.WebcamVideoStream().start()

    is_running = True

    while is_running:
        #get the frame, flip, and resize
        frame = vs.read()
        frame = cv2.flip(frame,1)
        frame = imutils.resize(frame, width = 600)
        blurred_frame = cv2.GaussianBlur(frame, (5,5), 0)
        hsv_converted_frame = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        #Create a mask for the frame, get ride of white dots, and make present
        mask = cv2.inRange(hsv_converted_frame, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations = 2)
        mask = cv2.dilate(mask, None, iterations = 2)

        #Find all contours in the masked image
        cnts,_ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #Define center of the ball to be detected as None
        center = None

        #If any object is detected, then only proceed
        if(len(cnts) > 0):
            #Find the contour with maximum area
            c = max(cnts, key = cv2.contourArea)
            #Find the center of the circle, and its radius of the largest detected contour.
            ((x,y), radius) = cv2.minEnclosingCircle(c)
            #Calculate the centroid of the ball, as we need to draw a circle around it.
            M = cv2.moments(c)
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

            #Proceed only if a ball of considerable size is detected
            if radius > 10:
                cv2.circle(frame, (int(x), int(y)), int(radius), (0,255,255), 2)
                cv2.circle(frame, center, 5, (0,255,255), -1)
                pts.appendleft(center)

        #If at least 10 frames have direction change, proceed
        if num_frames >= 10 and (len(pts) >= 10 and pts[9] is not None):
            #TODO: calculate direction
            pass

        # TODO: complete key press
        
        
        #Update counter
        num_frames += 1

        #Write the detected direction on the frame.
        cv2.putText(frame, direction, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        #Show the output frame.
        cv2.imshow('Game Control Window', frame)
        cv2.waitKey(1)

    # Close all windows and close the video stream.
    cv2.destroyAllWindows()
    vs.stop()
        



def finger_tracking(is_debugging):
    ''' 
    Controls a grid based game through the use of finger tracking 
    The finger mapping is as follows: x = up, x = down, x = left, x = right, x = exit

    Parameters:
    is_debugging (bool): indicates if the debugging mode is on

    Returns: None
    '''

    import cv2
    import imutils
    import numpy as np
    import time
    import multithreaded_webcam as mw
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    # configuration to set the model and using only one hand
    MODEL_PATH = 'hand_landmarker.task'
    num_hands = 1

    # Standard MediaPipe Hand Connections (The Skeleton)
    # Each tuple represents a line between two landmark indices
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    ##Sleep for 2 seconds to let camera initialize properly
    time.sleep(2)
    #Start video capture
    vs = mw.WebcamVideoStream().start()

    #Get the trained model for the hands
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    # function to count number of fingers up
    def count_fingers(hand_landmarks, handedness):
        """Counts fingers based on tip vs joint position."""
        #TODO: complete this code to count how many fingers are being held up
        pass

    global last_dir
    is_running = True

    while is_running:
        # Get frame, flip it, resize it, and convert to RGB
        img = vs.read()
        img = cv2.flip(img, 1)
        img = imutils.resize(img, width=600)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks from the input image.
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        # Draw Results if available
        if result.hand_landmarks:
            h, w, _ = img.shape
            
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                raw_label = result.handedness[i][0].category_name
                display_label = "Left" if raw_label == "Right" else "Right"
                
                # 1. DRAW SKELETON (Lines)
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    # Convert normalized coordinates (0-1) to pixel coordinates
                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    
                    # Draw the line (White)
                    cv2.line(img, start_point, end_point, (240, 240, 240), 2)

                # 2. DRAW LANDMARKS (Dots)
                wrist_coords = (0, 0)
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx == 0: wrist_coords = (cx, cy)
                    
                    # Different color for tips vs joints
                    color = (0, 255, 0) if idx in [4, 8, 12, 16, 20] else (0, 0, 255)
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

                # 3. COUNT & DISPLAY
                count = count_fingers(hand_landmarks, display_label)
                cv2.putText(img, f"Count: {count}", (wrist_coords[0], wrist_coords[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
            # TODO: complete key press

        #show video
        cv2.imshow('Finger Counter + Skeleton', img)
        cv2.waitKey(1)


    # Close all windows and stop the video stream
    cv2.destroyAllWindows()
    vs.stop()
    landmarker.close()

def unique_control(is_debugging):
    ''' 
    Fill in the description.

    Parameters:
    is_debugging (bool): indicates if the debugging mode is on

    Returns: None
    '''
    #TODO: fill in your code here and delete the pass
    pass

def main():
    is_debugging = False
    debug_quiry = input('Would you like to print the direction you are going while running the game (y or n)? ')
    if debug_quiry == 'y':
        is_debugging = True
        print("Printing in now turned on!")
    control_mode = input("How would you like to control the game?\n\tPress 1 for keyboard\n\tPress 2 for the mouse/trackpad\n\tPress 3 for the color tracker\n\tPress 4 for the finger tracker\n\tPress 5 for the unique tracker\n")
    if control_mode == '1':
        keypress(is_debugging)
    elif control_mode == '2':
        trackpad_mouse(is_debugging)
    elif control_mode == '3':
        color_tracker(is_debugging)
    elif control_mode == '4':
        finger_tracking(is_debugging)
    elif control_mode == '5':
        unique_control(is_debugging)

if __name__ == '__main__':
	main()
