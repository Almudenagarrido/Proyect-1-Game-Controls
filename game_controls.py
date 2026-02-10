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

        THRESHOLD = 30

        if last_position == (None, None):
            last_position = (x, y)
            return

        last_x, last_y = last_position
        dx = x - last_x
        dy = y - last_y

        if abs(dx) < THRESHOLD and abs(dy) < THRESHOLD:
            return

        if abs(dx) > abs(dy):
            if dx > 0 and last_dir != 'right':
                pyautogui.press('right')
                last_dir = 'right'
                debugging(is_debugging, 'mouse', 'right')

            elif dx < 0 and last_dir != 'left':
                pyautogui.press('left')
                last_dir = 'left'
                debugging(is_debugging, 'mouse', 'left')

        else:
            if dy > 0 and last_dir != 'down':
                pyautogui.press('down')
                last_dir = 'down'
                debugging(is_debugging, 'mouse', 'down')

            elif dy < 0 and last_dir != 'up':
                pyautogui.press('up')
                last_dir = 'up'
                debugging(is_debugging, 'mouse', 'up')

        last_position = (x, y)
    
    # listens for a mouse movement
    with mouse.Listener(on_move=on_move) as listener:
        listener.join() 

def color_tracker(is_debugging):
    ''' 
    Controls a grid based game through the use of tracking a colored object. 
    '''

    import cv2
    import imutils
    import numpy as np
    from collections import deque
    import time
    import pyautogui
    import multithreaded_webcam as mw

    colorLower = (95, 80, 40)
    colorUpper = (135, 255, 255)

    buffer = 20
    pts = deque(maxlen=buffer)

    num_frames = 0
    (dX, dY) = (0, 0)
    direction = ''
    last_dir = ''

    time.sleep(2)
    vs = mw.WebcamVideoStream().start()

    is_running = True

    while is_running:
        frame = vs.read()
        frame = cv2.flip(frame, 1)
        frame = imutils.resize(frame, width=600)

        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, colorLower, colorUpper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)

            if M['m00'] != 0:
                center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                    cv2.circle(frame, center, 5, (0, 255, 255), -1)
                    pts.appendleft(center)

        if num_frames >= 10 and len(pts) >= 10 and pts[9] is not None:
            dX = pts[0][0] - pts[9][0]
            dY = pts[0][1] - pts[9][1]

            THRESHOLD = 5
            direction = ''

            if abs(dX) > abs(dY):
                if dX > THRESHOLD:
                    direction = 'right'
                elif dX < -THRESHOLD:
                    direction = 'left'
            else:
                if dY > THRESHOLD:
                    direction = 'down'
                elif dY < -THRESHOLD:
                    direction = 'up'

        if direction != '' and direction != last_dir:
            pyautogui.press(direction)
            if is_debugging:
                print('Direction:', direction)
            last_dir = direction

        num_frames += 1

        cv2.putText(frame, direction, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        cv2.imshow('Game Control Window', frame)
        cv2.imshow('Mask', mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            is_running = False

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
    import time
    import multithreaded_webcam as mw
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    MODEL_PATH = 'hand_landmarker.task'
    num_hands = 1

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
        (0, 13), (13, 14), (14, 15), (15, 16),# Ring
        (0, 17), (17, 18), (18, 19), (19, 20) # Pinky
    ]

    time.sleep(2)
    vs = mw.WebcamVideoStream().start()

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

    def count_fingers(hand_landmarks):
        count = 0

        # Thumb (horizontal)
        if hand_landmarks[4].x > hand_landmarks[3].x:
            count += 1

        # Index finger
        if hand_landmarks[8].y < hand_landmarks[6].y:
            count += 1

        # Middle finger
        if hand_landmarks[12].y < hand_landmarks[10].y:
            count += 1

        # Ring finger
        if hand_landmarks[16].y < hand_landmarks[14].y:
            count += 1

        # Pinky
        if hand_landmarks[20].y < hand_landmarks[18].y:
            count += 1

        return count

    global last_dir
    is_running = True

    while is_running:
        img = vs.read()
        img = cv2.flip(img, 1)
        img = imutils.resize(img, width=600)
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        if result.hand_landmarks:
            h, w, _ = img.shape
            
            for hand_landmarks in result.hand_landmarks:
                
                for connection in HAND_CONNECTIONS:
                    start_idx = connection[0]
                    end_idx = connection[1]
                    
                    start_point = (int(hand_landmarks[start_idx].x * w), int(hand_landmarks[start_idx].y * h))
                    end_point = (int(hand_landmarks[end_idx].x * w), int(hand_landmarks[end_idx].y * h))
                    
                    cv2.line(img, start_point, end_point, (240, 240, 240), 2)

                wrist_coords = (0, 0)
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    if idx == 0: wrist_coords = (cx, cy)
                    
                    color = (0, 255, 0) if idx in [4, 8, 12, 16, 20] else (0, 0, 255)
                    cv2.circle(img, (cx, cy), 5, color, cv2.FILLED)

                count = count_fingers(hand_landmarks)
                cv2.putText(img, f"Count: {count}", (wrist_coords[0], wrist_coords[1] - 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                
        cv2.imshow('Finger Counter + Skeleton', img)
        cv2.waitKey(1)


    cv2.destroyAllWindows()
    vs.stop()
    landmarker.close()

def unique_control(is_debugging):
    """
    Controls a grid-based game using spacebar taps in a fun, unique way.

    Controls:
    - First spacebar press: starts moving forward.
    - Subsequent taps:
        - One tap  -> turn RIGHT
        - Two taps quickly -> turn LEFT

    The snake/Pacman keeps moving automatically; player just controls turning rhythmically.
    
    Parameters:
    is_debugging (bool): prints directions when True
    """
    import time
    from pynput import keyboard
    import pyautogui

    directions = ['up', 'right', 'down', 'left']
    current_idx = 0
    last_time = 0
    tap_count = 0
    started = False

    def press_direction(idx):
        pyautogui.press(directions[idx])
        if is_debugging:
            print("Direction:", directions[idx])

    def on_press(key):
        nonlocal current_idx, last_time, tap_count, started

        if key == keyboard.Key.space:
            now = time.time()

            if not started:
                started = True
                press_direction(current_idx)
                return

            if now - last_time < 0.4:
                tap_count += 1
            else:
                tap_count = 1

            if tap_count == 1:
                current_idx = (current_idx + 1) % 4
            elif tap_count == 2:
                current_idx = (current_idx - 1) % 4
                tap_count = 0

            press_direction(current_idx)
            last_time = now

        elif key == keyboard.Key.esc:
            print("Exiting unique control.")
            return False

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

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
