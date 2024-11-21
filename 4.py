import cv2
import mediapipe as mp
import numpy as np
import time
import pygame

# Initialize Pygame mixer for playing sounds
pygame.mixer.init()

# Set up Mediapipe for face mesh detection
mp_face_mesh = mp.solutions.face_mesh

# Define landmarks for left and right eyes
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_EAR(eye_landmarks):
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    EAR = (A + B) / (2.0 * C)
    return EAR

# Function to play sound
def play_sound(sound_path):
    pygame.mixer.music.load(sound_path)
    pygame.mixer.music.play()

# Sound functions for different stages
def notified2():
    play_sound('S1/notified2.mp3')

def set1():
    play_sound('S1/set1.mp3')

def set2():
    play_sound('S1/set2.mp3')

def set3():
    play_sound('S1/set3.mp3')

def n4():
    play_sound('S1/4.mp3')

# Initialize the camera
cap = cv2.VideoCapture(0)
end = False

# Blink count and set count variables
blink_count = 0
set_count = 0
EAR_THRESHOLD = 0.21
blink_in_progress = False  # Track if a blink is in progress

n4()  # Initial notification sound

# Start Mediapipe face mesh detection
with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Unable to access the camera.")
            break

        # Flip the image horizontally for a selfie view
        image = cv2.flip(image, 1)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to find face landmarks
        results = face_mesh.process(rgb_image)
        
        # Check if landmarks were found
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract left and right eye landmarks
                left_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in LEFT_EYE_LANDMARKS]
                right_eye = [(face_landmarks.landmark[i].x, face_landmarks.landmark[i].y) for i in RIGHT_EYE_LANDMARKS]
                
                # Calculate average EAR for both eyes
                left_EAR = calculate_EAR(left_eye)
                right_EAR = calculate_EAR(right_eye)
                avg_EAR = (left_EAR + right_EAR) / 2.0

                # Detect blink when EAR is below the threshold and no blink is in progress
                if avg_EAR < EAR_THRESHOLD and not blink_in_progress:
                    blink_in_progress = True  # Mark blink as in progress
                    blink_count += 1
                elif avg_EAR >= EAR_THRESHOLD and blink_in_progress:
                    blink_in_progress = False  # Reset blink progress when eyes are open
                else:
                    cv2.putText(image, f"OK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

                # Reset blink count after reaching 3 blinks and increase the set count
                if blink_count >= 3:
                    set_count += 1
                    blink_count = 0
                    if set_count == 1:
                        set1()  # Play sound for set 1
                    elif set_count == 2:
                        set2()  # Play sound for set 2
                    elif set_count == 3:
                        set3()  # Play sound for set 3
                        time.sleep(3)  # Pause for 3 seconds
                        end = True  # End the program after set 3
        
        # Display the blink and set counts on the screen
        cv2.putText(image, f"Blink Count: {blink_count}", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(image, f"Set Count: {set_count}", (30, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # End program if `end` is True
        if end:
            break
        
        # Show the image
        cv2.imshow('Bells Palsy', image)
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
