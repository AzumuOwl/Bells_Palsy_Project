import cv2
import mediapipe as mp
import numpy as np
import time
import pygame  # Import pygame

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Define eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def notified2():
    pygame.mixer.music.load('S1/notified2.mp3')
    pygame.mixer.music.play()

def set1():
    pygame.mixer.music.load('S1/set1.mp3')
    pygame.mixer.music.play()

def set2():
    pygame.mixer.music.load('S1/set2.mp3')
    pygame.mixer.music.play()

def set3():
    pygame.mixer.music.load('S1/set3.mp3')
    pygame.mixer.music.play()

def n4():
    pygame.mixer.music.load('S1/4.mp3')
    pygame.mixer.music.play()

# Function to calculate Eye Aspect Ratio (EAR)
def calculate_ear(landmarks, eye_points):
    A = np.linalg.norm(np.array([landmarks[eye_points[1]].x, landmarks[eye_points[1]].y]) - 
                       np.array([landmarks[eye_points[5]].x, landmarks[eye_points[5]].y]))
    B = np.linalg.norm(np.array([landmarks[eye_points[2]].x, landmarks[eye_points[2]].y]) - 
                       np.array([landmarks[eye_points[4]].x, landmarks[eye_points[4]].y]))
    C = np.linalg.norm(np.array([landmarks[eye_points[0]].x, landmarks[eye_points[0]].y]) - 
                       np.array([landmarks[eye_points[3]].x, landmarks[eye_points[3]].y]))

    ear = (A + B) / (2.0 * C)
    return ear

# Threshold for blinking
EAR_THRESHOLD = 0.25
closed_eyes_counter = 0
set_counter = 0
eyes_closed = False
end = False
# Open webcam
cap = cv2.VideoCapture(0)
n4()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame color to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            
            # Calculate EAR for both eyes
            left_ear = calculate_ear(landmarks, LEFT_EYE)
            right_ear = calculate_ear(landmarks, RIGHT_EYE)
            ear = (left_ear + right_ear) / 2.0
            if ear < EAR_THRESHOLD:
                #time.sleep(1)
                if not eyes_closed:
                    closed_eyes_counter += 1
                    eyes_closed = True
                    if closed_eyes_counter == 5:
                        set_counter += 1
                        closed_eyes_counter = 0

                        if set_counter == 1 and closed_eyes_counter == 0:
                            set1()
                        elif set_counter == 2 and closed_eyes_counter == 0:
                            set2()
                        elif set_counter == 3 and closed_eyes_counter == 0:
                            set3()
                            time.sleep(3)
                            end = True
            else:
                eyes_closed = False
                cv2.putText(frame, f"OK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)

    # Display the counters on the screen
    cv2.putText(frame, f"Blink Count: {closed_eyes_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Set Count: {set_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if end == True:
        break
    # Display the frame
    cv2.imshow("Bells Palsy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
