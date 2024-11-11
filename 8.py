import cv2
import mediapipe as mp
import pygame  # Import pygame
import time

# Set up MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
end = 0
pygame.mixer.init()

def set1():
    pygame.mixer.music.load('S1/set1.mp3')
    pygame.mixer.music.play()

def set2():
    pygame.mixer.music.load('S1/set2.mp3')
    pygame.mixer.music.play()

def set3():
    pygame.mixer.music.load('S1/set3.mp3')
    pygame.mixer.music.play()

def n8():
    pygame.mixer.music.load('S1/8.mp3')
    pygame.mixer.music.play()

# Smile detection function
def is_smiling(landmarks):
    # Face landmarks for lips
    upper_lip_top = landmarks[13]  # Top of upper lip
    lower_lip_bottom = landmarks[14]  # Bottom of lower lip

    # Calculate the distance between the upper and lower lips
    distance = lower_lip_bottom.y - upper_lip_top.y

    # Define a threshold to determine if smiling
    return distance > 0.02  # Adjust distance threshold as needed

# Start video capture
cap = cv2.VideoCapture(0)
n8()
smile_count = 0           # Smile counter
set_count = 0             # Set counter (increments every 5 smiles)
is_currently_smiling = False  # Current smile status
smile_reset = True        # Ensure a reset before counting next smile

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame color from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Smile detection
                smiling = is_smiling(face_landmarks.landmark)

                # Check smile counting condition
                if smiling and not is_currently_smiling and smile_reset:
                    # Count a smile when transitioning from not smiling to smiling
                    smile_count += 1
                    is_currently_smiling = True  # Update current smile status
                    smile_reset = False          # Require reset before next count
                    
                    # Check if smile count reaches 5
                    if smile_count >= 5:
                        set_count += 1      # Increment set count by 1
                        smile_count = 0     # Reset smile count to 0
                    if set_count == 1 and smile_count == 0:
                        set1()
                    elif set_count == 2 and smile_count == 0:
                        set2()
                    elif set_count == 3 and smile_count == 0:
                        set3()
                        time.sleep(3)
                        end = True
                elif not smiling:
                    # Reset smile status and allow next smile count
                    is_currently_smiling = False
                    smile_reset = True
                    cv2.putText(frame, "OK", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4)

                # Display text on screen
                cv2.putText(frame, f"Smile Count: {smile_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Set Count: {set_count}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if smiling:
                    cv2.putText(frame, "Smiling", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Not Smiling", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display video output
        cv2.imshow('Bells Palsy', frame)
        if end == True:
            break        
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
