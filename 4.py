import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize mediapipe face mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils

# Define eye landmarks
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

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

# Threshold for blinking (This value might need tuning)
EAR_THRESHOLD = 0.25
closed_eyes_counter = 0
set_counter = 0
eyes_closed = False  # Flag to indicate if eyes are currently closed
start_time = None  # To record time when eyes are first closed

# Open webcam
cap = cv2.VideoCapture(0)

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
            
            # Check if both eyes are closed
            if left_ear < EAR_THRESHOLD and right_ear < EAR_THRESHOLD:
                if not eyes_closed:
                    start_time = time.time()  # Start the timer
                    eyes_closed = True
                elif time.time() - start_time > 1:  # Check if 2 seconds have passed
                    closed_eyes_counter += 1
                    eyes_closed = False  # Reset flag for the next blink
                    start_time = None  # Reset timer

                    # Check if count reaches 10
                    if closed_eyes_counter >= 10:
                        set_counter += 1
                        closed_eyes_counter = 0  # Reset closed eyes counter for new set
            else:
                eyes_closed = False  # Reset flag when eyes are open
                start_time = None  # Reset timer

    # Display the counter on the screen
    cv2.putText(frame, f"Closed Eyes Count: {closed_eyes_counter}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Set Count: {set_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the frame
    cv2.imshow("Bells Palsy", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
