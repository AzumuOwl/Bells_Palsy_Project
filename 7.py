import cv2
import mediapipe as mp
import math
import numpy as np
import pygame  # Import pygame
import time
# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
end = 0

def set1():
    pygame.mixer.music.load('S1/set1.mp3')
    pygame.mixer.music.play()

def set2():
    pygame.mixer.music.load('S1/set2.mp3')
    pygame.mixer.music.play()

def set3():
    pygame.mixer.music.load('S1/set3.mp3')
    pygame.mixer.music.play()

def n7():
    pygame.mixer.music.load('S1/7.mp3')
    pygame.mixer.music.play()

# Function to calculate distance between two landmarks
def calculate_distance(landmark1, landmark2):
    x1, y1 = landmark1
    x2, y2 = landmark2
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# Parameters for smoothing and counting
distance_history = []
max_history_length = 10  # Number of frames to average over
count = 0  # Counter for occurrences
set_count = 0  # Counter for sets
threshold = 80  # Distance threshold
above_threshold = False  # State flag for crossing the threshold
max_count = 5  # Maximum count before increasing set

# Start video capture
cap = cv2.VideoCapture(0)
n7()
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    # Convert the BGR image to RGB before processing
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find facial landmarks
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = image.shape
            # Get the coordinates of landmark 57 and 287
            landmark_57 = face_landmarks.landmark[57]
            landmark_287 = face_landmarks.landmark[287]
            
            x1, y1 = int(landmark_57.x * w), int(landmark_57.y * h)
            x2, y2 = int(landmark_287.x * w), int(landmark_287.y * h)

            # Draw landmarks 57 and 287 on the image
            cv2.circle(image, (x1, y1), 5, (0, 255, 0), -1)
            cv2.circle(image, (x2, y2), 5, (0, 255, 0), -1)

            # Calculate the distance and add it to the history
            distance = calculate_distance((x1, y1), (x2, y2))
            distance_history.append(distance)

            # Keep history length within max_history_length
            if len(distance_history) > max_history_length:
                distance_history.pop(0)

            # Calculate the average distance for smoothing
            smoothed_distance = np.mean(distance_history)
            cv2.putText(image, f'Distance: {int(smoothed_distance)} pixels', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            if smoothed_distance > 75 and smoothed_distance <80 :
                cv2.putText(image, f"OK", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
            # Check for threshold crossing
            if smoothed_distance > threshold and not above_threshold:
                count += 1
                above_threshold = True
            elif smoothed_distance < threshold:
                above_threshold = False

            # Check if count has reached max_count
            if count >= max_count:
                set_count += 1
                count = 0  # Reset count for new set
            if set_count == 1 and count == 0:
                set1()
            elif set_count == 2 and count == 0:
                set2()
            elif set_count == 3 and count == 0:
                set3()
                time.sleep(3)
                end = True
            # Display count and set count
            cv2.putText(image, f'Count: {count}', (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f'Sets: {set_count}', (10, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image with landmarks, distance, count, and set count
    cv2.imshow('Bells Palsy', image)
    if end == True:
        break
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
