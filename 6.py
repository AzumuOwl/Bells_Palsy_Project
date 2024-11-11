import cv2
import mediapipe as mp
import collections
import pygame  # Import pygame
import time

pygame.mixer.init()

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
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

def n6():
    pygame.mixer.music.load('S1/6.mp3')
    pygame.mixer.music.play()
# Open the camera
cap = cv2.VideoCapture(0)
n6()

# Threshold for detecting nose flare
nose_flare_threshold = 0.10  # Set threshold to 0.07
num_frames_for_avg = 10  # Number of frames for calculating the average

# Queue for storing nose width values per frame
nose_width_queue = collections.deque(maxlen=num_frames_for_avg)

# Variables for counting and status
count = 0
set_count = 0
is_above_threshold = False  # Used to track whether the width is above or below the threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert color to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Check if a face is detected
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmarks for the nose that can be used
            nose_tip = face_landmarks.landmark[1]  # Tip of the nose
            nose_left = face_landmarks.landmark[49]  # Left point of the nose
            nose_right = face_landmarks.landmark[279]  # Right point of the nose

            # Calculate nose width using the distance between left and right points
            nose_width = abs(nose_left.x - nose_right.x)

            # Add the nose width to the queue for averaging
            nose_width_queue.append(nose_width)

            # Calculate the average nose width over multiple frames
            avg_nose_width = sum(nose_width_queue) / len(nose_width_queue)

            # Display the threshold and average nose width
            cv2.putText(frame, f'Average Nose Width: {avg_nose_width:.4f}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Threshold: {nose_flare_threshold:.4f}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f'Count: {count}', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Sets: {set_count}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check the nose width status against the threshold
            if avg_nose_width > 0.0975 and avg_nose_width < 0.1:
                cv2.putText(frame, f"OK", (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
            if avg_nose_width > nose_flare_threshold:
                if not is_above_threshold:
                    # If the value just exceeded the threshold, increment the count
                    count += 1
                    is_above_threshold = True

                    # If the count reaches 5, increment set_count and reset count
                    if count >= 5:
                        set_count += 1
                        count = 0  # Reset the count

                    if set_count == 1 and count == 0:
                        set1()
                    elif set_count == 2 and count == 0:
                        set2()
                    elif set_count == 3 and count == 0:
                        set3()
                        time.sleep(3)
                        end = True
            else:
                # Reset the status when the value is below the threshold
                is_above_threshold = False

            # Draw landmarks of the nose
            h, w, _ = frame.shape
            cv2.circle(frame, (int(nose_tip.x * w), int(nose_tip.y * h)), 5, (255, 0, 0), -1)
            cv2.circle(frame, (int(nose_left.x * w), int(nose_left.y * h)), 5, (0, 255, 0), -1)
            cv2.circle(frame, (int(nose_right.x * w), int(nose_right.y * h)), 5, (0, 255, 0), -1)

    cv2.imshow('Nose Flare Detection', frame)

    if end == True:
        break
    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
