import cv2
import mediapipe as mp
import pygame  # Import pygame
import time

# Initialize pygame mixer for playing sound
pygame.mixer.init()
# Initialize Mediapipe variables
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
end = 0

def notified2():
    # Load and play sound on button press
    pygame.mixer.music.load('S1/notified2.mp3')  # Replace 'S1/notified1.mp3' with your actual file path
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

def n5():
    pygame.mixer.music.load('S1/5.mp3')
    pygame.mixer.music.play()

# Function to calculate mouth width
def calculate_mouth_width(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        mouth_left = landmarks.landmark[61]  # Left point of the mouth
        mouth_right = landmarks.landmark[291]  # Right point of the mouth

        width = ((mouth_right.x - mouth_left.x) ** 2 + (mouth_right.y - mouth_left.y) ** 2) ** 0.5
        return width
    return None

# Start capturing video from the camera
cap = cv2.VideoCapture(0)
n5()
count = 0  # Variable to count occurrences where mouth width is less than or equal to 0.07
is_below_threshold = False  # Variable to track if the width is below the threshold
total_sets = 0  # Variable to store the count of completed sets of 10
end = False

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Calculate the width of the mouth
    mouth_width = calculate_mouth_width(frame)

    # Check and update the counter
    if mouth_width is not None:
        if mouth_width <= 0.08:
            if not is_below_threshold:  # If not already below the threshold
                count += 1  # Increase the count
                is_below_threshold = True  # Set the below-threshold flag

                # Check if the count reaches 10
                if count == 10:
                    total_sets += 1  # Increment the set count
                    count = 0  # Reset the count

                if total_sets == 1 and count == 0:
                    set1()
                elif total_sets == 2 and count == 0:
                    set2()
                elif total_sets == 3 and count == 0:
                    set3()
                    time.sleep(3)
                    end = True

        else:
            is_below_threshold = False  # Reset the flag when width is above the threshold

        # Display results
        cv2.putText(frame, f'Mouth Width: {mouth_width:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Count: {count}', (frame.shape[1] - 150, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Sets: {total_sets}', (frame.shape[1] - 150, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    if end == True:
        break
    cv2.imshow('Bells Palsy', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
