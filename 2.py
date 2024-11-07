import cv2
import mediapipe as mp
import math
import pygame  # Import pygame for sound playing
import time

# Initialize pygame mixer for playing sound
pygame.mixer.init()

# Initialize MediaPipe Face Mesh solution
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)

# Sound functions
def play_sound(file_path):
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()

def notified2():
    play_sound('S1/notified2.mp3')

def set1():
    play_sound('S1/set1.mp3')

def set2():
    play_sound('S1/set2.mp3')

def set3():
    play_sound('S1/set3.mp3')

def n2():
    play_sound('S1/2.mp3')

# Function to calculate the distance between two points
def calculate_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Values for calculating a moving average
distance_buffer = []
buffer_size = 10  # Buffer size for moving average calculation

# Variables to count occurrences where distance <= 72 and to count sets
counter = 0
set_counter = 0
threshold_distance = 135  # Distance threshold for checking
below_threshold = False  # Status to check if the distance has gone below threshold before
end = False

# Start camera
cap = cv2.VideoCapture(0)
n2()

while cap.isOpened() :
    success, image = cap.read()
    if not success:
        print("Unable to read image from the camera.")
        break

    # Convert the image color from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_rgb)

    # Check if a face is detected
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Landmark points for eyebrows
            left_eyebrow = face_landmarks.landmark[105]
            right_eyebrow = face_landmarks.landmark[334]

            # Calculate the position of each eyebrow point in pixels
            image_height, image_width, _ = image.shape
            left_eyebrow_pos = (int(left_eyebrow.x * image_width), int(left_eyebrow.y * image_height))
            right_eyebrow_pos = (int(right_eyebrow.x * image_width), int(right_eyebrow.y * image_height))
            
            # Calculate the distance and update the buffer
            distance = calculate_distance(left_eyebrow_pos, right_eyebrow_pos)
            distance_buffer.append(distance)
            if len(distance_buffer) > buffer_size:
                distance_buffer.pop(0)

            # Calculate the average distance
            avg_distance = sum(distance_buffer) / len(distance_buffer)
            if avg_distance > 135 and avg_distance <140 :
                cv2.putText(image, f"OK", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 4)
            
            # Check the average distance against the threshold
            if avg_distance <= threshold_distance:
                if not below_threshold:
                    counter += 1
                    below_threshold = True

                    if counter == 3:
                        set_counter += 1
                        counter = 0
                        if set_counter == 1 and counter == 0:
                            set1()
                        elif set_counter == 2 and counter == 0:
                            set2()
                        elif set_counter == 3 and counter == 0:
                            set3()
                            time.sleep(3)
                            end = True
            else:
                below_threshold = False

            # Display the average distance, counter, and set count
            cv2.putText(image, f"Avg Distance: {int(avg_distance)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(image, f"Counter: {counter}", (image_width - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(image, f"Set: {set_counter}", (image_width - 150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw circles on both eyebrows
            cv2.circle(image, left_eyebrow_pos, 5, (255, 0, 0), -1)
            cv2.circle(image, right_eyebrow_pos, 5, (255, 0, 0), -1)

    if end == True:
        break
    # Show the image
    cv2.imshow("Bells Palsy", image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release camera and close windows
cap.release()
cv2.destroyAllWindows()
