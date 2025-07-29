import cv2
import dlib
import imutils
import numpy as np
from scipy.spatial import distance
from imutils import face_utils
import pygame
from datetime import datetime

# Initialize pygame mixer and load alert sound
pygame.mixer.init()
pygame.mixer.music.load("music.wav")

# Eye Aspect Ratio (EAR) function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Constants
THRESH = 0.20
FRAME_CHECK = 20
flag = 0
blink_counter = 0
eye_closed_frames = 0
WELCOME_SHOWN = False

# Load dlib face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

# Get indexes for eyes from dlib shape
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab frame.")
        break

    frame = imutils.resize(frame, width=600)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    # Welcome message if face is detected for first time
    if len(faces) > 0 and not WELCOME_SHOWN:
        cv2.putText(frame, "Welcome! Face detected!", (150, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 3)
        WELCOME_SHOWN = True

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # Draw contours around eyes
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

        # EAR and alert logic
        status = "Status: Awake"
        status_color = (0, 255, 0)

        if ear < THRESH:
            flag += 1
            eye_closed_frames += 1
            if flag >= FRAME_CHECK:
                status = "Status: ALERT!"
                status_color = (0, 0, 255)
                cv2.putText(frame, "****************ALERT!****************", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.play()
            else:
                status = "Status: Drowsy"
                status_color = (0, 255, 255)
        else:
            if 1 <= eye_closed_frames < 3:
                blink_counter += 1
            flag = 0
            eye_closed_frames = 0

        # Display EAR, blinks, and status
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Blinks: {blink_counter}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 255, 255), 2)
        cv2.putText(frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, status_color, 2)

    # Timestamp at bottom
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(frame, timestamp, (400, 460), cv2.FONT_HERSHEY_PLAIN,
                1, (200, 200, 200), 1)

    # Always show the frame
    cv2.imshow("Drowsiness Detector", frame)
    key = cv2.waitKey(2) & 0xFF
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
