import cv2
import mediapipe as mp
import pyautogui
import numpy as np

pyautogui.FAILSAFE = False

# Function to move the mouse cursor based on the average pupil positions
def move_mouse(left, right, left_bounds, right_bounds):
    screen_width, screen_height = pyautogui.size()

    left_percent = (left - left_bounds[0]) / left_bounds[1]
    right_percent = (right - right_bounds[0]) / right_bounds[1]

    avg_percent = np.mean([left_percent, right_percent])
    mouse_x = int(avg_percent * screen_width)
    mouse_y = int(avg_percent * screen_height)
    print(f'mouse_x: {mouse_x}, mouse_y: {mouse_y}', end='\n')
    print(f'left_percent: {left_percent}, right_percent: {right_percent}', end='\n')
    # invert because video is flipped vertically by default on modern selfie cameras
    pyautogui.moveTo(mouse_x, mouse_y, duration=0.1)

    # left_x = left_bounds[0].x - avg_left.x
    # right_x = right_bounds[0].x - avg_right.x
    # avg_x = (avg_left.x + avg_right.x) / 2
    # avg_y = (avg_left.y + avg_right.y) / 2
    # mouse_x = int(avg_x * screen_width)
    # mouse_y = int(avg_y * screen_height)
    # # invert because video is flipped vertically by default on modern selfie cameras
    # pyautogui.moveTo(screen_width - mouse_x, mouse_y, duration=0.1)

def detect_pupils_and_control_mouse(every_n_frames=0):
    mp_face_mesh = mp.solutions.face_mesh

    # Initialize MediaPipe Face Mesh.
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # Open the default camera
    cap = cv2.VideoCapture(0)

    # Set camera resolution (adjust these values based on your camera's capabilities)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # Indices for the eye landmarks (including irises)
    # Original
    # left_eye_landmarks = [33, 133, 144, 145, 153, 154, 155, 159, 160, 161, 163, 173]
    # left_eye_landmarks = list(range(25, 180))
    left_left = 33
    left_right = 133 # possibly 155
    # Original
    # right_eye_landmarks = [362, 382, 384, 385, 386, 387, 388, 390, 398]
    # right_eye_landmarks = list(range(360, 370))
    # right_eye_landmarks = [362, 382, 384, 385, 386, 387, 388, 390, 398]
    right_right = 359 # possibly 388
    right_left = 362

    left_iris_landmarks = [468, 469, 470, 471]
    right_iris_landmarks = [473, 474, 475, 476]

    frame_count = 0  # Counter to track frames for processing

    while cap.isOpened():
        # Read every nth frame (e.g., every 3rd frame)
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if every_n_frames and frame_count % every_n_frames != 0:
            continue  # Skip frames to improve processing speed

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Process the image and detect face mesh landmarks.
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmarks for the left and right irises
                left_iris_points = []
                right_iris_points = []
                for idx in left_iris_landmarks:
                    left_iris_points.append((face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y))
                for idx in right_iris_landmarks:
                    right_iris_points.append((face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y))

                # Calculate average position of left iris (pupil)
                avg_left = np.mean(left_iris_points, axis=0)
                avg_right = np.mean(right_iris_points, axis=0)

                left_bounds  = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) for idx in [left_left, left_right]]
                right_bounds = [np.array([face_landmarks.landmark[idx].x, face_landmarks.landmark[idx].y]) for idx in [right_left, right_right]]

                # Move the mouse cursor based on average pupil positions
                move_mouse(avg_left, avg_right, left_bounds, right_bounds)

                # Draw landmarks for the left eye and iris
                for idx in [left_left, left_right, right_left, right_right]:
                    x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                    y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                for idx in [avg_left, avg_right]:
                    x = int(idx[0] * frame.shape[1])
                    y = int(idx[1] * frame.shape[0])
                    cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                    # Also show the index of the landmark
                    # cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                # for idx in left_iris_landmarks:
                #     x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                #     y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                #     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                #     # Also show the index of the landmark
                #     cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                # # Draw landmarks for the right eye and iris
                # for idx in [left_left, left_right, right_left, right_right]:
                #     x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                #     y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                #     cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                #     # Also show the index of the landmark
                #     cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                # # for idx in right_iris_landmarks:
                #     x = int(face_landmarks.landmark[idx].x * frame.shape[1])
                #     y = int(face_landmarks.landmark[idx].y * frame.shape[0])
                #     cv2.circle(frame, (x, y), 2, (0, 0, 255), -1)
                #     # Also show the index of the landmark
                #     cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Display the frame.
        cv2.imshow('Eye and Pupil Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_pupils_and_control_mouse()
