import cv2
import mediapipe as mp
import numpy as np


def calculate_angles(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)

    return angle


class bodyRotation():
    def __init__(self, landmarks: list):
        try:
            self.LEFT_SHOULDER_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                                                            landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y], [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                                                                                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y], [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                                                                                                                                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                                                           )
        except:
            self.LEFT_SHOULDER_ROTATION = 90
        try:
            self.LEFT_ANKLE_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                                         landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y], [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                                                                                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y], [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].x,
                                                                                                                                                                    landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y])
        except:
            self.LEFT_ANKLE_ROTATION = 90
        try:
            self.LEFT_ELBOW_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y], [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                                                                                           landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y], [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                                                                                                                                                 landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y])
        except:
            self.LEFT_ELBOW_ROTATION = 90

        try:
            self.RIGHT_ANKLE_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                                                         landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                                                                                                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x,
                                                                                                                                                                      landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y])
        except:
            self.RIGHT_ANKLE_ROTATION = 90
        try:
            self.RIGHT_SHOULDER_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                                            landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                                                                                 landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                                                                                                                                           landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                                                            )
        except:
            self.RIGHT_SHOULDER_ROTATION = 90
        try:
            self.RIGHT_ELBOW_ROTATION = calculate_angles([landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y], [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                                                                                                              landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y], [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                                                                                                                                                     landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y])
        except:
            self.RIGHT_ELBOW_ROTATION = 90

    def all(self):
        self.left()
        self.right()

    def left(self):
        self.left_shoulder_r()
        self.left_elbow_r()
        self.left_ankle_r()

    def right(self):
        self.right_shoulder_r()
        self.right_elbow_r()
        self.right_ankle_r()

    def left_shoulder_r(self):
        print(
            f"LEFT_SHOULDER_ROTATION | {self.LEFT_SHOULDER_ROTATION}")

    def left_ankle_r(self):
        print(f"LEFT_ANKLE_ROTATION | {self.LEFT_ANKLE_ROTATION}")

    def left_elbow_r(self):
        print(f"LEFT_ELBOW_ROTATION | {self.LEFT_ELBOW_ROTATION}")

    def right_shoulder_r(self):
        print(
            f"RIGHT_SHOULDER_ROTATION | {self.RIGHT_SHOULDER_ROTATION}")

    def right_ankle_r(self):
        print(f"RIGHT_ANKLE_ROTATION | {self.RIGHT_ANKLE_ROTATION}")

    def right_elbow_r(self):
        print(f"RIGHT_ELBOW_ROTATION | {self.RIGHT_ELBOW_ROTATION}")


mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
address = "https://192.168.137.167:8080/video"
cap.open(address)
with mp_pose.Pose(min_detection_confidence=0.75, min_tracking_confidence=0.75) as pose:

    while cap.isOpened():
        ret, frame = cap.read()

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        try:
            landmarks = results.pose_landmarks.landmark
            body = bodyRotation(landmarks)

            body.all()

        except:
            pass

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS, mp_drawing.DrawingSpec(color=(
            245, 117, 66), thickness=2, circle_radius=2), mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        cv2.imshow('MP Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
