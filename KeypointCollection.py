import cv2
import numpy as np
import os
import mediapipe as mp
import json

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

TRAIN_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MP_Data')
VAL_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MP_Data_Val')
TRAIN_VIDEO_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos_train')
VAL_VIDEO_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'videos_val')

with open("MSASL_classes.json", "r") as file:
    classes = json.load(file)   
    classes = classes[:-600]
    file.close()

actions = np.array(classes)
num_sequences = 30
fps = 30

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results != 0 and results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results != 0 and results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results != 0 and results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results != 0 and results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def FolderCreator(PATH):
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    for action in actions:
        for sequence in range(num_sequences):
            try:
                os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating directories: {e}")

            try:
                os.makedirs(os.path.join(PATH, action, str(sequence)), exist_ok=True)
            except Exception as e:
                print(f"An error occurred while creating directories: {e}")

def DataCollecter(DATA_PATH, VIDEO_PATH):
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for action in actions:
            for sequence in range(num_sequences):
                cap = cv2.VideoCapture(os.path.join(VIDEO_PATH, action, f'{sequence}.mp4'))

                frames_to_save = np.linspace(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-1, fps).astype(int)
                added_frames = 0

                for frame_num in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
                    ret, frame = cap.read()

                    image, results = mediapipe_detection(frame, holistic)
                    draw_landmarks(image, results)


                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

                    cv2.imshow('OpenCV Feed', image)

                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(added_frames))

                    if frame_num in frames_to_save:
                        np.save(npy_path, keypoints)
                        added_frames += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Creating folders...")
    FolderCreator(TRAIN_DATA_PATH)
    FolderCreator(VAL_DATA_PATH)
    print("Starting Data Collection...")
    DataCollecter(TRAIN_DATA_PATH, TRAIN_VIDEO_PATH)
    DataCollecter(VAL_DATA_PATH, VAL_VIDEO_PATH)
    print("Data Collection Completed")