import mediapipe as mp
import pandas as pd
import numpy as np

mp_hands = mp.solutions.hands

def mediapipe_inference(frame_iterator, fps,
                                 min_detection_conf=0.3, min_tracking_conf=0.3,
                                 max_num_hands=2):
    results_list = []

    with mp_hands.Hands(static_image_mode=False,
                        max_num_hands=max_num_hands,
                        min_detection_confidence=min_detection_conf,
                        min_tracking_confidence=min_tracking_conf) as hands:

        for frame_idx, frame in frame_iterator:
            timestamp = frame_idx / fps
            result = hands.process(frame)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
                    hand_label = result.multi_handedness[hand_idx].classification[0].label  # 'Left' or 'Right'

                    for j, lm in enumerate(hand_landmarks.landmark):
                        results_list.append({
                            'frame_id': frame_idx,
                            'timestamp_s': timestamp,
                            'hand_id': hand_label,
                            'joint_id': j,
                            'x': lm.x,
                            'y': lm.y,
                            'z': lm.z
                        })
            else:
                # if no hand detected, fill NaN entries (optional)
                for hand_label in ["Left", "Right"]:
                    for j in range(21):
                        results_list.append({
                            'frame_id': frame_idx,
                            'timestamp_s': timestamp,
                            'hand_id': hand_label,
                            'joint_id': j,
                            'x': np.nan,
                            'y': np.nan,
                            'z': np.nan
                        })

    df = pd.DataFrame(results_list)
    return df