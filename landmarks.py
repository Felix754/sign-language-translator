import mediapipe as mp
import pandas as pd
import cv2
import os
import numpy as np

DATA_DIR = './data'

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

# (MCP, PIP, DIP, TIP)
FINGERS = [
    [1, 2, 3, 4],     # thumb
    [5, 6, 7, 8],     # index
    [9, 10, 11, 12],  # middle
    [13, 14, 15, 16], # ring
    [17, 18, 19, 20]  # pinky
]

data = []
labels = []

for label in os.listdir(DATA_DIR):
    for img_name in os.listdir(os.path.join(DATA_DIR, label)):
        img_path = os.path.join(DATA_DIR, label, img_name)
        image = cv2.imread(img_path)
        if image is None:
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            continue

        hand = result.multi_hand_landmarks[0]
        lm = hand.landmark

        angles = []

        for a, b, c, d in FINGERS:
            # MCP-PIP-DIP
            angles.append(angle(
                [lm[a].x, lm[a].y, lm[a].z],
                [lm[b].x, lm[b].y, lm[b].z],
                [lm[c].x, lm[c].y, lm[c].z]
            ))
            # PIP-DIP-TIP
            angles.append(angle(
                [lm[b].x, lm[b].y, lm[b].z],
                [lm[c].x, lm[c].y, lm[c].z],
                [lm[d].x, lm[d].y, lm[d].z]
            ))

        data.append(angles)
        labels.append(label)

# ✅ 10 кутів, не 20
columns = [f"angle_{i}" for i in range(10)]
df = pd.DataFrame(data, columns=columns)
df["label"] = labels

df.to_csv("asl_angles.csv", index=False)
print("Saved asl_angles.csv with 10 angle features")
