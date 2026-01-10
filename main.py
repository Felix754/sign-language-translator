import cv2
import mediapipe as mp
import pickle
import numpy as np

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba)*np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))

FINGERS = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16],
    [17, 18, 19, 20]
]

with open("asl_model.pkl", "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

with mp_hands.Hands(max_num_hands=1) as hands:
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            lm = hand.landmark
            features = []

            for finger in FINGERS:
                a, b, c, d = finger

                features.append(angle(
                    [lm[a].x, lm[a].y, lm[a].z],
                    [lm[b].x, lm[b].y, lm[b].z],
                    [lm[c].x, lm[c].y, lm[c].z]
                ))

                features.append(angle(
                    [lm[b].x, lm[b].y, lm[b].z],
                    [lm[c].x, lm[c].y, lm[c].z],
                    [lm[d].x, lm[d].y, lm[d].z]
                ))

            probs = model.predict_proba([features])[0]
            prediction = model.classes_[probs.argmax()]
            conf = max(probs)

            cv2.putText(frame, f"{prediction} ({conf:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

        cv2.imshow("ASL", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()
