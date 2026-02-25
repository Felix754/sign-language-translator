import cv2
import mediapipe as mp
import pickle
import numpy as np
import time
from collections import deque, Counter

# ---------------- SETTINGS ----------------
CONF_THRESHOLD = 0.65
SMOOTHING_FRAMES = 8
STABLE_RATIO = 0.6
ADD_DELAY = 1.0
REPEAT_DELAY = 1.0
BAR_COLOR = (0, 200, 255)

# Dynamic signs
MOVE_THRESHOLD = 0.05
MOVE_FRAMES = 6
# ------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def clean_label(raw_label):
    if "_" in raw_label:
        return raw_label.split("_", 1)[1]
    return raw_label


def angle(a, b, c):
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return np.arccos(np.clip(cosang, -1.0, 1.0))


def resize_with_aspect_ratio(frame, target_w, target_h):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas


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
cv2.namedWindow("ASL", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ASL", 1000, 700)

prediction_history = deque(maxlen=SMOOTHING_FRAMES)
stable_prediction = ""
smooth_conf = 0.0

typed_text = ""
last_added = ""
last_add_time = 0
last_repeat_time = 0

# Cursor
cursor_visible = True
last_cursor_toggle = time.time()
CURSOR_BLINK_DELAY = 0.5

# Movement detection
movement_buffer = []
movement_detected = False

#Main hand recognition loop
with mp_hands.Hands(max_num_hands=1) as hands:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

       
        if time.time() - last_cursor_toggle > CURSOR_BLINK_DELAY:
            cursor_visible = not cursor_visible
            last_cursor_toggle = time.time()

        
        if result.multi_hand_landmarks:

            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

  
            wrist = hand.landmark[0]
            current_pos = np.array([wrist.x, wrist.y])

            movement_buffer.append(current_pos)

            if len(movement_buffer) > MOVE_FRAMES:
                movement_buffer.pop(0)

            if len(movement_buffer) == MOVE_FRAMES:
                dist = np.linalg.norm(movement_buffer[-1] - movement_buffer[0])
                movement_detected = dist > MOVE_THRESHOLD

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
            raw_prediction = model.classes_[probs.argmax()]
            current_conf = max(probs)

            if current_conf >= CONF_THRESHOLD:
                prediction_history.append(raw_prediction)

            if len(prediction_history) == SMOOTHING_FRAMES:
                most_common = Counter(prediction_history).most_common(1)[0]

                if most_common[1] / SMOOTHING_FRAMES >= STABLE_RATIO:
                    stable_prediction = most_common[0]
                    smooth_conf = current_conf

                    label = clean_label(stable_prediction).upper()

                    # Dynamic J & Z 
                    if label == "I" and movement_detected:
                        label = "J"
                    elif label == "D" and movement_detected:
                        label = "Z"

                    label_lower = label.lower()
                    current_time = time.time()

                    
                    if label_lower == "backspace":
                        if current_time - last_repeat_time > REPEAT_DELAY:
                            if len(typed_text) > 0:
                                typed_text = typed_text[:-1]
                            last_repeat_time = current_time

            
                    elif label_lower == "space":
                        if (
                            label != last_added and
                            current_time - last_add_time > ADD_DELAY
                        ):
                            typed_text += " "
                            last_added = label
                            last_add_time = current_time

                    else:
                        if (
                            label != last_added and
                            current_time - last_add_time > ADD_DELAY
                        ):
                            typed_text += label
                            last_added = label
                            last_add_time = current_time

        # No hand detected
        else:
            prediction_history.clear()
            stable_prediction = ""
            smooth_conf = 0.0
            last_added = ""
            last_repeat_time = 0
            movement_buffer.clear()
            movement_detected = False

        # UI part
        overlay_dark = frame.copy()
        cv2.rectangle(overlay_dark, (0, 0),
                      (frame.shape[1], frame.shape[0]),
                      (20, 20, 20), -1)
        frame = cv2.addWeighted(overlay_dark, 0.3, frame, 0.7, 0)

        h, w = frame.shape[:2]

        # Top panel
        panel_height = 80
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_height),
                      (10, 10, 10), -1)
        frame = cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)

        display_label = clean_label(stable_prediction).upper()

        cv2.putText(frame, display_label,
                    (w // 2 - 20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA)

        # Confidence bar
        bar_width = int(w * 0.5)
        bar_height = 10
        bar_x = (w - bar_width) // 2
        bar_y = 60

        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + bar_width, bar_y + bar_height),
                      (60, 60, 60), -1)

        fill_width = int(bar_width * smooth_conf)

        cv2.rectangle(frame,
                      (bar_x, bar_y),
                      (bar_x + fill_width, bar_y + bar_height),
                      BAR_COLOR, -1)

        # Bottom text
        bottom_height = 70
        overlay2 = frame.copy()
        cv2.rectangle(overlay2,
                      (0, h - bottom_height),
                      (w, h),
                      (15, 15, 15), -1)

        frame = cv2.addWeighted(overlay2, 0.95, frame, 0.05, 0)

        cursor = "|" if cursor_visible else ""
        display_text = typed_text[-80:] + cursor

        cv2.putText(frame,
                    display_text,
                    (20, h - 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA)

        # Resize window
        try:
            window_rect = cv2.getWindowImageRect("ASL")
            frame_resized = resize_with_aspect_ratio(
                frame,
                window_rect[2],
                window_rect[3]
            )
        except:
            break

        cv2.imshow("ASL", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if cv2.getWindowProperty("ASL", cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
