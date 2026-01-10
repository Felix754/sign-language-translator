import os
import cv2

DATA_DIR = './data'
os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 3
images_per_class = 100

cap = cv2.VideoCapture(0)

for class_id in range(number_of_classes):
    class_path = os.path.join(DATA_DIR, str(class_id))
    os.makedirs(class_path, exist_ok=True)
    
    print(f"\nReady to collect data for class {class_id}. Press 'Q' to begin.")

    while True:
        ret, frame = cap.read()
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Flip horizontally for mirror view
        cv2.putText(frame, f"Class {class_id} - Press Q to start", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Data Collection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    counter = 0
    while counter < images_per_class:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow('Data Collection', frame)
        cv2.imwrite(os.path.join(class_path, f'{counter}.jpg'), frame)
        counter += 1
        if cv2.waitKey(250) & 0xFF == ord('x'):
            break

cap.release()
cv2.destroyAllWindows()