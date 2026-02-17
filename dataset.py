import os
import cv2
import argparse
import shutil

DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)
images_per_class = 100


gestures = [
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "q",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "x",
    "y"
]


def class_folder(idx, name):
    return f"{idx}_{name}"


def collect(class_path):
    cap = cv2.VideoCapture(0)
    counter = len(os.listdir(class_path))

    print(f"\nCollecting: {os.path.basename(class_path)}")
    print("Press Q to start | X to stop")

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    while counter < images_per_class:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        cv2.imwrite(os.path.join(class_path, f"{counter}.jpg"), frame)
        counter += 1

        cv2.imshow("Capture", frame)
        if cv2.waitKey(250) & 0xFF == ord("x"):
            break

    cap.release()
    cv2.destroyAllWindows()


def update_all():
    for idx, name in enumerate(gestures):
        folder = class_folder(idx, name)
        path = os.path.join(DATA_DIR, folder)
        os.makedirs(path, exist_ok=True)
        collect(path)


def update_one(name):
    if name not in gestures:
        print("Gesture not found in gestures list")
        return

    idx = gestures.index(name)
    folder = class_folder(idx, name)
    path = os.path.join(DATA_DIR, folder)
    os.makedirs(path, exist_ok=True)
    collect(path)


def delete_one(name):
    if name not in gestures:
        print("Gesture not found in gestures list")
        return

    idx = gestures.index(name)
    folder = class_folder(idx, name)
    path = os.path.join(DATA_DIR, folder)

    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Deleted: {folder}")
    else:
        print("Class folder does not exist")


def add_new_class(name):
    idx = get_next_class_index()

    folder = class_folder(idx, name)
    path = os.path.join(DATA_DIR, folder)

    if os.path.exists(path):
        print("Class folder already exists")
        return

    os.makedirs(path)
    print(f"Added new gesture: {folder}")
    collect(path)


def get_next_class_index():
    existing = []

    for name in os.listdir(DATA_DIR):
        if "_" in name:
            try:
                idx = int(name.split("_", 1)[0])
                existing.append(idx)
            except ValueError:
                pass

    return max(existing) + 1 if existing else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--class", dest="class_name")
    parser.add_argument("--delete", action="store_true")
    parser.add_argument("--add", action="store_true")

    args = parser.parse_args()

    if args.all:
        update_all()
    elif args.class_name:
        if args.delete:
            delete_one(args.class_name)
        elif args.add:
            add_new_class(args.class_name)
        else:
            update_one(args.class_name)
    else:
        print("Use --all or --class NAME")


if __name__ == "__main__":
    main()
