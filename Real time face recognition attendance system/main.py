import cv2
import face_recognition
import os
import csv
from datetime import datetime
import numpy as np
KNOWN_FACES_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "known faces"
)
ATTENDANCE_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "attendance.csv"
)
known_face_encodings = []
known_face_roll_numbers = []


print()
print("===================================")
print("       LOADING KNOWN FACES")
print("===================================")
print("Folder:", KNOWN_FACES_FOLDER)
print()


if not os.path.exists(KNOWN_FACES_FOLDER):

    print("ERROR: 'known faces' folder does not exist!")
    print("Create a folder named 'known faces' next to main.py")
    exit()
valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".webp"
)


for filename in os.listdir(KNOWN_FACES_FOLDER):

    image_path = os.path.join(
        KNOWN_FACES_FOLDER,
        filename
    )
    if not os.path.isfile(image_path):
        continue
    if not filename.lower().endswith(valid_extensions):
        continue


    print("Loading:", filename)

    roll_number = os.path.splitext(filename)[0]
    if not roll_number.isdigit():

        print(
            "SKIPPED:",
            filename,
            "-> filename must be a roll number"
        )

        continue

    image = cv2.imread(image_path)


    if image is None:

        print("Could not read:", filename)
        continue
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    face_locations = face_recognition.face_locations(
        rgb_image
    )


    if len(face_locations) == 0:

        print(
            "NO FACE FOUND:",
            filename
        )

        continue

    encodings = face_recognition.face_encodings(
        rgb_image,
        face_locations
    )


    if len(encodings) == 0:

        print(
            "Could not create encoding:",
            filename
        )

        continue
    encoding = encodings[0]
    known_face_encodings.append(encoding)
    known_face_roll_numbers.append(roll_number)


    print(
        "SUCCESS - Roll Number:",
        roll_number
    )


print()
print("===================================")
print(
    "Known students:",
    len(known_face_roll_numbers)
)
print(
    "Roll Numbers:",
    known_face_roll_numbers
)
print("===================================")
print()


if len(known_face_encodings) == 0:

    print("ERROR: No known faces were loaded.")
    exit()

def mark_attendance(roll_number):

    today = datetime.now().strftime("%Y-%m-%d")
    current_time = datetime.now().strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Roll Number",
                "Date",
                "Time",
                "Status"
            ])


    rows = []

    already_present = False


    with open(
        ATTENDANCE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            rows.append(row)

            if (
                row["Roll Number"] == roll_number
                and row["Date"] == today
            ):

                already_present = True

    if already_present:

        return False

    new_row = {
        "Roll Number": roll_number,
        "Date": today,
        "Time": current_time,
        "Status": "Present"
    }


    rows.append(new_row)

    rows.sort(
        key=lambda row: int(row["Roll Number"])
    )

    with open(
        ATTENDANCE_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        fieldnames = [
            "Roll Number",
            "Date",
            "Time",
            "Status"
        ]

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )


        writer.writeheader()


        writer.writerows(rows)

    print(
        f"ATTENDANCE MARKED: "
        f"Roll No. {roll_number} | "
        f"{today} | "
        f"{current_time}"
    )


    return True

print("Starting webcam...")


video = cv2.VideoCapture(0)


if not video.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Webcam started!")
print("Press Q to quit.")
print()

while True:

    ret, frame = video.read()


    if not ret:

        print(
            "ERROR: Could not read frame."
        )

        break


    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )
    rgb_small_frame = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    face_locations = face_recognition.face_locations(
        rgb_small_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_small_frame,
        face_locations
    )

    for face_encoding, face_location in zip(
        face_encodings,
        face_locations
    ):

        matches = face_recognition.compare_faces(
            known_face_encodings,
            face_encoding,
            tolerance=0.5
        )
        face_distances = face_recognition.face_distance(
            known_face_encodings,
            face_encoding
        )
        roll_number = "Unknown"

        if len(face_distances) > 0:

            best_match_index = np.argmin(
                face_distances
            )


            if matches[best_match_index]:

                roll_number = known_face_roll_numbers[
                    best_match_index
                ]

        if roll_number != "Unknown":

            mark_attendance(
                roll_number
            )

        top, right, bottom, left = face_location


        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        if roll_number == "Unknown":

            rectangle_color = (
                0,
                0,
                255
            )

        else:

            rectangle_color = (
                0,
                255,
                0
            )

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            rectangle_color,
            2
        )

        cv2.rectangle(
            frame,
            (left, bottom - 35),
            (right, bottom),
            rectangle_color,
            cv2.FILLED
        )

        font = cv2.FONT_HERSHEY_DUPLEX


        cv2.putText(
            frame,
            "Roll No: " + roll_number,
            (left + 6, bottom - 6),
            font,
            0.7,
            (255, 255, 255),
            1
        )

    cv2.putText(
        frame,
        "LIVE ATTENDANCE SYSTEM",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Press Q to quit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    cv2.imshow(
        "Face Recognition Attendance",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):

        break

video.release()

cv2.destroyAllWindows()


print()
print("===================================")
print("     ATTENDANCE SYSTEM CLOSED")
print("===================================")
print()
print("Attendance saved to:")
print(ATTENDANCE_FILE)
print()