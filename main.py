import cv2
import face_recognition
import os
import numpy as np

known_face_encodings = []
known_face_names = []

# Find known_faces next to this Python file
folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "known " \
    "faces"
)

print("Known faces folder:")
print(folder)
print()

if not os.path.exists(folder):
    print("ERROR: known_faces folder does not exist!")
    exit()

for filename in os.listdir(folder):

    image_path = os.path.join(folder, filename)

    if not os.path.isfile(image_path):
        continue

    print("Loading:", filename)

    
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    if image is None:
        print("Could not read:", filename)
        continue

    
    rgb_image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    
    rgb_image = np.asarray(
        rgb_image,
        dtype=np.uint8
    )

    print("Format:", rgb_image.dtype, rgb_image.shape)

    try:
        encodings = face_recognition.face_encodings(
            rgb_image
        )

        if len(encodings) > 0:

            known_face_encodings.append(encodings[0])

            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

            print("SUCCESS:", name)

        else:
            print("NO FACE FOUND:", filename)

    except Exception as e:
        print("ERROR:", e)


print()
print("Known faces:", known_face_names)
print()



video = cv2.VideoCapture(0)

if not video.isOpened():
    print("ERROR: Could not open webcam.")
    exit()

print("Webcam started.")
print("Press Q to quit.")


while True:

    ret, frame = video.read()

    if not ret:
        break

    
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    
    face_encodings = face_recognition.face_encodings(
        rgb_frame,
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

        name = "Unknown"

        if True in matches:
            first_match = matches.index(True)
            name = known_face_names[first_match]

        top, right, bottom, left = face_location

        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            name,
            (left, bottom + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "Real-Time Face Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


video.release()
cv2.destroyAllWindows()