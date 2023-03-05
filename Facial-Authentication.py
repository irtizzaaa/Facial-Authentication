import cv2
import numpy as np
import face_recognition
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)

# load the known faces and their names from the local directory
known_face_encodings = []
known_face_names = []
for i in range(3):
    face_image = face_recognition.load_image_file(f"known_faces/face{i+1}.jpg")
    face_encoding = face_recognition.face_encodings(face_image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(f"Face {i+1}")

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # get the uploaded image from the form
        image = request.files["image"].read()

        # convert the image data to a numpy array
        image_data = np.frombuffer(image, dtype=np.uint8)

        # decode the image data as a color image
        image = cv2.imdecode(image_data, cv2.IMREAD_COLOR)

        # find all the faces in the image
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # loop through each face in the image
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # compare the face encoding to the known face encodings
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            # determine the closest match to a known face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                # if the face matches a known face, display the name of the face
                name = known_face_names[best_match_index]
                top, right, bottom, left = face_location
                cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(image, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # encode the image as a jpeg and return it as a response
        ret, jpeg = cv2.imencode(".jpg", image)
        return jpeg.tobytes()

    else:
        return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
