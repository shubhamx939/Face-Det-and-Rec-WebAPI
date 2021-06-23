import numpy as np
import os
import cv2
from pathlib import Path
import click
import re
import multiprocessing
import itertools
import sys
import PIL.Image

import face_recognition
from flask import Flask, jsonify, request, redirect




known_faces_path = Path("./known/")


def image_files_in_folder(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png)', f, flags=re.I)]


def scan_known_people(known_faces_dir):
    known_names = []
    known_face_encodings = []

    for file in image_files_in_folder(known_faces_dir):
        basename = os.path.splitext(os.path.basename(file))[0]
        img = face_recognition.load_image_file(file)
        encodings = face_recognition.face_encodings(img)

        if len(encodings) > 1:
            click.echo("WARNING: More than one face found in {}. Only considering the first face.".format(file))

        if len(encodings) == 0:
            click.echo("WARNING: No faces found in {}. Ignoring file.".format(file))
        else:
            known_names.append(basename)
            known_face_encodings.append(encodings[0])

    return known_names, known_face_encodings


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def upload_image():
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)

        file = request.files['file']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
           
            return detect_faces_in_image(file)

    
    return '''
    <!doctype html>
    <title>Face Detection and Recognition API</title>
    <h1>Upload a picture and see who is there in the picture!!!</h1>
    <form method="POST" enctype="multipart/form-data">
      <input type="file" name="file">
      <input type="submit" value="Upload">
    </form>
    '''



def detect_faces_in_image(file_stream):
    known_name_list = []
    known_face_encoding_list = []

    known_name_list, known_face_encoding_list = scan_known_people(known_faces_path)
   
    img = face_recognition.load_image_file(file_stream)

    unknown_face_encodings = face_recognition.face_encodings(img)
    

    face_found = False
    no_of_faces = 0
    name_found = []
    tolerance = 0.5
    show_distance = False
  

    if len(unknown_face_encodings) > 0:
        face_found = True
        no_of_faces = len(unknown_face_encodings)

        for i in range(len(unknown_face_encodings)):
            match_results = face_recognition.compare_faces([known_face_encoding_list], unknown_face_encodings[i])
            if (match_results == True):
                name_found = known_name_list[i]
        

        
        for unknown_encoding in unknown_face_encodings:
            distances = face_recognition.face_distance(known_face_encoding_list, unknown_encoding)
            result = list(distances <= tolerance)

            if True in result:
                for is_match, name, distance in zip(result, known_name_list, distances):
                     if is_match:
                        name_found.append(name)
            else:
                pass





    if len(unknown_face_encodings) == 0:
        face_found = 0
        name_found = "null"

    result = {
        "face_found_in_image": face_found,
        "name_of_peoples_found_in_image": name_found,
        "no_of_faces_found_in_image": no_of_faces
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=True)
