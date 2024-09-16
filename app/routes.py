from flask import Blueprint, render_template, request, current_app, redirect, url_for, flash
from .utils import detect_faces, recognize_faces, send_notification, add_known_face
import os
from werkzeug.utils import secure_filename

main = Blueprint('main', __name__)

@main.route('/')
def index():
    return render_template('index.html')

@main.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        flash('No image uploaded', 'error')
        return redirect(url_for('main.index'))
    
    image = request.files['image']
    face_images, face_encodings = detect_faces(image)
    recognized_faces = recognize_faces(face_images, face_encodings)
    
    if recognized_faces:
        send_notification(recognized_faces)
    
    return render_template('result.html', faces=recognized_faces)

@main.route('/add_known_face', methods=['GET', 'POST'])
def add_known_face_route():
    if request.method == 'POST':
        if 'image' not in request.files:
            flash('No image uploaded', 'error')
            return redirect(url_for('main.add_known_face_route'))
        
        image = request.files['image']
        name = request.form.get('name')
        
        if image.filename == '' or not name:
            flash('Both image and name are required', 'error')
            return redirect(url_for('main.add_known_face_route'))
        
        if add_known_face(image, name):
            flash(f'Successfully added {name} to known faces', 'success')
        else:
            flash('Failed to add known face. Please try again.', 'error')
        
        return redirect(url_for('main.index'))
    
    return render_template('add_known_face.html')

