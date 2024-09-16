import cv2
import dlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import smtplib
import os
from werkzeug.utils import secure_filename
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage

# Global variables
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor("D:\Programming\Projects\Image-Recognition-System\shape_predictor_68_face_landmarks.dat")
face_recognition_model = dlib.face_recognition_model_v1("D:\Programming\Projects\Image-Recognition-System\dlib_face_recognition_resnet_model_v1.dat")

known_face_encodings = []
known_face_names = []

def load_known_faces(directory):
    global known_face_encodings, known_face_names
    # Load known faces from the directory
    # This is a placeholder - you'd need to implement this based on your data structure
    pass

def detect_faces(image):
    # Convert the image file to a numpy array
    nparr = np.frombuffer(image.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_detector(gray)
    
    # Extract face regions and compute face encodings
    face_encodings = []
    face_images = []
    for face in faces:
        shape = shape_predictor(gray, face)
        face_encoding = np.array(face_recognition_model.compute_face_descriptor(img, shape))
        face_encodings.append(face_encoding)
        
        # Extract face image
        x, y, w, h = (face.left(), face.top(), face.width(), face.height())
        face_img = img[y:y+h, x:x+w]
        face_images.append(face_img)
    
    return face_images, face_encodings

def recognize_faces(face_images, face_encodings):
    recognized_faces = []
    for face_img, face_encoding in zip(face_images, face_encodings):
        # Compare with known face encodings
        similarities = cosine_similarity([face_encoding], known_face_encodings)[0]
        best_match_index = np.argmax(similarities)
        
        if similarities[best_match_index] > 0.6:  # Adjust this threshold as needed
            name = known_face_names[best_match_index]
        else:
            name = "Unknown"
        
        recognized_faces.append((name, face_img))
    
    return recognized_faces

def add_known_face(image, name):
    # Ensure the name is safe for use as a directory name
    safe_name = secure_filename(name)
    
    # Create a directory for the person if it doesn't exist
    person_dir = os.path.join(current_app.config['KNOWN_FACES_DIR'], safe_name)
    os.makedirs(person_dir, exist_ok=True)
    
    # Save the uploaded image
    image_filename = secure_filename(image.filename)
    image_path = os.path.join(person_dir, image_filename)
    image.save(image_path)
    
    # Detect face and compute encoding
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)
    
    if len(faces) != 1:
        os.remove(image_path)  # Remove the saved image if face detection fails
        return False
    
    shape = shape_predictor(gray, faces[0])
    face_encoding = np.array(face_recognition_model.compute_face_descriptor(img, shape))
    
    # Add the new face encoding and name to the known faces
    known_face_encodings.append(face_encoding)
    known_face_names.append(safe_name)
    
    return True

def send_notification(recognized_faces):
    # Email configuration
    sender_email = "your_email@example.com"
    receiver_email = "receiver@example.com"
    password = "your_password"
    
    # Create message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Face Recognition Alert"
    
    # Create the plain-text and HTML version of your message
    text = f"Recognized faces: {', '.join(name for name, _ in recognized_faces)}"
    html = f"""
    <html>
      <body>
        <h2>Face Recognition Alert</h2>
        <p>The following faces were recognized:</p>
        <ul>
          {"".join(f"<li>{name}</li>" for name, _ in recognized_faces)}
        </ul>
      </body>
    </html>
    """

    
    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    
    # Add HTML/plain-text parts to MIMEMultipart message
    message.attach(part1)
    message.attach(part2)
    
    # Attach face images
    for i, (name, face_img) in enumerate(recognized_faces):
        img_byte_arr = cv2.imencode('.jpg', face_img)[1].tostring()
        image = MIMEImage(img_byte_arr, name=f"face_{i}.jpg")
        message.attach(image)
    
    # Create secure connection with server and send email
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message.as_string())

    print("Notification sent successfully")

# Load known faces during initialization
load_known_faces('path/to/known_faces_directory')