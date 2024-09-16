import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'you-will-never-guess'
    SHAPE_PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
    FACE_RECOGNITION_MODEL_PATH = 'models/dlib_face_recognition_resnet_model_v1.dat'
    KNOWN_FACES_DIR = 'known_faces'
    SENDER_EMAIL = 'your_email@example.com'
    RECEIVER_EMAIL = 'receiver@example.com'
    EMAIL_PASSWORD = 'your_email_password'  # Use environment variables for sensitive data in production