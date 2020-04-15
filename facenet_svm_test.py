from os import listdir
import os
import numpy as np
from os.path import isdir
from PIL import Image
import pickle
from mtcnn.mtcnn import MTCNN
from glob import glob
import shutil
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib
import cv2

# Đường dẫn đến video
test_video ="test02.mp4"
facenet_model = load_model('facenet_keras.h5')
detector = MTCNN()
dest_size = (160, 160)

def get_embedding(model, face_pixels):
	# scale pixel values
	face_pixels = face_pixels.astype('float32')
	# standardize pixel values across channels (global)
	mean, std = face_pixels.mean(), face_pixels.std()
	face_pixels = (face_pixels - mean) / std
	# transform face into one sample
	samples = np.expand_dims(face_pixels, axis=0)
	# make prediction to get embedding
	yhat = model.predict(samples)
	return yhat[0]


# Load SVM model từ file
pkl_filename = 'faces_svm.pkl'
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

# Load ouput_enc từ file để hiển thị nhãn
pkl_filename = 'output_enc.pkl'
with open(pkl_filename, 'rb') as file:
    output_enc = pickle.load(file)

cap = cv2.VideoCapture(test_video)

while(True):

    # Capture ảnh từ video
    ret, frame = cap.read()
    if not ret:
        break

    pixels = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect khuôn mặt
    results = detector.detect_faces(pixels)

    if len(results) > 0:
        # Chỉ lấy khuôn mặt đầu tiên, ta coi các ảnh train chỉ có 1 mặt
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(dest_size)

        # Lây face embeding
        face_emb =  get_embedding(facenet_model, np.array(image))
        # Chuyển thành tensor
        face_emb = np.expand_dims(face_emb, axis=0)
        # Predict qua SVM
        y_hat = pickle_model.predict(face_emb)

        # Lấy nhãn và viết lên ảnh
        predict_names = output_enc.inverse_transform(y_hat)
        if predict_names!=None:
            cv2.putText(frame,predict_names[0],(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

