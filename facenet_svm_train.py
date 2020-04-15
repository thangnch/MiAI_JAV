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

raw_folder ="data/raw/"
processed_folder ="data/processed/"
facenet_model = load_model('facenet_keras.h5')
detector = MTCNN()

def get_faces(raw_folder=raw_folder, processed_folder=processed_folder, copy=True):

    dest_size = (160, 160)
    print("Bắt đầu xử lý crop mặt...")

    # Lặp qua các folder con trong thư mục raw
    for folder in listdir(raw_folder):

        # Tạo thư mục chứa ảnh processed
        os.mkdir(processed_folder + folder);
        # Lặp qua các file trong từng thư mục chứa các em
        for file in listdir(raw_folder  + folder):

            image = Image.open(raw_folder + folder + "/" + file)
            image = image.convert('RGB')

            # Nếu không cần xử lý thì chỉ resize và save
            if copy:
                image = image.resize(dest_size)
                # Save file
                image.save(processed_folder + folder + "/" + file)
            else:
                # Nếu cần xử lý thì đọc ảnh và lấy mặt
                pixels = asarray(image)

                results = detector.detect_faces(pixels)

                # Chỉ lấy khuôn mặt đầu tiên, ta coi các ảnh train chỉ có 1 mặt
                x1, y1, width, height = results[0]['box']
                x1, y1 = abs(x1), abs(y1)
                x2, y2 = x1 + width, y1 + height
                face = pixels[y1:y2, x1:x2]
                image = Image.fromarray(face)
                image = image.resize(dest_size)

                # Save file
                image.save(processed_folder + folder + "/" + file )

    return

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

def load_faces(train_folder = processed_folder):
    if os.path.exists("faces_data.npz"):
        data = np.load('faces_data.npz')
        X_train,y_train = data["arr_0"],data["arr_1"]
        return X_train, y_train
    else:
        X_train = []
        y_train = []

        # enumerate folders, on per class
        for folder in listdir(train_folder):
            # Lặp qua các file trong từng thư mục chứa các em
            for file in listdir(train_folder + folder):
                # Read file
                image = Image.open(train_folder + folder + "/" + file)
                # convert to RGB, if needed
                image = image.convert('RGB')
                # convert to array
                pixels = np.asarray(image)

                # Thêm vào X
                X_train.append(pixels)
                y_train.append(folder)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Check dữ liệu
        print(X_train.shape)
        print(y_train.shape)
        print(y_train[0:5])

        # Convert du lieu y_train
        output_enc = LabelEncoder()
        output_enc.fit(y_train)
        y_train = output_enc.transform(y_train)
        pkl_filename = "output_enc.pkl"
        with open(pkl_filename, 'wb') as file:
            pickle.dump(output_enc, file)

        print(y_train[0:5])

        # Convert du lieu X_train sang embeding
        X_train_emb = []
        for x in X_train:
            X_train_emb.append( get_embedding(facenet_model, x))

        X_train_emb = np.array(X_train_emb)

        print("Load faces done!")
        # Save
        np.savez_compressed('faces_data.npz', X_train_emb, y_train);
        return X_train_emb, y_train


# Main program
X_train, y_train = load_faces()

# Train SVM với kernel tuyến tính
model = SVC(kernel='linear',probability=True)
model.fit(X_train, y_train)

# Save model
pkl_filename = "faces_svm.pkl"
with open(pkl_filename, 'wb') as file:
    pickle.dump(model, file)

print("Saved model")


