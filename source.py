#Importing the libraries
import pandas as pd
import numpy as np
import glob
import re
import cv2
from sklearn.preprocessing import MultiLabelBinarizer
import os
import pickle
from scipy import misc
from keras.preprocessing import image

def precision_recall(y_test, y_pred):
    TP = 0
    FP = 0
    FN = 0
    for x in y_test:
        if x in y_pred:
            TP += 1
        else:
            FN += 1
    for x in y_pred:
        if x not in y_test:
            FP += 1
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / float(TP + FP)
    
    if TP + FN == 0:
        recall = 0
    else:
        recall = TP / float(TP + FN)
    return(precision , recall)

#Importing the Dataset
dataset = pd.read_csv("MovieGenre.csv")
poster_names = dataset.iloc[:, 2].values
poster_genres = dataset.iloc[:, 4].values
poster_links = dataset.iloc[:, 5].values

#Converting the Objects to Lists
poster_names_list = poster_names.copy().tolist()
poster_genres_list = poster_genres.copy().tolist()
poster_links_list = poster_links.copy().tolist()

#Creating Reference Lists
correct_movie_training = []
correct_movie_testing = []

#Creating the Training Data Images
dir_path = "Dataset\\training_set\\"
x_train = []
for file in os.listdir(dir_path):
    if os.path.join(dir_path, file) in glob.glob("Dataset\\training_set\\*.jpg"):
        print(os.path.join(dir_path, file))
        img = cv2.imread(os.path.join(dir_path, file))
        if img is None:
            pass
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            correct_movie_training.append(file)
            x_train.append(img)
x_train_copy = x_train

#Saving the list of training images
f = open("x_train_copy.pckl", "wb")
pickle.dump(x_train_copy, f)
f.close()

#Saving the list of training movies which gave no error
f = open("correct_movie_training.pckl", "wb")
pickle.dump(correct_movie_training, f)
f.close()

#Loading the list of training movies which gave no error
f = open("correct_movie_training.pckl", "rb")
correct_movie_training = pickle.load(f)
f.close()

#Creating the Testing Data Images
dir_path = "Dataset\\test_set\\"
x_test = []
for file in os.listdir(dir_path):
    if os.path.join(dir_path, file) in glob.glob("Dataset\\test_set\\*.jpg"):
        print(os.path.join(dir_path, file))
        img = cv2.imread(os.path.join(dir_path, file))
        if img is None:
            pass
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            correct_movie_testing.append(file)
            x_test.append(img)
x_test_copy = x_test

#Saving the list of testing images
f = open("x_test_copy.pckl", "wb")
pickle.dump(x_test_copy, f)
f.close()

#Saving the list of testing movies which gave no error
f = open("correct_movie_testing.pckl", "wb")
pickle.dump(correct_movie_testing, f)
f.close()

#loading the list of testing movies which gave no error
f = open("correct_movie_testing.pckl", "rb")
correct_movie_testing = pickle.load(f)
f.close()

#Creating the Training Data Labels
ground_truth_training = []
incorrect_index_training = []
for file in correct_movie_training:
    file_name = re.split('\.', file)
    for char in file_name:
        if char == '':
            file_name.remove(char)
    if file_name[0] == '':
        file_name.remove(file_name[0])
    try:
        genre_list = re.split('\|', poster_genres_list[poster_names_list.index(str(file_name[0]))])
        temp_set = set(genre_list)
        ground_truth_training.append(temp_set)
    except:
        incorrect_index_training.append(correct_movie_training.index(file))
#Accounting for the index shift due to deletion of an element
index_shift = 0
for index in incorrect_index_training:
    del x_train_copy[index - index_shift]
    index_shift = index_shift + 1
    
mlb_train = MultiLabelBinarizer()
y_train = mlb_train.fit_transform(ground_truth_training)

#Creating the dictionary mapping genre names to indices
genre_index_to_name = {}
genre_classes = list(mlb_train.classes_)
for x in genre_classes:
    genre_index_to_name[genre_classes.index(x)] = x

#Saving the dictionary
f = open("genre_index_to_name.pckl", "wb")
pickle.dump(genre_index_to_name, f)
f.close()

#Loading the dictionary
f = open("genre_index_to_name.pckl", "rb")
genre_index_to_name = pickle.load(f)
f.close()

#Saving the ground_truth_training labels
f = open("y_train.pckl", "wb")
pickle.dump(y_train, f)
f.close()

#Loading the ground truth training labels
f = open("y_train.pckl", "rb")
y_train = pickle.load(f)
f.close()
            
#Creating the Testing Data Labels
ground_truth_testing = []
incorrect_index_testing = []
for file in correct_movie_testing:
    file_name = re.split('\.', file)
    for char in file_name:
        if char == '':
            file_name.remove(char)
    if file_name[0] == '':
        file_name.remove(file_name[0])
    try:
        genre_list = re.split('\|', poster_genres_list[poster_names_list.index(str(file_name[0]))])
        temp_set = set(genre_list)
        ground_truth_testing.append(temp_set)
    except:
        incorrect_index_testing.append(correct_movie_testing.index(file))
#Accounting for the index shift due to deletion of an element
index_shift_1 = 0
for index in incorrect_index_testing:
    del x_test_copy[index - index_shift_1]
    index_shift_1 = index_shift_1 + 1
    
mlb_test = MultiLabelBinarizer()
y_test = mlb_test.fit_transform(ground_truth_testing)

#Saving the ground_truth_testing labels
f = open("y_test.pckl", "wb")
pickle.dump(y_test, f)
f.close()

#Loading the ground truth testing labels
f = open("y_test.pckl", "rb")
y_test = pickle.load(f)
f.close()

#Using the VGG16Net to extract the features from the posters
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input

#Creating the VGG16 network for feature extraction
model_vgg16 = VGG16(weights = "imagenet", include_top = False)

#Extracting the training features
feature_list_training = []
for training_image in x_train:
    x = misc.imresize(training_image, (224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    features = model_vgg16.predict(x)
    feature_list_training.append(features)
    print(len(feature_list_training))

#Saving the extracted training features
f = open("feature_list_training.pckl", "wb")
pickle.dump(feature_list_training, f)
f.close()

#Loading the extracted training features
f = open("feature_list_training.pckl", "rb")
feature_list_training = pickle.load(f)
f.close()

#Extracting the testing features
feature_list_testing = []
for testing_image in x_test:
    x = misc.imresize(testing_image, (224, 224))
    x = image.img_to_array(x)
    x = np.expand_dims(x, axis = 0)
    x = preprocess_input(x)
    features = model_vgg16.predict(x)
    feature_list_testing.append(features)
    print(len(feature_list_testing))

#Saving the extracted testing features
f = open("feature_list_testing", "wb")
pickle.dump(feature_list_testing, f)
f.close()

#Loading the extracted testing features
f = open("feature_list_testing", "rb")
feature_list_testing = pickle.load(f)
f.close()

#Creating the training feature set
(a1, b1, c1, d1) = feature_list_training[0].shape
feature_size_train = a1 * b1 * c1 * d1

np_features_train = np.zeros((len(feature_list_training), feature_size_train))
for i in range(len(feature_list_training)):
    feature = feature_list_training[i]
    reshaped_feature = feature.reshape(1, -1)
    np_features_train[i] = reshaped_feature

x_train = np_features_train

#Creating the testing feature set
(a2, b2, c2, d2) = feature_list_testing[0].shape
feature_size_test = a2 * b2 * c2 * d2

np_features_test = np.zeros((len(feature_list_testing), feature_size_test))
for i in range(len(feature_list_testing)):
    feature = feature_list_testing[i]
    reshaped_feature = feature.reshape(1, -1)
    np_features_test[i] = reshaped_feature

x_test = np_features_test

#Creating the ANN for classification
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

classifier = Sequential()
classifier.add(Dense(units = 1024, input_shape = (25088, ), activation = 'relu'))
classifier.add(Dense(units = 256, activation = 'relu'))
classifier.add(Dense(units = 24, activation = 'sigmoid'))
opt = optimizers.rmsprop(lr = 0.0001, decay = 1e-6)
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.fit(x_train, y_train, batch_size = 32, epochs = 25, verbose = 1)

#Saving the classifier
classifier_json = classifier.to_json()
with open("model.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("model.h5")
print("Classifier saved to disk")

#Loading the classifier
from keras.models import model_from_json
json_file = open("model.json", "r")
loaded_classifier_json = json_file.read()
classifier = model_from_json(loaded_classifier_json)
classifier.load_weights("model.h5")
print("Loaded classifier from file")
classifier.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

y_pred = classifier.predict(x_test)


precision_list = []
recall_list = []

for i in range(len(y_test)):
    y_test_row = y_test[i]
    y_pred_row = y_pred[i]
    actual_genre_list = []
    predicted_genre_list = []
    for j in range(24):
        if y_test_row[j] == 1:
            actual_genre_list.append(genre_index_to_name[j])
    topx = np.argsort(y_pred_row)[-5:]
    for genre_index in topx:
        predicted_genre_list.append(genre_index_to_name[genre_index])
    precision, recall = precision_recall(actual_genre_list, predicted_genre_list)
    precision_list.append(precision)
    recall_list.append(recall)
    print("Actual: ", ','.join(actual_genre_list), " Predicted: ", ','.join(predicted_genre_list))

print("Overall Precision: ", np.mean(np.asarray(precision_list)), "Overall Recall: ", np.mean(np.asarray(recall_list)))