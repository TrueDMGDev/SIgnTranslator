import numpy as np
import os
from matplotlib import pyplot as plt
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_class_weight
from keras.utils import to_categorical
from keras.models import Sequential, load_model
import keras.layers as layers
import tensorflow as tf
from keras.optimizers import Adam, RMSprop, SGD, schedules
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
import json
from keras.regularizers import l2

global actions, num_sequences, fps, TRAIN_DATA_PATH, VAL_DATA_PATH


def Globals(num_classes):
    global actions, num_sequences, fps, TRAIN_DATA_PATH, VAL_DATA_PATH

    num = 1000 - num_classes
    with open("MSASL_classes.json", "r") as file:
        classes = json.load(file)
        classes = classes[:-num ]
        file.close()

    actions = np.array(classes)
    num_sequences = 30
    fps = 30

    TRAIN_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MP_Data')
    VAL_DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'MP_Data_Val')

def jitter(x, sigma=0.05):
    noise = np.random.normal(loc=0, scale=sigma, size=x.shape)
    return x + noise


def SetCreation(PATH):
    label_map = {label:num for num, label in enumerate(actions)}
    sequences, labels = [], []
    for action in actions:
        for sequence in range(num_sequences):
            window = []
            for frame_num in range(fps):
                try:
                    res = np.load(os.path.join(PATH, action, str(sequence), "{}.npy".format(frame_num)))
                    if PATH == TRAIN_DATA_PATH:
                        res = jitter(res)
                    window.append(res)
                except:
                    window = []
                    break
            if not len(window) == 0:
                sequences.append(window)
                labels.append(label_map[action])

    class_weight = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    class_weight = dict(enumerate(class_weight))

    x = np.array(sequences)

    del window
    del sequences

    y = to_categorical(labels).astype(int)

    if PATH == TRAIN_DATA_PATH:
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05)
        return x_train, x_test, y_train, y_test, class_weight
    else:
        return x, y
        
def ModelCreation():
    #Creating model
    model = Sequential()
    model.add(layers.Input(shape=(30,1662)))
    model.add(layers.LSTM(64, return_sequences=True, activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(128, return_sequences=True, activation='tanh'))
    model.add(layers.Dropout(0.3))
    model.add(layers.LSTM(64, return_sequences=False, activation='tanh', ))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(actions.shape[0], activation='softmax'))

    return model


def Training(x_train, y_train, x_val, y_val, class_weight):
    log_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Logs')
    MODEL_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Models')

    tb_callback = TensorBoard(log_dir)
    cp_callback = ModelCheckpoint(filepath=os.path.join(MODEL_PATH, 'sign_recognizer_acc.keras'), monitor='val_categorical_accuracy', save_best_only=True, verbose=1, mode='max', save_freq='epoch')
    es_callback = EarlyStopping(monitor='val_categorical_accuracy', mode='max', patience=75, start_from_epoch=50, verbose=1, restore_best_weights=True)
    callbacks = [tb_callback, cp_callback, es_callback]

    model = ModelCreation()

    optimizer = RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
            
    model.fit(x_train, y_train, epochs=2000, validation_data=(x_val, y_val), callbacks=[callbacks])
    tf.saved_model.save(model, export_dir='sign_recognizer_tflite')

    return model

def Summary(model, x_test, y_test):
    model.summary()
    res = model.predict(x_test)
    for i in range(len(res)):
        print('---------------------------------')
        print(actions[np.argmax(res[i])])
        print(actions[np.argmax(y_test[i])])
        print('---------------------------------')


if __name__ == "__main__":
    while True:
        num_classes = input("Enter the number of classes(10-400): ")
        if num_classes.isdigit():
            num_classes = int(num_classes)
            if num_classes >= 10 and num_classes <= 400:
                break
            else:
                print("Please enter a valid number!")
        else:
            print("Please enter a number!")
    
    Globals(num_classes)

    x_train, x_test, y_train, y_test, class_weight = SetCreation(TRAIN_DATA_PATH)
    x_val, y_val = SetCreation(VAL_DATA_PATH)

    model = Training(x_train, y_train, x_val, y_val, class_weight)

    Summary(model, x_test, y_test)