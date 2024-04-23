import tensorflow as tf
import os

saved_model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'sign_recognizer_tflite')
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)

tflite_model = converter.convert()

with open('converted_model.tflite', 'wb') as f:
    f.write(tflite_model)