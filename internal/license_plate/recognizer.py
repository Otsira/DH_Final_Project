import tensorflow as tf
import string
import numpy as np
import cv2


from .config import recognizer, custom_objects


class Recognizer():
    def __init__(self) -> None:
        self.alphabet = string.digits + string.ascii_uppercase + '_'
        self.model = tf.keras.models.load_model(
            recognizer.path, custom_objects=custom_objects)

    def check_low_conf(self, probs, thresh=0.3):
        '''
        Add position of chars. that are < thresh
        '''
        return [i for i, prob in enumerate(probs) if prob < thresh]

    @tf.function
    def predict_from_array(self, img):
        pred = self.model(img, training=False)
        return pred

    def probs_to_plate(self, prediction) -> list:
        '''
        This function takes the predictions of the images and return
        '''
        prediction = prediction.reshape((7, 37))
        probs = np.max(prediction, axis=-1)
        prediction = np.argmax(prediction, axis=-1)
        plate = list(map(lambda x: self.alphabet[x], prediction))
        return plate, probs

    def read_plate(self, frame) -> dict:
        im = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        img = cv2.resize(im, dsize=(140, 70),
                         interpolation=cv2.INTER_LINEAR)
        img = img[np.newaxis, ..., np.newaxis] / 255.
        img = tf.constant(img, dtype=tf.float32)
        prediction = self.predict_from_array(img).numpy()
        plate, probs = self.probs_to_plate(prediction)
        plate_str = ''.join(plate)
        print(f'License Plate #: {plate_str}', flush=True)
        print(f'Confidence: {probs}', flush=True)
        return {'plate': plate_str, 'probs': probs.tolist()}
