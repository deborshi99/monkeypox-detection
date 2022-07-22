from cgi import test
from utils.constants import MODEL_DIR
from keras.preprocessing import image
from keras.utils import load_img, img_to_array
#from tensorflow.keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import cv2
import os 
import numpy as np
import time
import shutil
from keras.applications.resnet import preprocess_input

def get_output():
    os.makedirs("./processed_data", exist_ok=True)
    input_dir = "./input_data"
    proessed_dir = "./processed_data"
    model = load_model(MODEL_DIR)
    test_image_path = []
    for i in range(len(os.listdir(input_dir))):
        l = os.path.join(input_dir, os.listdir(input_dir)[i])
        test_image_path.append(l)


    for i in test_image_path:
        img = cv2.imread(i)
        img_resize = cv2.resize(img, (224, 224))
        cv2.imwrite(f"./processed_data/{i.split('/')[-1]}", img_resize)

    processed_image_path = []
    for i in range(len(os.listdir(proessed_dir))):
        l = os.path.join(proessed_dir, os.listdir(proessed_dir)[i])
        processed_image_path.append(l)
    result = []
    for i in processed_image_path:
        start_time = time.time()
        img = load_img(i, target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        images = np.vstack([img_array])
        classes = model.predict(images, batch_size=len(os.listdir(proessed_dir)))
        end_time = time.time()
        time_taken = end_time-start_time
        
        if classes[0][0] < 0.5:
            result.append({
                "image name": i.split("/")[-1],
                "prediction": "monkey pox",
                "time taken": time_taken
            }) 
        elif classes[0][0] >= 0.5:
            result.append({
                "image name": i.split("/")[-1],
                "prediction": "other pox",
                "time taken": time_taken
            })
    return result

    


    


