# -*- coding: utf-8 -*-


from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import json
import numpy as np
import cv2
import os
import image_similarity_measures
from sys import argv
from image_similarity_measures.quality_metrics import rmse, ssim, sre
from itertools import islice


app = FastAPI()


origins = [

    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
    Reqskills : str

@app.post('/imagesimiliar_prediction')
def diabetes_predds(input_parameters : model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    reqskills = input_dictionary['Reqskills']

    #productlist.clear()
    img = reqskills
    
    #dataset
    url = "https://haycouture.000webhostapp.com/user/imgsimiliar/dataset/"
    folder_name = "dataset"

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # download each file in the folder
    for filename in os.listdir(folder_name):
        file_url = url + filename
        file_path = os.path.join(folder_name, filename)
        urllib.request.urlretrieve(file_url, file_path)
        print(f"Downloaded {filename} to {file_path}")
    
    #test
    url = "https://haycouture.000webhostapp.com/user/imgsimiliar/test/"
    folder_name = "test"

    # create the folder if it does not exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    # download each file in the folder
    for filename in os.listdir(folder_name):
        file_url = url + filename
        file_path = os.path.join(folder_name, filename)
        urllib.request.urlretrieve(file_url, file_path)
        print(f"Downloaded {filename} to {file_path}")

    test_img = cv2.imread('./test/' + img)
    test_img = np.squeeze(test_img)
    ssim_measures = {}
    rmse_measures = {}
    sre_measures = {}

    scale_percent = 100 # percent of original img size
    width = int(test_img.shape[1] * scale_percent / 100)
    height = int(test_img.shape[0] * scale_percent / 100)
    dim = (width, height)

    data_dir = './dataset'

    for file in os.listdir(data_dir):
        img_path = os.path.join(data_dir, file)
        data_img = cv2.imread(img_path)
        resized_img = cv2.resize(data_img, dim, interpolation = cv2.INTER_AREA)
        ssim_measures[img_path]= ssim(test_img, resized_img)
        rmse_measures[img_path]= rmse(test_img, resized_img)
        sre_measures[img_path]= sre(test_img, resized_img)



    ssims = calc_closest_val(ssim_measures, True)

    sres = calc_closest_val(sre_measures, True)

    res_lt = []
    for x in range (0, len(ssims)):
        res_lt.append(ssims[x])
        res_lt.append(sres[x])
    res_lt = list(dict.fromkeys(res_lt))#duplicate remove

    #res_lt = json.dumps(res_lt)

    return json.dumps(res_lt)


def calc_closest_val(dicts, checkMax):
    productlist = []
    result = {}
    if (checkMax):
        closest = max(dicts.values())
    else:
        closest = min(dicts.values())


    sorted_dicts = sorted(dicts.items(), key=lambda x:x[1], reverse=True)
    dicts = dict(sorted_dicts)

    out = dict(islice(dicts.items(), 5))


    for key, value in out.items():
        print(key[8:])
        print(key,value)
        productlist.append(key[8:])
        if (value == closest):
            result[key] = closest

    return productlist

