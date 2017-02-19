#!/usr/bin/env python
# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script uses the Vision API's label detection capabilities to find a label
based on an image's content.

To run the example, install the necessary libraries by running:

    pip install -r requirements.txt

Run the script on an image to get a label, E.g.:

    ./label.py <path-to-image>
"""
import numpy as np
import cv2
from PIL import Image
from resizeimage import resizeimage
import urllib
from urllib.request import urlopen
import time

from flask import Flask, jsonify, request
app = Flask(__name__)

# [START import_libraries]
import argparse
import base64
import io
import os

from google.cloud import vision


import googleapiclient.discovery
# [END import_libraries]

# OpenCV
# Returns filepath of grabCut .png image
def grabCut(filepath):
    # open image from url
    req = urllib.request.urlopen(filepath)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr,-1) # 'load it as it is'

    # open image from local file
    # img = cv2.imread(filepath)
    dim = (600, 600)
    # perform the actual resizing of the image and show it
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    # cv2.imwrite('resized.jpeg', resized)

    # img = cv2.imread('resizeddog.jpeg')              # img.shape : (413, 620, 3)
    img = resized
    mask = np.zeros(img.shape[:2],np.uint8)   # img.shape[:2] = (413, 620)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # (x,y,width,height)
    rect = (25,25,550,550)

    # this modifies mask 
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    # If mask==2 or mask== 1, mask2 get 0, other wise it gets 1 as 'uint8' type.
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    # adding additional dimension for rgb to the mask, by default it gets 1
    # multiply it with input image to get the segmented image
    img_cut = img*mask2[:,:,np.newaxis]

    # change white/black background color to alpha/clear background
    tmp = cv2.cvtColor(img_cut, cv2.COLOR_BGR2GRAY)
    _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
    b, g, r = cv2.split(img_cut)
    rgba = [b,g,r, alpha]
    img_cut_alpha = cv2.merge(rgba,4)

    if not os.path.exists('./tmp'):
        os.makedirs('./tmp')

    filename = str(time.time())
    filepath = './tmp/' + filename + '.png'
    cv2.imwrite(filepath, img_cut_alpha)
    return filepath

def detect_labels(path):
    """Detects labels in the file."""
    vision_client = vision.Client()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    img = vision_client.image(content=content)

    animalType = 'Cannot identify'
    identified = False
    labels = img.detect_labels()
    # print('Labels:')
    for label in labels:
        if label.description == 'dog' or label.description == 'cat':
            animalType = label.description
            # print('Type: ' + animalType)
            identified = True

    if (not identified):
        # print('Type: Cannot identify')
        return 'Cannot identify'
    return animalType

def detect_properties(path):
    """Detects image properties in the file."""
    vision_client = vision.Client()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    img = vision_client.image(content=content)

    properties = img.detect_properties()
    # print('Properties:')
    avg_red = 0
    avg_green = 0
    avg_blue = 0
    for prop in properties:
        for color in prop.colors:
            # color.pixel_fraction
            # print('fraction: {}'.format(color.score))
            # print('r: {}'.format(color.color.red))
            # print('g: {}'.format(color.color.green))
            # print('b: {}'.format(color.color.blue))
            avg_red += color.color.red * color.score
            avg_green += color.color.green * color.score
            avg_blue += color.color.blue * color.score
     # print('Average color: ' + '( r:', round(avg_red), 'g:', round(avg_green), 'b:', round(avg_blue), ')')
    return '#%02x%02x%02x' % (round(avg_red), round(avg_green), round(avg_blue))

def main(photo_file):
    """Run a label request on a single image"""
    # run grabCut to segment and extract animal from image
    img_cut_alpha_path = grabCut(photo_file)
    # identify type of animal
    animalType = detect_labels(img_cut_alpha_path)
    # identify color of animal in hexadecimal format
    animalColor = detect_properties(img_cut_alpha_path)
    return jsonify({ "petType": animalType, "color": animalColor })

@app.route("/")
def test():
    return "hi there"

@app.route("/recognize/")
def recognize():
    url = request.args.get('url')
    return main(url)


# [START run_application]
if __name__ == '__main__':
    app.run()
    # parser = argparse.ArgumentParser()
    # parser.add_argument('image_file', help='The image you\'d like to label.')
    # args = parser.parse_args()
    # main(args.image_file)
# [END run_application]