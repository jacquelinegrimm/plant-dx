#!/usr/bin/env python
# coding: utf-8

import subprocess
subprocess.run(['pip', 'install', '-Uqq', 'fastai'])
subprocess.run(['pip', 'install', '-Uqq', 'timm'])

from fastai.vision.all import *
import gradio as gr

def diagnosis(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Apple - Apple Scab', 'Apple - Black Rot', 'Apple - Cedar Apple Rust', 'Apple - Healthy', 
              'Blueberry - Healthy', 'Cherry - Powdery Mildew', 'Cherry - Healthy', 
              'Corn - Gray Leaf Spot', 'Corn - Common Rust', 'Corn - Northern Leaf Blight', 
              'Corn - Healthy', 'Grape - Black Rot', 'Grape - Esca (Black Measles)', 'Grape - Isariopsis Leaf Spot', 
              'Grape - Healthy', 'Orange - Citrus Greening', 'Peach - Bacterial Spot', 'Peach - Healthy', 
              'Bell Pepper - Bacterial Spot', 'Bell Pepper - Healthy', 'Potato - Early Blight', 'Potato - Late Blight', 
              'Potato - Healthy', 'Raspberry - Healthy', 'Soybean - Healthy', 'Squash - Powdery Mildew', 'Strawberry - Leaf Scorch', 
              'Strawberry - Healthy', 'Tomato - Bacterial Spot', 'Tomato - Early Blight', 'Tomato - Late Blight', 'Tomato - Leaf Mold', 
              'Tomato - Septoria Leaf Spot', 'Tomato - Spider Mites', 'Tomato - Target Spot', 
              'Tomato - Yellow Leaf Curl Virus', 'Tomato - Tomato Mosaic Virus', 'Tomato - Healthy')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories, map(float,probs)))

#Creates a Gradio interface for image classification
examples = ['PotatoHealthy1.jpeg', 'PotatoEarlyBlight1.jpeg', 'TomatoYellowCurlVirus2.jpeg', 
            'CornCommonRust2.jpeg', 'AppleScab1.jpeg', 'AppleCedarRust4.jpeg']
intf = gr.Interface(fn=classify_image, inputs='image', outputs='label', examples=examples)
intf.launch(inline=False)