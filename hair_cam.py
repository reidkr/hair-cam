# -*- coding: utf-8 -*-

"""
hair_cam.py
~~~~~~~~~~

This module implements a CNN trained to recognize hair type.

:copyright: (c) 2020 by Kemar Reid
"""

import os
from io import BytesIO
import json
import streamlit as st
import PIL
from fastai.vision import *
import time

# Data
with open('data/hair_type_desc.json', mode='r') as f:
    hair_type_desc = json.load(f)

# Definitions


def predict_hair_type(uploaded_file):
    """
    TODO:
    """
    image = open_image(uploaded_file)
    # try:
    defaults.device = torch.device('cpu')  # Use CPU
    pred, pred_label, pred_prob = model.predict(image)
    return (pred, pred_prob)
    # except:
    #     return None


# Get model directory
path = os.getcwd() + '/model'

# Load model
model = load_learner(path, 'stage-2-resnet-50-balanced-subtype.pkl')

st.markdown('''
# Hair[CAM]
## Hair type prediction for better hair days
''')

# Hair type chart
h_type_chart = PIL.Image.open('imgs/hair_type_chart.png', mode='r')
st.sidebar.image(h_type_chart, use_column_width=True)

# Preloaded images
if st.button(label='Demo'):
    with open('imgs/jon_snow.jpg', "rb") as image:
        f = image.read()
        uploaded_file = BytesIO(f)
else:
    uploaded_file = st.file_uploader(label='Choose a photo...')

if uploaded_file is not None:
    image = PIL.Image.open(uploaded_file, mode='r')
    st.image(image, caption='Photo uploaded', use_column_width=True)

    # Predict hair type

if st.button(label='Click to get your hair type'):
    # st.write('''
    # \n
    # ### Getting your hair type...
    # ''')
    with st.spinner(text='Getting your hair type...'):
        time.sleep(3)
        st.success('Done!')
    label, prob = predict_hair_type(uploaded_file)  # Make prediction
    hair_types = ['1', '2A', '2B', '2C', '3A', '3B',
                  '3C', '4A', '4B', '4C']
    # if hair_types[int(label)] == 'Wavy':
    #     thickness = st.radio(
    #         label='Select your hair thickness: ',
    #         options=('Thin', 'Medium', 'Thick'))
    #     st.write('You selected: ', thickness)
    h_type = hair_types[int(label)]
    st.write(f'### Your hair type is `{h_type}`!')
    st.balloons()
    # st.write(f'Confidence: {prob}') # Load into pandas and use to report

    # About your hair type

    st.sidebar.markdown(f'''
    ### About your hair type:
    \n
    _{hair_type_desc[h_type]}_
    ''')

    # Famous people with similar hair types

    st.write('''
        \n
        ### Famous people with similar hair type:
        ''')
    celeb_image = PIL.Image.open(f'imgs/{h_type}.jpg')
    st.image(celeb_image, use_column_width=True)
