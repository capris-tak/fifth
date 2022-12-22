import streamlit as st

title = st.text_input('Movie title', 'Life of Brian')
st.write('The current movie title is', title)


import numpy as np
import cv2
from insightface.app import FaceAnalysis
 
image_file = "input.jpg"
img = cv2.imread(image_file)
 
app = FaceAnalysis()
app.prepare(ctx_id=0, det_size=(640, 640))
 
faces = app.get(img)
rimg = app.draw_on(img, faces)

#faces = app.get(np.asarray(img))
st.write("faces:" + str(len(faces)))
 
rimg = app.draw_on(img, faces)
st.image(rimg)
