import streamlit as st
import cv2
from PIL import Image
import numpy as np

from gluoncv import utils
import mxnet as mx 

from autogluon.vision import ObjectDetector
#import torch
#import imutils

uploaded_image = st.file_uploader('Choose an image..',type=['png', 'jpg','jpeg','webp'])

if uploaded_image is not None:
	XML_PATH = "haarcascade_frontalface_default.xml"
	#INPUT_IMG_PATH = "input.jpg"
	#OUTPUT_IMG_PATH = "出力する画像のパス"
	classifier = cv2.CascadeClassifier(XML_PATH)
	image=Image.open(uploaded_image)
	img_array = np.array(image)
	img = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
	color = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)

	targets = classifier.detectMultiScale(color)

	for x, y, w, h in targets:
		cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
	#cv2.imwrite(OUTPUT_IMG_PATH, img)
	st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
	
	
	
	
	# 顔検出インスタンス生成
	cascadePath = 'haarcascade_frontalface_alt2.xml'
	faceCascade = cv2.CascadeClassifier(cascadePath)
	# 目検出インスタンス生成
	eye_cascadePath = 'haarcascade_eye.xml'
	eye_cascade = cv2.CascadeClassifier(eye_cascadePath)	
	# グレースケールに変換
	gray = cv2.cvtColor(img_array,cv2.COLOR_BGR2GRAY)
	# 顔検出
	faces = faceCascade.detectMultiScale( 
		gray,
		scaleFactor = 1.2,
		minNeighbors = 3,
		minSize = (10, 10)
		)
	# 顔検出箇所に矩型描画のためのループ
	for(x,y,w,h) in faces:
		# 顔箇所を四角で描画
		img_cascade = cv2.rectangle(img_array, (x,y), (x+w,y+h), (0,255,0), 2)
	st.image(img_cascade)#cv2.cvtColor(img_cascade, cv2.COLOR_BGR2RGB))
 


'''
import numpy as np
import cv2
from insightface.app import FaceAnalysis
 
image_file = "input.jpg"
img = cv2.imread(image_file)
 
app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
 
faces = app.get(np.asarray(img))
rimg = app.draw_on(img, faces)

#st.write("faces:" + str(len(faces)))
#rimg = app.draw_on(img, faces)
#st.image(rimg)

from retinaface import RetinaFace
import cv2
 
img_path = "input.jpg"
img = cv2.imread(img_path)
 
resp = RetinaFace.detect_faces(img_path, threshold = 0.5)
st.write("faces:" + str(len(resp)))
 
def int_tuple(t):
    return tuple(int(x) for x in t)
 
for key in resp:
    identity = resp[key]
 
    #---------------------
 
    landmarks = identity["landmarks"]
    diameter = 1
    cv2.circle(img, int_tuple(landmarks["left_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["right_eye"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["nose"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_left"]), diameter, (0, 0, 255), -1)
    cv2.circle(img, int_tuple(landmarks["mouth_right"]), diameter, (0, 0, 255), -1)
 
    facial_area = identity["facial_area"]
    cv2.rectangle(img, (facial_area[2], facial_area[3]), (facial_area[0], facial_area[1]), (255, 255, 255), 1)
    #facial_img = img[facial_area[1]: facial_area[3], facial_area[0]: facial_area[2]]
    #plt.imshow(facial_img[:, :, ::-1])
st.image(img)
#cv2.imwrite('output.'+img_path.split(".")[1], img)
 
#------------------------------
#alignment
img_path = "dataset/img11.jpg"
 
resp = RetinaFace.extract_faces(img_path = img_path, align = True)
 
for img in resp:
    plt.imshow(img[:, :, ::-1])
    plt.axis('off')
    plt.show()
    cv2.imwrite('outputs/'+img_path.split("/")[1], img)
'''
