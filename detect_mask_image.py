# Su dung
# python detect_mask_image.py --image examples/example_01.png

# Khai bao thu vien
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import argparse
import pickle
import cv2
import os

# Cung cap doi so cho chuong trinh
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-f", "--face", type=str,
	default="models",
	help="path to face detector model directory")
ap.add_argument("-m", "--model", type=str,
	default="models/mask_detector.model",
	help="path to trained face mask detector model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Tai model phat hien khuon mat
print("[INFO] loading face detector model...")
prototxtPath = os.path.sep.join([args["face"], "deploy.prototxt"])
weightsPath = os.path.sep.join([args["face"],
	"res10_300x300_ssd_iter_140000.caffemodel"])
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("models/recognizer.pickle", "rb").read())
le = pickle.loads(open("models/le.pickle", "rb").read())

# Tai model phan loai deo khau trang voi khong deo khau trang
print("[INFO] loading face mask detector model...")
model = load_model(args["model"])

# Doc anh dau vao, sao chep, lay kich thuoc anh
image = cv2.imread(args["image"])
orig = image.copy()
(h, w) = image.shape[:2]

# chuyen anh sang blob de tranh anh bi nhieu sang
blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
	(104.0, 177.0, 123.0))

# cho blob vao mo hinh phat hien khuon mat
print("[INFO] computing face detections...")
net.setInput(blob)
detections = net.forward()

# lap qua cac phat hien khuon mat
for i in range(0, detections.shape[2]):
	# trich xuat do tin cay (i.e., phan tram) cua phat hien
	# khuon mat
	confidence = detections[0, 0, i, 2]

	# loc cac phat hien yeu bang cach dam bao do tin cay lon hon
	# do tin cay toi thieu
	if confidence > args["confidence"]:
		# tinh toa do (x, y)-cua hop gioi han cho doi tuong
		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		(startX, startY, endX, endY) = box.astype("int")

		# dam bao cac hop gioi han nam trong kich thuoc cua
		# frame anh
		(startX, startY) = (max(0, startX), max(0, startY))
		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

		# Trich xuat ROI cua khuon mat, chuyen doi tu he mau BGR qua
		# he mau RGB, thay doi kich thuoc ve 224 x 224 va xu ly truoc
		face = image[startY:endY, startX:endX]
		(fH, fW) = face.shape[:2]

		# ensure the face width and height are sufficiently large
		if fW < 20 or fH < 20:
			continue

		# construct a blob for the face ROI, then pass the blob
		# through our face embedding model to obtain the 128-d
		# quantification of the face
		faceBlob = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96),
			(0, 0, 0), swapRB=True, crop=False)
		embedder.setInput(faceBlob)
		vec = embedder.forward()

		# perform classification to recognize the face
		preds = recognizer.predict_proba(vec)[0]
		j = np.argmax(preds)
		proba = preds[j]
		name = le.classes_[j]

		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
		face = cv2.resize(face, (224, 224))
		face = img_to_array(face)
		face = preprocess_input(face)
		face = np.expand_dims(face, axis=0)

		# Dua face vao mo hinh de xac dinh xem co khau trang hay khong
		(mask, withoutMask) = model.predict(face)[0]

		# Tao nhan va mau sac de ve hop gioi han va ten nhan
		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0)
		else:
			label = "{} No Mask: {:.2f}%".format(name, max(mask, withoutMask) * 100)
			color = (0, 0, 255)

		# label = "Mask" if mask > withoutMask else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Them xac suat phat hien trong nhan 
		# label = "{} {}: {:.2f}%".format(name ,label, max(mask, withoutMask) * 100)

		# hien thi nhan va bounding box hinh chu nhat cho frame dau ra
		cv2.putText(image, label, (startX - 15, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

# Hien thi anh dau ra
cv2.imshow("Output", image)
cv2.waitKey(0)