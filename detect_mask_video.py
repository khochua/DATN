# Su dung
# python detect_mask_video.py

# Cai dat cac thu vien can thiet
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import pickle
import time
import cv2
import os

def detect_and_predict_mask(frame, faceNet, maskNet):
	# lay kich thuoc cua frame va khoi tao blob
	# cho frame
	(h, w) = frame.shape[:2]
	blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
		(104.0, 177.0, 123.0))

	# Cho blob vao mo hinh phat hien khuon mat
	faceNet.setInput(blob)
	detections = faceNet.forward()

	# Khoi tao lists chua anh khuon mat, vi tri tuong ung cua khuon mat,
	# danh sach cac du doan khuon mat
	faces = []
	locs = []
	preds = []
	name = ""

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
			face = frame[startY:endY, startX:endX]
			# ensure the face width and height are sufficiently large
			# if fW < 20 or fH < 20:
			# 	continue

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

			# chuyen doi tu he mau BGR qua he mau RGB, thay doi kich thuoc
			# ve 224 x 224 va xu ly truoc
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
			face = cv2.resize(face, (224, 224))
			face = img_to_array(face)
			face = preprocess_input(face)

			# Them khuon mat vao faces va toa do hop gioi han vao locs
			faces.append(face)
			locs.append((startX, startY, endX, endY))

	# Chi dua ra du doan neu mot khuon mat duoc phat hien
	if len(faces) > 0:
		# dua ra du doan ve tat ca *all*
		# khuon mat cung mot luc thay ve du doan tung khuong mat
		# trong vong lap `for` o tren
		faces = np.array(faces, dtype="float32")
		preds = maskNet.predict(faces, batch_size=32)

	# tra ve 2 bo vi tri khuon mat va vi tri tuong ung
	# cua chung
	return (locs, preds, name)

# Cung cap doi so cho chuong trinh
ap = argparse.ArgumentParser()
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
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load our serialized face embedding model from disk
print("[INFO] loading face recognizer...")
embedder = cv2.dnn.readNetFromTorch("models/openface_nn4.small2.v1.t7")

# load the actual face recognition model along with the label encoder
recognizer = pickle.loads(open("models/recognizer.pickle", "rb").read())
le = pickle.loads(open("models/le.pickle", "rb").read())

# Tai model phan loai deo khau trang voi khong deo khau trang
print("[INFO] loading face mask detector model...")
maskNet = load_model(args["model"])

# Khoi tao luong video va bat camera
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# khoi tao voi file MP4
# vs = cv2.VideoCapture("examples/1.mp4")

# Lap qua cac khung hinh tu luong Video
while True:
	# lay khung hinh tu luong video stream va thay doi
	# kich thuoc chieu rong toi da la 400 pixels
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# xac dinh khuon mat trong khung va xac dinh xem ho co deo khau
	# trang hay khong
	(locs, preds, name) = detect_and_predict_mask(frame, faceNet, maskNet)

	# lap qua vi tri cac khuon mat duoc phat hien va vi tri tuong ung
	# cua chung
	for (box, pred) in zip(locs, preds):
		# Giai nen hop gioi han va du doan
		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred

		# Tao nhan va mau sac de ve hop gioi han va ten nhan
		if mask > withoutMask:
			label = "Mask"
			color = (0, 255, 0)
		else:
			label = "{} No Mask: {:.2f}%".format(name, max(mask, withoutMask) * 100)
			color = (0, 0, 255)

		# label = "Mask" if mask > withoutMask else "No Mask"
		# color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

		# Them xac suat xuat hien trong nhan
		# label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

		# hien thi nhan va hop gioi han khuon mat
		# frame
		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	# hien thi cac frame
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# nhan `q` de thoat khoi vong lap
	if key == ord("q"):
		break

# Dong tat ca cac cua so
cv2.destroyAllWindows()
vs.stop()