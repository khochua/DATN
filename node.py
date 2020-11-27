	# lay khung hinh tu luong video stream va thay doi
	# kich thuoc chieu rong toi da la 400 pixels
	k, frame = vs.read()
	frame = cv2.resize(frame, (200,400))
