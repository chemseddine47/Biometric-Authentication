import cv2
import numpy as np
from keras.models import load_model
from cvzone.FaceMeshModule import FaceMeshDetector

IMG_SIZE = (64, 56)
thr = 0.99

#crops rectangle of eye region
def crop_eye(gray, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / IMG_SIZE[0]

  margin_x, margin_y = w / 2, h / 2

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.integer)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

model = load_model("model_large.h5")

cap = cv2.VideoCapture("caps/user18_cap1.mp4")
detector = FaceMeshDetector(maxFaces=1)

while cap.isOpened():
    
    ret, img_ori = cap.read()
    
    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img, faces = detector.findFaceMesh(img, draw=False)

    for face in faces:


        x1 = face[130]
        x2 = face[29]
        x3 = face[28]
        x4 = face[243]
        x5 = face[22]
        x6 = face[24]

        y1 = face[463]
        y2 = face[258]
        y3 = face[259]
        y4 = face[359]
        y5 = face[254]
        y6 = face[252]


        shape_r = [x1,x2,x3,x4,x5,x6]
        shape_l = [y1,y2,y3,y4,y5,y6]
        eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shape_r)
        eye_img_r, eye_rect_r = crop_eye(gray, eye_points=shape_l)

        eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)
        eye_img_r = cv2.resize(eye_img_r, dsize=IMG_SIZE)
        #eye_img_r = cv2.flip(eye_img_r, flipCode=1)

        eye_img_l = eye_img_l / 255.0
        eye_img_l = np.array(eye_img_l)
        eye_img_l = eye_img_l.reshape(1, 56, 64, 1)

        eye_img_r = eye_img_r / 255.0
        eye_img_r = np.array(eye_img_r)
        eye_img_r = eye_img_r.reshape(1, 56, 64, 1)

        

        predictions = model.predict(eye_img_r)
        predictionsl = model.predict(eye_img_l)
        
        resultr = "intruder r" if np.max(predictions, axis=1) < thr else "user r"+str(np.argmax(predictions, axis=1))
        resultl = "intruder l" if np.max(predictionsl, axis=1) < thr else "user l"+str(np.argmax(predictionsl, axis=1))

        cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)
        cv2.rectangle(img, pt1=tuple(eye_rect_r[0:2]), pt2=tuple(eye_rect_r[2:4]), color=(255,255,255), thickness=2)

        cv2.putText(img, resultl, tuple(eye_rect_l[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(img, resultr, tuple(eye_rect_r[0:2]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
 
        cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
