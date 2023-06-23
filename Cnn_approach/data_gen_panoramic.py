
import cv2
import os
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector

IMG_SIZE =(616, 408)

#crop eye region rec
def crop_eye(gray, eye_points):
  x1, y1 = np.amin(eye_points, axis=0)
  x2, y2 = np.amax(eye_points, axis=0)
  cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

  w = (x2 - x1) * 1.2
  h = w * IMG_SIZE[1] / (IMG_SIZE[0])

  margin_x, margin_y = w / 2, h / 4

  min_x, min_y = int(cx - margin_x), int(cy - margin_y)
  max_x, max_y = int(cx + margin_x), int(cy + margin_y)

  eye_rect = np.rint([min_x, min_y, max_x, max_y]).astype(np.integer)

  eye_img = gray[eye_rect[1]:eye_rect[3], eye_rect[0]:eye_rect[2]]

  return eye_img, eye_rect

#save each frame of video 
def save_img(user,imgl,count):
  user = user - 1
  dir = "eye_images/"+ str(user)
  if not os.path.exists(dir):
    os.makedirs(dir)

  cv2.imwrite(os.path.join(dir, f"eye_{count:04d}.jpg"), imgl)
  


#save one image for each eye for testing (valid for testing one user)
def save_im(imgr,imgl):
  dir = "test/"
  if not os.path.exists(dir):
    os.makedirs(dir)

  cv2.imwrite(os.path.join(dir, "left_eye.jpg"), imgl)
  cv2.imwrite(os.path.join(dir, "right_eye.jpg"), imgr)

def enrole(path,i,count):

  cap = cv2.VideoCapture(path)

  while cap.isOpened():
    ret, img_ori = cap.read()

    img_ori = cv2.resize(img_ori, dsize=(0, 0), fx=0.5, fy=0.5)

    img = img_ori.copy()
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img, faces = detector.findFaceMesh(img, draw=False)

    for face in faces:
   

      x1 = face[33]
      x2 = face[160]
      x3 = face[387]
      x4 = face[263]
      x5 = face[373]
      x6 = face[144]


      shape_r = [x1,x2,x3,x4,x5,x6]
      
      eye_img_l, eye_rect_l = crop_eye(gray, eye_points=shape_r)
      eye_img_l = cv2.resize(eye_img_l, dsize=IMG_SIZE)


      #cv2.imshow('l', eye_img_l)
      #cv2.imshow('r', eye_img_r)

      #frame count to use it as index for images(to avoid overriting )
      count+=1

      save_img(i,eye_img_l,count)
      #save_im(eye_img_r,eye_img_l)

      cv2.rectangle(img, pt1=tuple(eye_rect_l[0:2]), pt2=tuple(eye_rect_l[2:4]), color=(255,255,255), thickness=2)

    cv2.imshow('result', img)
    if cv2.waitKey(1) == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()
  return count

# main
detector = FaceMeshDetector(maxFaces=1)

i = 1
while i <= 1:
   count = 0
   j = 1
   while j < 11:
       print(count)
       video = "caps/user" + str(i) + "_cap" + str(j) + ".mp4"
       print(video)
       count = enrole(video, i, count)
       j += 1
   i += 1


