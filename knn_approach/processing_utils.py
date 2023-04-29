import time
import cv2
import cvzone
import matplotlib.pyplot as plt
import numpy as np
from cvzone.FaceMeshModule import FaceMeshDetector
from cvzone.PlotModule import LivePlot
from numpy.fft import fft
import os
from knn_engine import ord_users

# landmarks des yeux
right_eye = [145, 33, 159, 133]
left_eye = [362, 374, 362, 263]

detector = FaceMeshDetector(maxFaces=1)
plot1 = LivePlot(384, 216, [30, 55], invert=True)
plot2 = LivePlot(384, 216, [30, 55], invert=True)

right_list = []
left_list = []


# compteur du temps
def calculate_time(a):
    a = time.time() - a
    return round(a, 2)

# enregistrement de la séquence vidéo
def sequence_video(a, x):
    action = []
    action2 = []
    temps = []
    h = float(time.time())
    cam = cv2.VideoCapture(a)

    while True:

        t = calculate_time(h)
        temps.append(t)

        ret, frame = cam.read()

        if not ret: 
            continue

        frame, faces = detector.findFaceMesh(frame, draw=False)
   
        if faces:
            face = faces[0]
            for i in right_eye:
                cv2.circle(frame, face[i], 1, (255, 255, 255), cv2.FILLED)
            for i in left_eye:
                cv2.circle(frame, face[i], 1, (255, 255, 255), cv2.FILLED)

            right_up = face[159]
            right_down = face[145]
            right_right = face[133]
            right_left = face[33]

            left_up = face[386]
            left_down = face[374]
            left_right = face[362]
            left_left = face[263]

            right_ver, _ = detector.findDistance(right_down, right_up)
            right_hor, _ = detector.findDistance(right_right, right_left)

            left_ver, _ = detector.findDistance(left_down, left_up)
            left_hor, _ = detector.findDistance(left_right, left_left)

            cv2.line(frame, right_up, right_down, (255, 0, 255), 1)
            cv2.line(frame, right_right, right_left, (255, 0, 255), 1)

            cv2.line(frame, left_up, left_down, (255, 0, 255), 1)
            cv2.line(frame, left_right, left_left, (255, 0, 255), 1)

            right_overall = (right_ver / right_hor) * 100
            right_list.append(right_overall)
            if len(right_list) > 6:
                right_list.pop(0)
                
            right_overall = sum(right_list) / len(right_list)

            left_overall = (left_ver / left_hor) * 100
            left_list.append(left_overall)
            if len(left_list) > 6:
                left_list.pop(0)
            left_overall = sum(left_list) / len(left_list)

            action.append(round(right_overall, 2))
            action2.append(round(left_overall, 2))

            

            cvzone.putTextRect(frame, f'Time elapsed : {t}', (50, 100))

            right_plot = plot1.update(right_overall)
            left_plot = plot2.update(left_overall)
            resized = cv2.resize(frame, (384, 216))
            img = cvzone.stackImages([resized, right_plot, left_plot], 2, 1)

            cv2.imshow("CameraVR", img)
            key = cv2.waitKey(1)
            if key == ord('q') or t > x:
                break

    cam.release()
    cv2.destroyAllWindows()

    return temps, action, action2


# calcul de la mo
def avr_list(y):
    avr = sum(y) / len(y)
    i = 0
    new_list = []
    while i < len(y):
        new_list.append(abs(avr - y[i]))
        i += 1
    return new_list


# calcul du min, max
def max_min(p, y):
    max_value = None
    max_idx = None
    min_value = None
    min_idx = None
    avr = sum(y) / len(y)

    for idx, num in enumerate(y):
        if max_value is None or num > max_value:
            max_value = num
            max_idx = idx

    for idx, num in enumerate(y):
        if min_value is None or num < min_value:
            min_value = num
            min_idx = idx

    return p, max_idx, round(max_value, 2), min_idx, round(min_value, 2), round(avr, 2)


# calcul du fft
def fft_calcul(y, t):
    fe = 50
    tfd = fft(y)
    n = len(y)
    spectre = np.absolute(tfd) * 2 / n

    return spectre


# calcul des relatifs
def relatif_avr(y, coeff, signal=0):
    _, _, maxi, _, mini, avr = max_min('', y)
    nb = 0
    i = 0
    born = avr * coeff
    if (mini <= born) and (born <= maxi):
        while i < len(y):
            if signal == 1:
                if y[i] >= born:
                    nb += 1
            elif signal == 0:
                if y[i] <= born:
                    nb += 1
            i += 1
    else:
        print("Erreur!")
    return round(nb / len(y) * 100, 2)


# découper le lien
def splitting(adr):
    _, filename = os.path.split(adr)
    filename = '.'.join(filename.split('.')[:-1])
    user, cap = filename.split("_")
    return user, cap


# values de dict
def to_dict(user, cap, min_quotient, max_quotient, avr_quotient, max_fft, relatif_min, relatif_max):
    vect = {"Username": user,
            "Cap": cap,
            "Quotient minimum": min_quotient,
            "Quotient maximum": max_quotient,
            "Quotient moyen": avr_quotient,
            "FFT maximum": max_fft,
            "Relatif minimum": relatif_min,
            "Relatif maximum": relatif_max,
            }

    return vect


# qui à le plus d'occurences
def most_frequent(list):
    counter = 0
    num = list[0]

    for i in list:
        curr_frequency = list.count(i)
        if curr_frequency > counter:
            counter = curr_frequency
            num = i

    return num


def car_extraction(y, z, user, temps):
    _, _, max_quotient, _, min_quotient, avr_quotient = max_min(user, y)
    _, _, max_quotient2, _, min_quotient2, avr_quotient2 = max_min(user, z)
    y_fft = fft_calcul(y, temps)
    z_fft = fft_calcul(z, temps)
    _, _, max_fft, _, _, _ = max_min(user, y_fft)
    _, _, max_fft2, _, _, _ = max_min(user, z_fft)
    relatif_min = relatif_avr(y, 0.95, 0)
    relatif_max = relatif_avr(y, 1.04, 1)
    relatif_min2 = relatif_avr(z, 0.95, 0)
    relatif_max2 = relatif_avr(z, 1.04, 1)

    cible = min_quotient, max_quotient, avr_quotient, max_fft, relatif_min, relatif_max
    cible2 = min_quotient2, max_quotient2, avr_quotient2, max_fft2, relatif_min2, relatif_max2

    return cible, cible2


def row_to_list(table):
    liste = []

    for index, row in table.iterrows():
        vecteur = (row['Quotient minimum'], row['Quotient maximum'], row['Quotient moyen'],row['FFT maximum'],
                   row['Relatif minimum'],row['Relatif maximum'])

        point = {"Username": row['Username'], "Coordinates": vecteur}

        liste.append(point)

    return liste


def print_inf(seuil, values, liste):
    if len(seuil) != 0:
        users = ord_users(values, liste)
        #print(users)
        whoisit = most_frequent(users)
        #print('The user is:', whoisit)
    else:
        #print("Not in rang")
        whoisit = "null"
    return whoisit
