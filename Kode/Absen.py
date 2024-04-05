import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import keyboard

path = 'Absen'
images = []
classNames = []
myList = os.listdir(path)

#label untuk tiap berkas

label_mapping = {
    '1.jpg': {'name': 'marco','nim': '14S21025'},
    '2.jpg': {'name': 'rizki','nim': '14S21040'},
    '3.jpg': {'name': 'perez','nim': '14S21038'},
}

for cl in myList:
    if cl.lower().endswith(('.jpg', '.jpeg', '.png')):
        curImg = cv2.imread(f'{path}/{cl}')
        images.append(curImg)

        if cl in label_mapping:
            data = label_mapping[cl]
            classNames.append({'name': data['name'], 'nim': data['nim']})

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def markAttendance(name, nim):
    with open('Absensi.txt', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'{name}, {nim}, {dtString}\n')

encodeListKnown = findEncodings(images)
print('Encoding Complete')

cap = cv2.VideoCapture(0)
camera_active = True  # Menambahkan variabel untuk memantau apakah kamera aktif

while camera_active:
    if keyboard.is_pressed('q'):  # Mengecek apakah tombol 'q' ditekan
        camera_active = False  # Menonaktifkan kamera dan keluar dari loop
        break

    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            data = classNames[matchIndex]
            name = data['name'].upper()
            nim = data['nim']
        y1, x2, y2, x1 = faceLoc
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        markAttendance(name, nim)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()