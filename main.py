import os
import pickle
from datetime import datetime
import cv2
import face_recognition
import numpy as np
import cvzone
import firebase_admin
from firebase_admin import credentials, db, storage
import threading
import cProfile

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facial-attendance-6834e-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facial-attendance-6834e.appspot.com"
})

bucket = storage.bucket()  # --Initialising bucket method for getting image from Storage

# For webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# for adding background to webcam
imgBackground = cv2.imread('Resources/background.png')

# importing modes(images) into our list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
# print(len(imgModeList))   ----For Testing----

# Loading the Encodings which we generated in EncodeGenerator.py
print("Loading Encode File...")
file = open('Encodefile.p', 'rb')  # ---Opening File ----
encodelistKnownwithIds = pickle.load(file)  # ---add all info of file in there---
file.close()
encodelistKnown, studentIds = encodelistKnownwithIds  # ---Separating encodings with ids which we joined in EncodeGenerator.py
# print(studentIds) ----For Testing---
print("Encode File Loaded")

modeType = 0
counter = 0  # For downloading from data base only once
id = -1
imgStudent = []

# Thread for webcam
def webcam_thread():
    global cap, imgBackground, modeType, counter, id, imgStudent

    frame_counter = 0
    while True:
        success, img = cap.read()

        # ----For reducing size of image cause it takes less computation--
        imgS = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)

        # Frame skipping logic
        frame_counter += 1
        if frame_counter % 2 == 0:
            continue  # Skip face recognition on every other frame

        # ---Determining faces and their encodings in Web cam frame
        faceCurFrame = face_recognition.face_locations(imgS, number_of_times_to_upsample=1)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # for overlapping background/modes and webcam
        imgBackground[162:162 + 480, 55:55 + 640, :] = img
        imgBackground[44:44 + 633, 808:808 + 414, :] = imgModeList[modeType]

        # --When attendance marked get mode to mode 0 or active mode
        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)

                # ---- Extracting index of the img with the lowest distance value
                matchIndex = np.argmin(faceDis)

                # --- Making sure the least distance is detected as match
                if matches[matchIndex]:
                    # ----Creating Reactangle box using cvzone can be also created using cv2
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4  # --Resizing so that bbox doesn't appear small
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]
                    print(id)
                    # --For changing modes
                    if counter == 0:
                        counter = 1
                        modeType = 1

            if counter != 0:
                # --For Switching to mode 1 while Detecting Face---
                if counter == 1:
                    # --Downloading Student info from  Realtime DB---
                    studentInfo = db.reference(f'Students/{id}').get()
                    print(studentInfo)
                    # --Getting image from storage--
                    blob = bucket.get_blob(f'Images/{id}.jpg')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgStudent = cv2.imdecode(array, cv2.COLOR_BGR2RGB)

                    # --Updating the attendance and making sure attendance is marked daily once or after some time
                    datetimeObj = datetime.strptime(studentInfo['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObj).total_seconds()
                    print(secondsElapsed)

                    # ---After counter 30 only it can mark new attendance,
                    # for demonstration purpose IRL it can be a day or hours
                    if secondsElapsed > 30:
                        # --Creating reference to the specific student
                        ref = (db.reference(f'Students/{id}'))

                        # ---Updating total attendance
                        studentInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(studentInfo['total_attendance'])
                        # ---Updating last attendance time--
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                # --So that image, name etc don't show on mode 3 when students' attendance is already marked
                if modeType != 3:
                    # --Changing Background mode(to marked) after some delay using counter after successfully taking attendance
                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    # --for adding info in mode 1
                    if counter <= 10:
                        # --Putting info from Realtime DB into mode
                        cv2.putText(imgBackground, str(studentInfo['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['major']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(studentInfo['standing']), (910, 625), cv2.FONT_HERSHEY_COMPLEX,
                                    0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['Sem']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(studentInfo['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        # --Centering the name in mode--
                        (w, h), _ = cv2.getTextSize(studentInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1,
                                                    1)  # --Getting width of name
                        offset = (414 - w) // 2  # Subtracting width of text from total mode component and dividing by 2.
                        # This will center name.
                        cv2.putText(imgBackground, str(studentInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        # --Showing image of student in mode
                        imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1

                    # ---After marked mode changing mode back active mode (starting mode) basically resetting modes
                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        studentInfo = []
                        imgStudent = []
                        imgBackground[44:44 + 633, 808:808 + 414, :] = imgModeList[modeType]

        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

# Start webcam thread
webcam_thread = threading.Thread(target=webcam_thread, daemon=True)
webcam_thread.start()


# Main loop (if any)
while True:
    # Perform any main thread operations here
    pass
