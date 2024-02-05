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
import aiohttp
import asyncio

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facial-attendance-6834e-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facial-attendance-6834e.appspot.com"
})

# Initialize Firebase Storage bucket
bucket = storage.bucket()

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Load background image
imgBackground = cv2.imread('Resources/background.png')

# Load modes(images) into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = [cv2.imread(os.path.join(folderModePath, path)) for path in modePathList]

# Load face encodings from the Encodefile
print("Loading Encode File...")
with open('Encodefile.p', 'rb') as file:
    encodelistKnownwithIds = pickle.load(file)
encodelistKnown, studentIds = encodelistKnownwithIds
print("Encode File Loaded")

# Initialize variables
modeType = 0
counter = 0
id = -1
imgStudent = []

# Asynchronous function to download image from storage
async def download_image_from_storage(student_id):
    blob = bucket.get_blob(f'Images/{student_id}.jpg')
    async with aiohttp.ClientSession() as session:
        async with session.get(blob.media_link) as resp:
            array = np.frombuffer(await resp.read(), np.uint8)
    return cv2.imdecode(array, cv2.COLOR_BGR2RGB)

# Asynchronous function to download student info from the Realtime DB
async def download_student_info(student_id):
    return db.reference(f'Students/{student_id}').get()

# Asynchronous webcam thread
async def webcam_thread():
    global cap, imgBackground, modeType, counter, id, imgStudent

    frame_counter = 0
    while True:
        success, img = cap.read()

        imgS = cv2.cvtColor(cv2.resize(img, (0, 0), None, 0.25, 0.25), cv2.COLOR_BGR2RGB)

        frame_counter += 1
        if frame_counter % 2 == 0:
            continue

        faceCurFrame = face_recognition.face_locations(imgS, number_of_times_to_upsample=1)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        imgBackground[162:162 + 480, 55:55 + 640, :] = img
        imgBackground[44:44 + 633, 808:808 + 414, :] = imgModeList[modeType]

        if faceCurFrame:
            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodelistKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodelistKnown, encodeFace)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    id = studentIds[matchIndex]
                    print(id)

                    if counter == 0:
                        counter = 1
                        modeType = 1

            if counter != 0:
                if counter == 1:
                    student_info_task = asyncio.create_task(download_student_info(id))
                    student_info = await student_info_task
                    print(student_info)

                    img_student_task = asyncio.create_task(download_image_from_storage(id))
                    imgStudent = await img_student_task

                    datetimeObj = datetime.strptime(student_info['last_attendance_time'], "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObj).total_seconds()
                    print(secondsElapsed)

                    if secondsElapsed > 30:
                        ref = db.reference(f'Students/{id}')
                        student_info['total_attendance'] += 1
                        ref.child('total_attendance').set(student_info['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:
                    if 10 < counter < 20:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        cv2.putText(imgBackground, str(student_info['total_attendance']), (861, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(student_info['major']), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.5,
                                    (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                        cv2.putText(imgBackground, str(student_info['standing']), (910, 625), cv2.FONT_HERSHEY_COMPLEX,
                                    0.6, (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(student_info['Sem']), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                                    (100, 100, 100), 1)
                        cv2.putText(imgBackground, str(student_info['starting_year']), (1125, 625),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (100, 100, 100), 1)

                        (w, h), _ = cv2.getTextSize(student_info['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(student_info['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        if imgStudent is not None:
                            imgBackground[175:175 + 216, 909:909 + 216] = imgStudent

                    counter += 1

                    if counter >= 20:
                        counter = 0
                        modeType = 0
                        student_info = []
                        imgStudent = []
                        imgBackground[44:44 + 633, 808:808 + 414, :] = imgModeList[modeType]

        else:
            modeType = 0
            counter = 0

        cv2.imshow("Face Attendance", imgBackground)
        cv2.waitKey(1)

# Main function to run asynchronous tasks
async def main():
    await asyncio.gather(
        webcam_thread(),
        # Add other asynchronous tasks here if any
    )

# Run the main function
if __name__ == "__main__":
    asyncio.run(main())
