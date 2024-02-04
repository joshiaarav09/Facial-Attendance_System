#All the encodings will be generated here

import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import db
from firebase_admin import storage
from firebase_admin import credentials


#---For storing images in DB
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://facial-attendance-6834e-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "facial-attendance-6834e.appspot.com"

})

#importing the student images into  list for encoding purposes
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
studentIds = [] # for importing student IDs/names from photos
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath,path)))
    studentIds.append(os.path.splitext(path)[0]) #for separating id part and '.' format part and adding to our empty student Ids list

    #To send images to DB
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


print(studentIds)


#---Generating encodings
#---All the student images will be provided via loop and encodings will be stored as list
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #---converting BGR to RGB
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
print("Encoding starting")
encodelistKnown = findEncodings(imgList)
encodelistKnownwithIds = [encodelistKnown, studentIds]
print("Encoding Complete")


#Saving Encodings in file
file = open("Encodefile.p", 'wb')
pickle.dump(encodelistKnownwithIds,file) #--Picke used to dump into files
file.close()
print("File Saved")
