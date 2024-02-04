import firebase_admin
from firebase_admin import credentials
from firebase_admin import db


cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://facial-attendance-6834e-default-rtdb.asia-southeast1.firebasedatabase.app/"

})
ref = db.reference('Students')

data ={
    "2038":
        {
            "name": "Aarav Joshi",
            "major": "Csit",
            "starting_year": 2076,
            "standing": "E",
            "total_attendance": 8,
            "Sem": "7",
            "last_attendance_time": "2024-01-01 10:00:00"

        },
    "321":
        {
            "name": "Joshi Aarav",
            "major": "Csit",
            "starting_year": 2076,
            "standing": "P",
            "total_attendance": 9,
            "Sem": "8",
            "last_attendance_time": "2024-01-01 10:00:00"

        },
    "123":
        {
            "name": "Elon Musk",
            "major": "Csit",
            "starting_year": 2077,
            "standing": "G",
            "total_attendance": 8,
            "Sem": "6",
            "last_attendance_time": "2024-01-01 10:00:00"

        },
    "678":
        {
            "name": "Pawan Bhatt",
            "major": "Csit",
            "starting_year": 2077,
            "standing": "VG",
            "total_attendance": 1,
            "Sem": "7",
            "last_attendance_time": "2024-01-01 10:00:00"

        }
}

#--For sending data to DB--
for key,value in data.items():
    ref.child(key).set(value)
