from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import datetime
import os
import pymysql

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/upload/")
async def upload_action(request: Request, t1: str = Form(...), t2: str = Form(...), t3: str = Form(...), t4: str = Form(...), t5: UploadFile = File(...)):
    cascPath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascPath)
    option = 0
    status = 'Child not found in missing database'
    
    contents = await t5.read()
    nparr = np.fromstring(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray,1.3,5)
    print("Found {0} faces!".format(len(faces)))
    img = ''
    if len(faces) > 0:
        for (x, y, w, h) in faces:
            img = frame[y:y + h, x:x + w]
            option = 1
    if option == 1:
        with open('model/model.json', "r") as json_file:
            loaded_model_json = json_file.read()
            missing_child_classifier = model_from_json(loaded_model_json)
        missing_child_classifier.load_weights("model/model_weights.h5")
        missing_child_classifier.make_predict_function()   
        img = cv2.resize(img, (64,64))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1,64,64,3)
        img = np.asarray(im2arr)
        img = img.astype('float32')
        img = img/255
        preds = missing_child_classifier.predict(img)
        if(np.amax(preds) > 0.60):
            status = 'Child found in missing database'
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(t5.filename)
    # db_connection = pymysql.connect(host='127.0.0.1',port = 3306,user = 'root', password = 'root', database = 'MissingChildDB',charset='utf8')
    # db_cursor = db_connection.cursor()
    # query = "INSERT INTO missing(person_name,child_name,contact_no,location,image,upload_date,status) VALUES('"+t1+"','"+t2+"','"+t3+"','"+t4+"','"+filename+"','"+str(current_time)+"','"+status+"')"
    # db_cursor.execute(query)
    # db_connection.commit()
    # print(db_cursor.rowcount, "Record Inserted")
    context= {'data':'Thank you for uploading. '+status}
    return JSONResponse(content=context)