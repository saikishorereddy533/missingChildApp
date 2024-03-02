from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
import datetime
import os
import pymysql
from fastapi.templating import Jinja2Templates
import cloudinary
import cloudinary.api
import cloudinary.uploader

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")

cloudinary.config(
                cloud_name="ddy07zvko",
                api_key="883112828989144",
                api_secret="cpXSBlg_BpEGaBfnw5o2wDN3ThA"
            )

valid_username = "admin"
valid_password = "password"

@app.get("/login",response_class=HTMLResponse)
async def getLogin(request: Request):
    return templates.TemplateResponse("login.html",{"request": request})

@app.post("/login/", response_class=HTMLResponse)
async def login(request: Request, username: str = Form(...), password: str = Form(...)):
    if username == valid_username and password == valid_password:
        return templates.TemplateResponse("upload.html", {"request": request, "message": "Login successful!"})
    else:
        return templates.TemplateResponse("login.html", {"request": request, "message": "Invalid credentials. Please try again."})

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
        else:
            result = cloudinary.uploader.upload(contents, folder="missing_children")
            status = f"Image uploaded to Cloudinary. Public URL: {result['secure_url']}"
    now = datetime.datetime.now()
    current_time = now.strftime("%Y-%m-%d %H:%M:%S")
    filename = os.path.basename(t5.filename)
    context = {'data': f"Thank you for uploading. {status}"}
    return JSONResponse(content=context)

@app.get("/images/", response_class=HTMLResponse)
async def list_images(request: Request):
    images = cloudinary.api.resources(type="upload", prefix="missing_children")

    image_html = ""
    for image in images["resources"]:
        image_url = image["secure_url"]
        image_html += f'<img src="{image_url}" alt="Image" style="max-width: 300px; margin: 10px;">'

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Stored Images</title>
    </head>
    <body>
        <h1>Stored Images</h1>
        {image_html}
    </body>
    </html>
    """

    return HTMLResponse(content=html_content)