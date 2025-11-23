import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, Response
import mediapipe as mp
from main_2 import router
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
origins = ["https://mtakira.github.io","https://Oshota501.github.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["*"],  # 広げてプリフライト失敗要因を除去
    allow_headers=["*"],
)

app.include_router(router)

mp_face_mesh = mp.solutions.face_mesh


def get_human_points_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return np.empty((0, 6), dtype=np.float32)

    height, width = image.shape[:2]
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1) as face_mesh:
        results = face_mesh.process(image_rgb)

    if not results.multi_face_landmarks:
        return np.empty((0, 6), dtype=np.float32)

    face_landmarks = results.multi_face_landmarks[0]

    pts = []
    for lm in face_landmarks.landmark:
        x = float(lm.x)
        y = float(lm.y)
        z = float(lm.z)
        px = int(np.clip(round(x * (width - 1)), 0, width - 1))
        py = int(np.clip(round(y * (height - 1)), 0, height - 1))
        r, g, b = image_rgb[py, px].astype(np.float32) / 255.0
        pts.append([x, y, z, r, g, b])

    return np.array(pts, dtype=np.float32)


def get_object_points_from_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if color is None:
        return np.empty((0, 6), dtype=np.float32)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    height, width = color.shape[:2]
    detector = cv2.AKAZE_create()
    keypoints = detector.detect(gray, None)

    if not keypoints:
        return np.empty((0, 6), dtype=np.float32)

    pts = []
    for kp in keypoints:
        x_px, y_px = kp.pt
        x = float(x_px / (width - 1))
        y = float(y_px / (height - 1))
        px = int(np.clip(round(x_px), 0, width - 1))
        py = int(np.clip(round(y_px), 0, height - 1))
        b, g, r = color[py, px].astype(np.float32) / 255.0
        z = 0.0
        pts.append([x, y, z, r, g, b])

    return np.array(pts, dtype=np.float32)


@app.post("/pointcloud")
async def pointcloud(file: UploadFile = File(...)):
    image_bytes = await file.read()
    human_pts = get_human_points_from_bytes(image_bytes)

    if human_pts.shape[0] > 0:
        logging.info("face detected: %d points", human_pts.shape[0])
        return Response(content=human_pts.tobytes(), media_type="application/octet-stream")

    object_pts = get_object_points_from_bytes(image_bytes)
    logging.info("object kp: %d points", object_pts.shape[0])
    return Response(content=object_pts.tobytes(), media_type="application/octet-stream")


# 互換: 以前の名称を使っているフロントを救済
@app.post("/pointcloud2")
async def pointcloud2(file: UploadFile = File(...)):
    return await pointcloud(file)


@app.get("/")
async def testendpoint():
    return Response(content="success", media_type="text/plain")
