import cv2
import dlib
import numpy as np
import time
import csv
from imutils import face_utils

# ------------------------
# Configuración
# ------------------------
Puntos_Ojo_Izquierdo = list(range(36, 42))  # dlib points
Puntos_Ojo_Derecho = list(range(42, 48))

# Archivo CSV para guardar fijaciones
CSV_FILENAME = "fixations.csv"

# ------------------------
# Inicializar detector y predictor
# ------------------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat") 
# Descarga desde: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

cap = cv2.VideoCapture(0)

# Variables de fijación
last_gaze = None
fix_start = None
FIXATION_THRESHOLD = 30  # píxeles de tolerancia
FIXATION_TIME = 0.5      # segundos

# Preparar CSV
with open(CSV_FILENAME, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "x", "y", "fixation"])

def centroid(eye_points):
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    return int(np.mean(xs)), int(np.mean(ys))

print("Presiona 'q' para salir...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    gaze_point = None

    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        eye_left = shape[Puntos_Ojo_Izquierdo]
        eye_right = shape[Puntos_Ojo_Derecho]

        # Calcular centro de los ojos
        left_center = centroid(eye_left)
        right_center = centroid(eye_right)
        gaze_point = ((left_center[0] + right_center[0]) // 2,
                      (left_center[1] + right_center[1]) // 2)

        # Dibujar ojos y centro
        cv2.polylines(frame, [eye_left], True, (0,255,0), 1)
        cv2.polylines(frame, [eye_right], True, (0,255,0), 1)
        cv2.circle(frame, gaze_point, 5, (0,0,255), -1)

    # Determinar fijación
    fixation = False
    timestamp = time.time()

    if gaze_point is not None:
        if last_gaze is not None:
            dist = np.linalg.norm(np.array(gaze_point) - np.array(last_gaze))
            if dist < FIXATION_THRESHOLD:
                if fix_start is None:
                    fix_start = timestamp
                elif (timestamp - fix_start) >= FIXATION_TIME:
                    fixation = True
            else:
                fix_start = timestamp
        else:
            fix_start = timestamp
        last_gaze = gaze_point

    # Guardar datos
    if gaze_point is not None:
        with open(CSV_FILENAME, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, gaze_point[0], gaze_point[1], fixation])

    # Mostrar resultado
    if gaze_point is not None:
        color = (0,255,0) if fixation else (0,0,255)
        cv2.circle(frame, gaze_point, 15, color, 2)

    cv2.imshow("Eye Tracker", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Datos guardados en {CSV_FILENAME}")
