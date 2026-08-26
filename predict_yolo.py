from ultralytics import YOLO

MODEL_PATH = r"models\weld_detector\weld_detector_best.pt"

IMAGE_PATH = r"C:\Users\Raga\OneDrive\Desktop\MaterialIntegrityAI\test_images\BT-9-85-_jpg.rf.f6e7676ad5ee1277af22319cc377110b.jpg"

print("Loading trained model...")

model = YOLO(MODEL_PATH)

print("Running prediction...")

results = model.predict(
    source=IMAGE_PATH,
    conf=0.25,
    save=True
)

print("Prediction completed.")