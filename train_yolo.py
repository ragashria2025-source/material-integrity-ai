from ultralytics import YOLO

print("Loading YOLO model...")

model = YOLO("yolov8n.pt")

print("Starting weld defect training...")

model.train(
    data="dataset/weld_surface/data.yaml",
    epochs=10,
    imgsz=640,
    batch=8,
    device="cpu"
)

print("Training completed!")