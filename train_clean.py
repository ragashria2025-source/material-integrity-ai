from ultralytics import YOLO

# Start from a pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train on the cleaned weld dataset
results = model.train(
    data="dataset/weld_surface_clean/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    patience=10,
    workers=2,
    project="runs",
    name="weld_clean",
    exist_ok=True
)

print("\nTraining finished!")
print("Best model:")
print("runs/weld_clean/weights/best.pt")