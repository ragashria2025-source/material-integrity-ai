from ultralytics import YOLO

model = YOLO("models/weld_detector/weld_detector_best.pt")

results = model.val(
    data="dataset/weld_surface/data.yaml",
    split="test",
    imgsz=640,
    conf=0.25
)

print("\n===== MODEL EVALUATION =====")
print(f"mAP50:     {results.box.map50:.4f}")
print(f"mAP50-95:  {results.box.map:.4f}")
print(f"Precision: {results.box.mp:.4f}")
print(f"Recall:    {results.box.mr:.4f}")
print("============================")