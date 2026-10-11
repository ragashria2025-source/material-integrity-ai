from ultralytics import YOLO
import numpy as np
from PIL import Image


MODEL_PATH = r"models/weld_detector/weld_detector_best.pt"

_model = YOLO(MODEL_PATH)


def run_post_weld_model(image, conf=0.20, iou=0.45):
    """
    Run the trained YOLO weld-defect detector on an image.

    Parameters
    ----------
    image : PIL.Image.Image or numpy.ndarray
        Input weld image.

    conf : float
        Minimum detection confidence.

    iou : float
        IoU threshold for non-maximum suppression.

    Returns
    -------
    dict
        Dictionary containing detected defects and annotated image.
    """

    if isinstance(image, Image.Image):
        image = image.convert("RGB")
        source = np.array(image)
    else:
        source = image

    result = _model.predict(
        source=source,
        conf=conf,
        iou=iou,
        verbose=False,
    )[0]

    defects = []

    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        bbox = [
            float(value)
            for value in box.xyxy[0].tolist()
        ]

        defects.append(
            {
                "name": result.names[class_id],
                "class_id": class_id,
                "confidence": confidence,
                "bbox": bbox,
            }
        )

    # Create annotated image with YOLO detections
    overlay = result.plot()

    # YOLO returns BGR; Streamlit expects RGB
    overlay = overlay[:, :, ::-1]

    return {
        "defects": defects,
        "overlay": overlay,
    }