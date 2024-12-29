
from ultralytics import YOLO


# Load a model
model = YOLO("yolo11n.yaml")

# Train the model
train_results = model.train(
    data="Fruits-detection/data.yaml",  # path to dataset YAML
    epochs=10,  # number of training epochs
    imgsz=640,  # training image size

)

# Evaluate model performance on the validation set
metrics = model.val()

# Export the model to ONNX format
path = model.export(format="onnx")  # return path to exported model