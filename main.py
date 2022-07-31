from funcs import load_model

model = load_model(weights='weights/crowdhuman_yolov5m.pt', map_location='cpu')
model.eval()

print(model)