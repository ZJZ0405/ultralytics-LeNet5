from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("LeNet5-cls.yaml", task='classify')
    results = model.train(cfg="LeNet5-cfg.yaml", batch=128, cache="disk")