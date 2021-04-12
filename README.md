# try-onnxruntime

## setup

1. install libomp (when you use MacOS.)
```bash
$ brew install libomp
```

2. install project dependencies in project root directory.
```bash
$ poetry install
```

3. download YOLO v3 ONNX model.
```bash
$ wget -O model/yolov3-10.onnx https://github.com/onnx/models/raw/master/vision/object_detection_segmentation/yolov3/model/yolov3-10.onnx 
```

4. download images to images folder.
