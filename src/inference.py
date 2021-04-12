import os
import time
from pathlib import Path
import onnxruntime as rt
import numpy as np
from PIL import Image


MODEL_PATH = Path("./model/yolov3-10.onnx")
IMAGES_PATH = Path("./images/")


def letterbox_image(image, size):
    """resize image with unchanged aspect ratio using padding"""
    iw, ih = image.size
    w, h = size
    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new("RGB", size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
    return new_image


def preprocess(img):
    model_image_size = (416, 416)
    boxed_image = letterbox_image(img, tuple(reversed(model_image_size)))
    image_data = np.array(boxed_image, dtype="float32")
    image_data /= 255.0
    image_data = np.transpose(image_data, [2, 0, 1])
    image_data = np.expand_dims(image_data, 0)
    return image_data

def inference(session, inputs):
    start = time.time()
    results = [session.run([], {"input_1": image, "image_shape": shape}) for image, shape in inputs]
    end = time.time()
    inference_time = np.round((end - start), 2)
    print(f"inference time: {inference_time / len(inputs)} sec")
    # [print(f"boxes: {boxes},\nscores: {scores},\nclass_indices: {class_indices}") for boxes, scores, class_indices in results]

def main():
    session_options = rt.SessionOptions()
    session_options.enable_profiling = True
    # session_options.intra_op_num_threads = 6
    # session_options.execution_mode = rt.ExecutionMode.ORT_PARALLEL
    # session_options.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL
    # session_options.inter_op_num_threads = 6
    # session_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = rt.InferenceSession(str(MODEL_PATH), session_options)

    images = []
    for path in os.listdir(IMAGES_PATH):
        image = Image.open(f"./images/{path}")
        shape = np.array([image.size[1], image.size[0]], dtype=np.float32).reshape(1, 2)
        images.append((preprocess(image), shape))

    for _ in range(10):
        inference(session, images)

if __name__ == "__main__":
    main()