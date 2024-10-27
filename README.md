
# Object Detection Projects

This repository contains multiple Python scripts for object detection using different algorithms and frameworks. Each project leverages a unique approach to identify objects in images or video streams, from classical computer vision techniques to deep learning models.

## Table of Contents
1. [RetinaNet Object Detection](#retinanet-object-detection)
2. [ORB Object Detection](#orb-object-detection)
3. [SIFT Object Detection](#sift-object-detection)
4. [Real-Time Object Detection with MobileNet SSD](#real-time-object-detection-with-mobilenet-ssd)

---

## RetinaNet Object Detection

This project uses the **ImageAI** library with a pre-trained **RetinaNet** model to perform object detection on images.

### Overview

The script detects objects in an image using a **RetinaNet** model with a **ResNet50** backbone trained on the **COCO dataset**. Detected objects are highlighted in the output image, and a list of detections with confidence scores is displayed in the console.

### Requirements

- Python 3.x
- [ImageAI](https://github.com/OlafenwaMoses/ImageAI)
- TensorFlow
- OpenCV
- Pre-trained model file: `resnet50_coco_best_v2.0.1.h5`

### Installation

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install imageai tensorflow opencv-python
   ```

3. Download the pre-trained model file [`resnet50_coco_best_v2.0.1.h5`](https://github.com/OlafenwaMoses/ImageAI/releases/download/1.0/resnet50_coco_best_v2.0.1.h5) and place it in the same directory as the script.

### Usage

1. Place the input image in the project directory and name it `image.jpg`.
2. Run the detection script:

   ```bash
   python RetinaNetDetection.py
   ```

3. The script will generate an output image (`imagenew.jpg`) with detected objects highlighted.

### Output

- An output image with detected objects.
- Console output of detected objects with confidence scores.

---

## ORB Object Detection

This project utilizes the **ORB (Oriented FAST and Rotated BRIEF)** algorithm from the **OpenCV** library for real-time object detection using a webcam.

### Overview

The script captures real-time video from a webcam and compares it to a template image using ORB. Matches between keypoints are used to detect the object in the video feed.

### Requirements

- Python 3.x
- [OpenCV](https://opencv.org/)

### Installation

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install opencv-python numpy
   ```

3. Place the template image (e.g., `iphone7.jpg`) in the project directory.

### Usage

1. Connect your webcam.
2. Run the detection script:

   ```bash
   python orb_object_detection.py
   ```

3. The script will display a real-time video feed with a region of interest (ROI) to detect objects.

### Output

- Real-time video feed with a highlighted detection box.
- A console message indicating the number of ORB matches.

---

## SIFT Object Detection

This project uses the **SIFT (Scale-Invariant Feature Transform)** algorithm from the **OpenCV** library for real-time object detection.

### Overview

The script uses a webcam feed to match keypoints between a template image and the current frame using the **SIFT** algorithm. The **FlannBasedMatcher** is used for efficient keypoint matching.

### Requirements

- Python 3.x
- [OpenCV](https://opencv.org/) with `xfeatures2d` module enabled

### Installation

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install opencv-python-headless opencv-contrib-python numpy
   ```

3. Place the template image (e.g., `iphone7.jpg`) in the project directory.

### Usage

1. Connect your webcam.
2. Run the detection script:

   ```bash
   python sift_object_detection.py
   ```

3. The script will display a real-time video feed with detected objects highlighted.

### Output

- Real-time video feed with detection results.
- A green box indicates successful detection.

---

## Real-Time Object Detection with MobileNet SSD

This project utilizes **MobileNet SSD (Single Shot Detector)** for real-time object detection using a webcam feed.

### Overview

The script leverages a pre-trained **MobileNet SSD** model with OpenCV's `dnn` module to detect and classify objects in real-time from a video stream. Detected objects are highlighted with bounding boxes and labels.

### Requirements

- Python 3.x
- [OpenCV](https://opencv.org/)
- [imutils](https://github.com/jrosebr1/imutils)
- A pre-trained Caffe model (`MobileNetSSD_deploy.caffemodel`) and prototxt file (`MobileNetSSD_deploy.prototxt.txt`).

### Installation

1. Clone this repository.
2. Install the required packages:

   ```bash
   pip install opencv-python numpy imutils
   ```

3. Download the pre-trained Caffe model and prototxt file and place them in the project directory.

### Usage

1. Connect your webcam.
2. Run the detection script with the required arguments:

   ```bash
   python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
   ```

3. The script will display a real-time video feed with detected objects and FPS information.

### Output

- Real-time video feed with bounding boxes and labels.
- Console output showing FPS metrics.

---

## License

All projects in this repository are licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request.
