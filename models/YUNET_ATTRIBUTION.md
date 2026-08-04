# YuNet attribution

FrameGuard uses `face_detection_yunet_2023mar.onnx` from the OpenCV Model Zoo.
The OpenCV Zoo states that the files in the YuNet directory are licensed under
the MIT License.

Model/project attribution:

- YuNet: A Tiny Millisecond-level Face Detector
- Wei Wu, Hanyang Peng, and Shiqi Yu
- OpenCV Model Zoo / Shenzhen Institute of Artificial Intelligence and Robotics for Society

The model is used only to detect face bounding boxes. FrameGuard does not use a
face-recognition model and does not generate biometric identity embeddings.
