# SFace attribution

FrameGuard uses `face_recognition_sface_2021dec.onnx` from the OpenCV model
collection for reference-face verification.

Model/project attribution:

- SFace: Sigmoid-Constrained Hypersphere Loss for Robust Face Recognition
- Yaoyao Zhong, Weihong Deng, Jiani Hu, Dongyue Zhao, Xian Li, and Dongchao Wen
- OpenCV face recognition model distribution

FrameGuard uses SFace only to compare a user-supplied reference image with face
candidates detected in the current video. It does not assign names, search a
face database, or persist biometric embeddings.
