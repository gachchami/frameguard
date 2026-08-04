# Image attribution

The test dataset uses two public sample images:

- `subject_a.png`: cropped from `skimage.data.astronaut`, a NASA image commonly
  distributed with scikit-image.
- `subject_b.png`: cropped from Matplotlib's `grace_hopper.jpg` sample image, a
  United States Navy portrait of Grace Hopper.

The generated videos are synthetic transformations created for FrameGuard
software testing. The reference labels `subject_a` and `subject_b` are used so
the application does not need to identify or name the people in the images.
