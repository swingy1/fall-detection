# fall-detection
# Multi-Scenario Human Fall Detection Based on MediaPipe Pose

This repository implements a **vision-based human fall detection system** using **MediaPipe Pose**. The system detects falls in real time from RGB videos and classifies fall directions into **forward, backward, left, right**, and **sitting fall**. It is designed to be lightweight, interpretable, and applicable across multiple indoor scenes without wearable sensors.

This project is developed as a **Final Project (Topic 4: Self-Defined Project)** for a computer vision course.

---

## 1. Requirements

### Software Environment

- Python 3.8 or later
- OpenCV
- MediaPipe
- NumPy
- Pillow
- imageio

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## 2. Pretrained Models

This project uses **pretrained pose estimation models provided by MediaPipe Pose**. No additional model training is required.

The models are automatically downloaded when MediaPipe is installed and initialized.

---

## 3. Preparation for Testing

### File Structure

```text
├── fall_detection.py        # Main fall detection script
├── requirements.txt        # Python dependencies
├── videos/                 # (Optional) Test videos
├── README.md               # Project documentation
```

### Running the System

To run the fall detection system on a video file:

```bash
python fall_detection.py
```

Before running, make sure to update the video path in `fall_detection.py`:

```python
video_path = "path_to_your_video_file"
```

The system processes the video frame by frame and displays real-time fall detection results, including:

- Human pose landmarks
- Knee and coronal plane angles
- Fall warning overlays
- Fall direction classification

Press **`q`** to exit during execution.

---

## 4. Fall Detection Method

The system detects falls based on:

- **Knee joint angles** in the sagittal plane
- **Torso inclination angle** for forward/backward falls
- **Coronal plane knee angles** for left/right falls
- **Temporal consistency constraint** to suppress false positives

Fall directions supported:

- Forward Fall
- Backward Fall
- Left Fall
- Right Fall
- Sitting Fall

---

## 5. Visualization

The system provides real-time visualization, including:

- Pose skeleton overlay
- Joint angle diagrams (sagittal and coronal planes)
- Fall warning banners
- Rehabilitation training suggestions based on fall type

---

## 6. Notes

- This project is designed for **single-camera RGB video input**.
- Severe occlusion or extreme camera angles may affect performance.
- All parameters are fixed and empirically selected.

---

## 7. Project Report

The final project report is written in English using the provided **LaTeX template (`Final_report_template.tex`)** and includes:

- System design
- Experimental setup
- Qualitative multi-scene results
- Discussion and future work

---

## 8. License

This project is for **academic use only**.
