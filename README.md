
# Fall Detection
## Multi-Scenario Human Fall Detection Based on MediaPipe Pose

This repository implements a **vision-based human fall detection system** using **MediaPipe Pose**. The system detects falls in real-time from RGB videos and classifies fall directions into **forward**, **backward**, **left**, **right**, and **sitting fall**. It is designed to be lightweight, interpretable, and applicable across multiple indoor scenes without wearable sensors.

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

Create a virtual environment using the new Python version：
```bash
python -m venv cv_env
```

Activate the virtual environment：
```bash
.\cv_env\Scripts\activate
```


### Installation

Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes the necessary libraries to run the project, including `mediapipe`, `opencv-python`, `numpy`, `imageio`, and `pillow`.


or can install them dependently:

```bash
pip install mediapipe==0.10.21 numpy==1.26.4 opencv-python==4.9.0.80 imageio pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 2. Pretrained Models

This project uses **pretrained pose estimation models provided by MediaPipe Pose**. No additional model training is required.

The models are automatically downloaded when **MediaPipe** is installed and initialized.

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

To run the fall detection system on a video file, follow these steps:

1. Clone this repository to your local machine.
2. Navigate to the project folder.
3. Run the system using the following command:

```bash
python fall_detection.py
```
When choosing input option, if there is the warning InputOptionWarning.jpg , just ignore it and input 1/2.

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

- **Knee joint angles** in the sagittal plane.
- **Torso inclination angle** for forward/backward falls.
- **Coronal plane knee angles** for left/right falls.
- **Temporal consistency constraint** to suppress false positives.

Supported fall directions:

- **Forward Fall**
- **Backward Fall**
- **Left Fall**
- **Right Fall**
- **Sitting Fall**

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

---

## 9. Installation and Setup Guide (Additional Setup Instructions)

### Project Folder Structure

```text
├── fall_detection.py        # Main fall detection script
├── requirements.txt        # Python dependencies
├── videos/                 # (Optional) Test videos
├── README.md               # Project documentation
```

### 1. Clean Up Old Virtual Environment

To avoid potential interference, we need to clear any previous corrupted virtual environments.

Run the following command in the terminal to completely delete the old `cv_env` environment (this will not cause an error if the environment hasn’t been created previously):

```bash
rm -rf /Users/liyining/ComputerVisionProj/cv_env
```

### 2. Create and Activate a New Virtual Environment

#### 2.1 Create the Virtual Environment

Use the following command to create a new virtual environment named `cv_env`:

```bash
python3 -m venv /Users/liyining/ComputerVisionProj/cv_env
```

After successful execution, a `cv_env` folder will appear in the `/Users/liyining/ComputerVisionProj` directory, indicating that the virtual environment has been created.

#### 2.2 Activate the Virtual Environment

Activate the virtual environment first before installing dependencies:

```bash
source /Users/liyining/ComputerVisionProj/cv_env/bin/activate
```

After activation, the terminal prompt should show `(cv_env)`, indicating the virtual environment is active:

```plaintext
(cv_env) liyining@MacBook-Pro ComputerVisionProj %
```

### 3. Install Dependencies

Once the virtual environment is activated, run the following commands to install the required dependencies, using a domestic mirror to speed up the download process and avoid version conflicts.

#### 3.1 Upgrade `pip`

Upgrade `pip` to the latest version to ensure that dependency resolution works correctly:

```bash
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3.2 Install Required Dependencies

Run the following command to install the required dependencies with specific versions:

```bash
pip install mediapipe==0.10.21 numpy==1.26.4 opencv-python==4.9.0.80 imageio pillow -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3.3 Verify Installed Dependencies

Once installation is complete, run the following command to verify that the dependencies are installed successfully and check their versions:

```bash
pip list
```

You should see the following packages with the corresponding versions:
- `mediapipe==0.10.21`
- `numpy==1.26.4`
- `opencv-python==4.9.0.80`
- `imageio` (version is not restricted)
- `pillow` (version is not restricted)

---

### Summary
This setup guide provides detailed steps for setting up your environment and running the fall detection system. After completing the setup, your virtual environment will be ready for use, and the necessary dependencies will be installed. You can now run the fall detection system and process videos for real-time fall detection.

If you encounter any issues or need further assistance, feel free to reach out!

