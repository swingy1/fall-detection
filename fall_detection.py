import imageio
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time
import math
import platform
import csv
import textwrap
from datetime import datetime
from collections import defaultdict
from mediapipe.framework.formats import landmark_pb2
import os

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Core configuration parameters
SLOW_MOTION_FACTOR = 2  # Video slow-motion
FALL_ANGLE_THRESHOLD = 45
FALL_DURATION_THRESHOLD = 1.0
BALANCE_WEIGHT_STABILITY = 0.3
BALANCE_WEIGHT_TILT = 0.3
BALANCE_WEIGHT_SYMMETRY = 0.4
BALANCE_HISTORY_LENGTH = 30  # Number of historical frames in the balance score trend chart (About 1s)
CAMERA_RESOLUTION = (1280, 720)
CSV_FILE_PATH = 'data/fall_detection.csv'
FALL_VIDEO_DIR = 'fall_videos/'
FALL_VIDEO_FRAME_BUFFER = 90
FALL_SPEED_LIGHT = 0.1
FALL_SPEED_MODERATE = 0.3
FALL_CONTACT_AREA_THRESH = 3
HEAD_HIP_Y_DIFF_THRESH = 0.1
PERSON_TRACKING_DIST_THRESH = 0.1

# Fall detection parameters
last_fall_time = 0
fall_detected = False
fall_start_time = 0

# Define key joints
KEY_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_SHOULDER,
    mp_pose.PoseLandmark.RIGHT_SHOULDER,
    mp_pose.PoseLandmark.LEFT_HIP,
    mp_pose.PoseLandmark.RIGHT_HIP,
    mp_pose.PoseLandmark.LEFT_KNEE,
    mp_pose.PoseLandmark.RIGHT_KNEE,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE
]

# Training suggestions for Nine fall types
TRAINING_SUGGESTIONS = {
    "Forward": [
        "1. Reminder training: Stand with feet shoulder-width apart",
        "2. Slowly rise onto tiptoes to the highest point",
        "3. Pause for 2 seconds then slowly lower",
        "4. Forward step: Take a step forward",
        "5. Shift weight to front leg and slowly squat"
    ],
    "Left": [
        "1. Crossover step training",
        "2. Side lunge: Take a step to the side",
        "3. Shift weight to the side leg",
        "4. Slowly squat down",
        "5. Alternate between left and right legs"
    ],
    "Right": [
        "1. Crossover step training",
        "2. Side lunge: Take a step to the side",
        "3. Shift weight to the side leg",
        "4. Slowly squat down",
        "5. Alternate between left and right legs"
    ],
    "Backward": [
        "1. Single leg back lift: Stand holding a chair",
        "2. Slowly lift one leg backward",
        "3. Core anti-rotation training",
        "4. Sitting or standing, hold a light object",
        "5. Slowly rotate torso to one side"
    ],
    "Sitting": [
        "1. Seated rotation training",
        "2. Sit on a chair, hold a light object",
        "3. Slowly rotate to both sides",
        "4. Wall squat training",
        "5. Use leg strength to slowly stand up"
    ],
    "Forward-Left": [
        "1. Diagonal step training",
        "2. Lunge forward-left and shift weight",
        "3. Single leg balance (left leg)",
        "4. Core stability training",
        "5. Slow turning while walking"
    ],
    "Forward-Right": [
        "1. Diagonal step training",
        "2. Lunge forward-right and shift weight",
        "3. Single leg balance (right leg)",
        "4. Core stability training",
        "5. Slow turning while walking"
    ],
    "Backward-Left": [
        "1. Backward diagonal step training",
        "2. Standing left-back lean (hold support)",
        "3. Left hip abductor training",
        "4. Core anti-rotation exercises",
        "5. Slow backward walking"
    ],
    "Backward-Right": [
        "1. Backward diagonal step training",
        "2. Standing right-back lean (hold support)",
        "3. Right hip abductor training",
        "4. Core anti-rotation exercises",
        "5. Slow backward walking"
    ]
}

"""Get compatible font for Windows/macOS with specified size"""
def get_available_font(font_size=20):
    try:
        return ImageFont.truetype("arial.ttf", font_size)
    except:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
        except:
            return ImageFont.load_default(size=font_size)


"""Calculate knee angle (sagittal plane)"""
def calculate_knee_angle(landmarks, leg_side):
    if leg_side == "left":
        hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    else:
        hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
        return None

    hip_to_knee = [knee.x - hip.x, knee.y - hip.y]
    knee_to_ankle = [ankle.x - knee.x, ankle.y - knee.y]

    dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[1] * knee_to_ankle[1]
    hip_to_knee_length = math.sqrt(hip_to_knee[0] ** 2 + hip_to_knee[1] ** 2)
    knee_to_ankle_length = math.sqrt(knee_to_ankle[0] ** 2 + knee_to_ankle[1] ** 2)

    if hip_to_knee_length * knee_to_ankle_length == 0:
        return None

    angle = math.degrees(math.acos(dot_product / (hip_to_knee_length * knee_to_ankle_length)))
    return angle


"""Calculate knee coronal plane angle"""
def calculate_coronal_plane_angle(landmarks, leg_side):
    if leg_side == "left":
        hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
        ankle = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    else:
        hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
        ankle = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
        return None

    # Coronal plane angle calculation (x-z plane)
    hip_to_knee = [knee.x - hip.x, 0, knee.z - hip.z]
    knee_to_ankle = [ankle.x - knee.x, 0, ankle.z - knee.z]

    cross_product = hip_to_knee[1] * knee_to_ankle[2] - hip_to_knee[2] * knee_to_ankle[1]
    dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[2] * knee_to_ankle[2]

    hip_to_knee_length = math.sqrt(hip_to_knee[0] ** 2 + hip_to_knee[2] ** 2)
    knee_to_ankle_length = math.sqrt(knee_to_ankle[0] ** 2 + knee_to_ankle[2] ** 2)

    if hip_to_knee_length * knee_to_ankle_length == 0:
        return None

    angle = math.degrees(math.atan2(cross_product, dot_product))
    return abs(angle)


"""Classify fall direction"""
def classify_fall_direction(landmarks):
    if (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].visibility < 0.5 or
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].visibility < 0.5 or
            landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value].visibility < 0.5 or
            landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value].visibility < 0.5):
        return "Unknown (low visibility)"

    # Extract joint coordinates
    left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
    left_knee = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE.value]
    right_knee = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    left_shoulder = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    right_shoulder = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

    # Calculate the human stability reference center
    shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
    shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2

    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2

    knee_center_y = (left_knee.y + right_knee.y) / 2

    # Sitting Judgement
    if hip_center_y > knee_center_y - 0.05 and abs(shoulder_center_y - hip_center_y) < 0.08:
        return "Sitting"

    # Falling Direction Vector
    dx = shoulder_center_x - hip_center_x
    dy = knee_center_y - shoulder_center_y
    abs_dx = abs(dx)
    abs_dy = abs(dy)

    # Shoulder Line Roll
    shoulder_dx = right_shoulder.x - left_shoulder.x
    shoulder_dy = right_shoulder.y - left_shoulder.y
    shoulder_slope = shoulder_dy / (abs(shoulder_dx) + 1e-6)  # Positive: right fall; Negative: left fall

    print(
        f"dx={dx:.3f}, dy={dy:.3f}, "
        f"shoulder_slope={shoulder_slope:.2f}"
    )

    # Threshold
    TH_MAIN = 0.18  # main direction (forward & backward) threshold
    TH_ROLL = 0.15  # shoulder line tilt
    TH_SIDE_WALK = 0.04  # Extremely weak left-right compensation

    # Define Nine Directions
    # 1. Main Direction
    if abs_dy > TH_MAIN:

        # Forward
        if dy > 0.26:
            if shoulder_slope > TH_ROLL:
                return "Forward-Right"
            elif shoulder_slope < -TH_ROLL:
                return "Forward-Left"
            else:
                return "Forward"

        # Backward
        else:
            if shoulder_slope > TH_ROLL:
                return "Backward-Right"
            elif shoulder_slope < -TH_ROLL:
                return "Backward-Left"
            else:
                return "Backward"

    # Pure sideways fall
    if abs_dx > TH_SIDE_WALK:
        return "Right" if dx > 0 else "Left"

    return "Standing"


"""Fall detection function"""
def detect_fall(knee_angles, current_time, landmarks):
    global last_fall_time, fall_detected, fall_start_time

    if knee_angles['left'] is None or knee_angles['right'] is None:
        return False, 0.0, "no_fall"

    average_angle = (knee_angles['left'] + knee_angles['right']) / 2

    if average_angle < FALL_ANGLE_THRESHOLD:
        if not fall_detected:
            fall_start_time = current_time
            fall_detected = True
        else:
            fall_duration = current_time - fall_start_time
            if fall_duration >= FALL_DURATION_THRESHOLD:
                last_fall_time = current_time
                direction = classify_fall_direction(landmarks)
                return True, min(1.0, fall_duration / FALL_DURATION_THRESHOLD), direction
    else:
        fall_detected = False

    return False, 0.0, "no_fall"


"""Track multiple persons and assign/maintain IDs"""
def track_persons(prev_persons, current_landmarks_list):
    current_persons = []
    used_prev_indices = set()

    for landmarks in current_landmarks_list:
        left_hip = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]
        current_hip_x = (left_hip.x + right_hip.x) / 2
        current_hip_y = (left_hip.y + right_hip.y) / 2

        best_match_id = None
        min_dist = float('inf')
        for prev_id, prev_data in prev_persons.items():
            if prev_id in used_prev_indices:
                continue
            dist = ((current_hip_x - prev_data['hip_x']) ** 2 + (current_hip_y - prev_data['hip_y']) ** 2) ** 0.5
            if dist < PERSON_TRACKING_DIST_THRESH and dist < min_dist:
                min_dist = dist
                best_match_id = prev_id

        if best_match_id is not None:
            used_prev_indices.add(best_match_id)
            current_persons.append({
                'id': best_match_id,
                'landmarks': landmarks,
                'hip_x': current_hip_x,
                'hip_y': current_hip_y
            })
        else:
            new_id = 1
            while new_id in prev_persons:
                new_id += 1
            current_persons.append({
                'id': new_id,
                'landmarks': landmarks,
                'hip_x': current_hip_x,
                'hip_y': current_hip_y
            })

    return {p['id']: p for p in current_persons}


"""Calculate balance score (0-100)"""
def calculate_balance_score(landmarks, image_width, image_height):

    # 1. Stability score (core landmarks visibility)
    core_landmarks = [
        mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE
    ]
    stable_count = sum(1 for lm in core_landmarks if landmarks.landmark[lm.value].visibility >= 0.7)
    stability_score = (stable_count / len(core_landmarks)) * 100 * BALANCE_WEIGHT_STABILITY

    # 2. Tilt score (torso angle)
    shoulder_center_x = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x + landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x) / 2
    shoulder_center_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y + landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y) / 2
    hip_center_x = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].x + landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_HIP.value].x) / 2
    hip_center_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    torso_vec_x = hip_center_x - shoulder_center_x
    torso_vec_y = hip_center_y - shoulder_center_y
    tilt_angle = abs(math.degrees(math.atan(torso_vec_x / torso_vec_y))) if torso_vec_y != 0 else 90
    tilt_score = 100 * BALANCE_WEIGHT_TILT if tilt_angle <= 15 else (
        0 if tilt_angle >= 45 else (1 - (tilt_angle - 15) / 30) * 100 * BALANCE_WEIGHT_TILT)

    # 3. Symmetry score (knee angle + hip height)
    left_knee_angle = calculate_knee_angle(landmarks, "left") or 180
    right_knee_angle = calculate_knee_angle(landmarks, "right") or 180
    knee_angle_diff = abs(left_knee_angle - right_knee_angle)
    left_hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y * image_height
    right_hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value].y * image_height
    hip_y_diff = abs(left_hip_y - right_hip_y) / image_height
    knee_symmetry = 100 if knee_angle_diff <= 10 else max(0, 100 - (knee_angle_diff - 10) * 2)
    hip_symmetry = 100 if hip_y_diff <= 0.05 else max(0, 100 - (hip_y_diff - 0.05) * 2000)
    symmetry_score = ((knee_symmetry + hip_symmetry) / 2) * BALANCE_WEIGHT_SYMMETRY

    return min(100, round(stability_score + tilt_score + symmetry_score))



"""Calculate fall speed (pixels/second)"""
def calculate_fall_speed(prev_hip_y, current_hip_y, frame_interval):

    if frame_interval <= 0:
        return 0
    return abs((current_hip_y - prev_hip_y) / frame_interval)


"""Classify fall severity: light/moderate/severe"""
def classify_fall_severity(fall_speed, landmarks, image_height):

    severity = "light" if fall_speed < FALL_SPEED_LIGHT else (
        "moderate" if fall_speed < FALL_SPEED_MODERATE else "severe")

    # Check contact area
    ground_threshold = image_height * 0.9
    contact_landmarks = [mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.RIGHT_KNEE,
                         mp_pose.PoseLandmark.LEFT_ANKLE, mp_pose.PoseLandmark.RIGHT_ANKLE,
                         mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP]
    contact_count = sum(
        1 for lm in contact_landmarks if landmarks.landmark[lm.value].y * image_height >= ground_threshold)
    if contact_count >= FALL_CONTACT_AREA_THRESH and severity == "light":
        severity = "moderate"

    # Check head position
    head_y = landmarks.landmark[mp_pose.PoseLandmark.NOSE.value].y
    hip_center_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value].y + landmarks.landmark[
        mp_pose.PoseLandmark.RIGHT_HIP.value].y) / 2
    if head_y > hip_center_y + HEAD_HIP_Y_DIFF_THRESH:
        severity = "severe"

    return severity

"""Draw balance score trend chart"""
def draw_balance_trend(draw, balance_history, area_rect):
    x1, y1, x2, y2 = area_rect
    width = x2 - x1
    height = y2 - y1

    if len(balance_history) < 2:
        return

    # Background
    draw.rounded_rectangle(
        [(x1 - 12, y1 - 28), (x2 + 12, y2 + 12)],
        radius=12,
        fill=(20, 20, 20, 160)
    )

    def map_score(score):
        return y2 - (score / 100) * height

    points = [
        (x1 + i / (len(balance_history) - 1) * width, map_score(score))
        for i, score in enumerate(balance_history)
    ]

    # Broken Line
    line_color = (0, 220, 180)
    draw.line(points, fill=line_color, width=4)

    # Key points to increase readability
    for x, y in points:
        draw.ellipse(
            [(x - 3, y - 3), (x + 3, y + 3)],
            fill=(255, 255, 255)
        )

    # Coordinate axis
    draw.line([(x1, y1), (x1, y2)], fill=(180, 180, 180), width=1)
    draw.line([(x1, y2), (x2, y2)], fill=(180, 180, 180), width=1)

    font = get_available_font(14)
    draw.text((x1 - 30, y2 - 8), "0", fill=(220, 220, 220), font=font)
    draw.text((x1 - 40, y1 - 8), "100", fill=(220, 220, 220), font=font)
    draw.text((x1 + 6, y1 - 22), "Balance Trend", fill=(220, 220, 220), font=font)

"""Initialize data export (CSV + folder)"""
def init_data_export():
    os.makedirs("data", exist_ok=True)
    os.makedirs(FALL_VIDEO_DIR, exist_ok=True)
    with open(CSV_FILE_PATH, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "person_id", "left_knee_angle", "right_knee_angle",
            "left_coronal_angle", "right_coronal_angle", "balance_score",
            "is_fall", "fall_direction", "fall_severity"
        ])

"""Write detection data to CSV"""
def write_detection_data(timestamp, person_id, knee_angles, coronal_angles, balance_score, is_fall, fall_direction,
                         fall_severity):
    with open(CSV_FILE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, person_id,
            knee_angles['left'] if knee_angles['left'] is not None else "N/A",
            knee_angles['right'] if knee_angles['right'] is not None else "N/A",
            coronal_angles['left'] if coronal_angles['left'] is not None else "N/A",
            coronal_angles['right'] if coronal_angles['right'] is not None else "N/A",
            balance_score,
            "Yes" if is_fall else "No",
            fall_direction if is_fall else "N/A",
            fall_severity if is_fall else "N/A"
        ])

"""Initialize fall video writer"""
def init_fall_video_writer(frame_width, frame_height, fps, fall_timestamp):
    video_path = f"{FALL_VIDEO_DIR}fall_{fall_timestamp}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    return cv2.VideoWriter(video_path, fourcc, fps, (frame_width, frame_height)), video_path



"""Draw fall warning with severity color"""
def draw_fall_warning(draw, confidence, direction, severity="moderate", person_id=""):
    width, height = draw.im.size

    box_width = int(width * 0.62)
    box_height = int(120 + 50 * confidence)
    box_x = (width - box_width) // 2
    box_y = int(height * 0.05)

    severity_colors = {
        "light": (255, 230, 120, 235),
        "moderate": (255, 170, 60, 240),
        "severe": (220, 60, 60, 245)
    }
    bg_color = severity_colors.get(severity, (255, 80, 80, 240))

    # Background
    draw.rounded_rectangle(
        [(box_x, box_y), (box_x + box_width, box_y + box_height)],
        radius=22,
        fill=bg_color
    )

    # Font
    font_title = get_available_font(48)
    font_body = get_available_font(22)

    title = "FALL DETECTED"
    body = (f"Person ID: {person_id}\n"
            f"Direction: {direction}\n"
            f"Severity: {severity.upper()}"
            )

    center_x = box_x + box_width // 2
    center_y = box_y + box_height // 2

    # stroke text
    def stroke_text(pos, text, font):
        draw.text(pos, text, fill=(255, 255, 255), align="center",
                 font=font, stroke_width=2, stroke_fill=(0, 0, 0))

    stroke_text((box_x + 24, box_y + 14), title, font_title)


    draw.multiline_text(
        (center_x, center_y),
        body,
        fill=(255, 255, 255),
        font=font_body,
        spacing=6,
        align="center",
        anchor="mm",
        stroke_width=2,
        stroke_fill=(0, 0, 0)
    )

"""Draw training suggestions"""
def draw_training_suggestions(image, direction):
    if direction not in TRAINING_SUGGESTIONS:
        return

    draw = ImageDraw.Draw(image)
    width, height = image.size
    suggestion_height = 250
    draw.rectangle([(0, height - suggestion_height), (width, height)], fill=(0, 0, 0, 128))

    title_font = get_available_font(40)
    suggestion_font = get_available_font(30)
    title = f"{direction.replace('_', ' ').title()} Training Suggestions:"
    draw.text((10, height - suggestion_height + 10), title, fill=(255, 255, 255), font=title_font)

    for i, suggestion in enumerate(TRAINING_SUGGESTIONS[direction]):
        draw.text((30, height - suggestion_height + 60 + i * 40), suggestion, fill=(255, 255, 255),
                  font=suggestion_font)

"""Draw sagittal and coronal plane angle diagrams"""
def draw_angle_diagrams(image, knee_angles, coronal_angles):
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Sagittal diagram
    sagittal_x = width - 200
    sagittal_y = height - 200
    draw.ellipse([(sagittal_x - 50, sagittal_y - 50), (sagittal_x + 50, sagittal_y + 50)], outline=(255, 255, 255))
    left_knee_angle = knee_angles['left'] or 180
    right_knee_angle = knee_angles['right'] or 180
    draw.line([(sagittal_x, sagittal_y), (sagittal_x - 50 * math.sin(math.radians(left_knee_angle / 2)),
                                          sagittal_y + 50 * math.cos(math.radians(left_knee_angle / 2)))],
              fill=(255, 0, 0), width=3)
    draw.line([(sagittal_x, sagittal_y), (sagittal_x + 50 * math.sin(math.radians(right_knee_angle / 2)),
                                          sagittal_y + 50 * math.cos(math.radians(right_knee_angle / 2)))],
              fill=(0, 255, 0), width=3)

    # Coronal diagram
    coronal_x = width - 200
    coronal_y = height - 400
    draw.ellipse([(coronal_x - 50, coronal_y - 50), (coronal_x + 50, coronal_y + 50)], outline=(255, 255, 255))
    left_coronal_angle = coronal_angles['left'] or 0
    right_coronal_angle = coronal_angles['right'] or 0
    draw.line([(coronal_x, coronal_y), (coronal_x - 50 * math.sin(math.radians(left_coronal_angle)),
                                        coronal_y + 50 * math.cos(math.radians(left_coronal_angle)))], fill=(255, 0, 0),
              width=3)
    draw.line([(coronal_x, coronal_y), (coronal_x + 50 * math.sin(math.radians(right_coronal_angle)),
                                        coronal_y + 50 * math.cos(math.radians(right_coronal_angle)))],
              fill=(0, 255, 0), width=3)

    font = get_available_font(20)
    draw.text((width - 250, height - 250), "Sagittal Angles", fill=(255, 255, 255), font=font)
    draw.text((width - 250, height - 450), "Coronal Angles", fill=(255, 255, 255), font=font)

"""Draw real-time angles with offset (for multiple persons)"""
def draw_real_time_angles(draw, knee_angles, coronal_angles, offset_y=0):
    font = get_available_font(24)
    y_offset = 10 + offset_y

    texts = [
        f"Left Knee: {knee_angles['left']:.1f}°" if knee_angles['left'] else "Left Knee: N/A",
        f"Right Knee: {knee_angles['right']:.1f}°" if knee_angles['right'] else "Right Knee: N/A",
        f"Left Coronal: {coronal_angles['left']:.1f}°" if coronal_angles['left'] else "Left Coronal: N/A",
        f"Right Coronal: {coronal_angles['right']:.1f}°" if coronal_angles['right'] else "Right Coronal: N/A"
    ]

    for text in texts:
        bbox = draw.textbbox((10, y_offset), text, font=font)
        draw.rectangle([(bbox[0] - 5, bbox[1] - 5), (bbox[2] + 5, bbox[3] + 5)], fill=(0, 0, 0, 180))
        draw.text((10, y_offset), text, fill=(255, 255, 255), font=font)
        y_offset += 30

"""Draw key landmarks and connections"""
def draw_landmarks(image, landmarks):
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size

    for landmark in KEY_LANDMARKS:
        if landmarks.landmark[landmark.value].visibility > 0.5:
            x = landmarks.landmark[landmark.value].x * image_width
            y = landmarks.landmark[landmark.value].y * image_height
            draw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=(255, 0, 0))

    for connection in mp_pose.POSE_CONNECTIONS:
        start_idx = connection[0]
        end_idx = connection[1]

        if (landmarks.landmark[start_idx].visibility > 0.5 and
                landmarks.landmark[end_idx].visibility > 0.5):
            start_point = (landmarks.landmark[start_idx].x * image_width,
                           landmarks.landmark[start_idx].y * image_height)
            end_point = (landmarks.landmark[end_idx].x * image_width,
                         landmarks.landmark[end_idx].y * image_height)
            draw.line([start_point, end_point], fill=(0, 0, 255), width=2)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot Open the Video：", video_path)
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    delay_ms = int(1000 / (original_fps / SLOW_MOTION_FACTOR))
    print(f"The Video has been Loaded！Original FPS = {original_fps:.1f}, FPS After Slow Motion"
          f" = {original_fps/SLOW_MOTION_FACTOR:.1f}")

    global last_fall_time, fall_detected, fall_start_time
    last_fall_time = 0
    fall_detected = False
    fall_start_time = 0
    prev_persons = {}
    person_balance_history = defaultdict(list)
    frame_buffer = []
    fall_video_writer = None
    fall_video_recording = False
    fall_video_frame_count = 0

    init_data_export()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Done. No more frames.")
                if fall_video_recording and fall_video_writer:
                    for buf_frame in frame_buffer:
                        fall_video_writer.write(buf_frame)
                    fall_video_writer.release()
                break

            frame_buffer.append(frame)
            if len(frame_buffer) > FALL_VIDEO_FRAME_BUFFER:
                frame_buffer.pop(0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            if not results.pose_landmarks:
                cv2.imshow("Fall Detection", frame)
                if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                    break
                continue

            current_landmarks_list = [results.pose_landmarks] if isinstance(results.pose_landmarks, landmark_pb2.NormalizedLandmarkList) else []
            current_persons = track_persons(prev_persons, current_landmarks_list)
            prev_persons = {p_id: {'hip_x': p['hip_x'], 'hip_y': p['hip_y']} for p_id, p in current_persons.items()}

            for person_id, person_data in current_persons.items():
                landmarks = person_data['landmarks']

                knee_angles = {
                    "left": calculate_knee_angle(landmarks, "left"),
                    "right": calculate_knee_angle(landmarks, "right"),
                }
                coronal_angles = {
                    "left": calculate_coronal_plane_angle(landmarks, "left"),
                    "right": calculate_coronal_plane_angle(landmarks, "right"),
                }
                balance_score = calculate_balance_score(landmarks, frame_width, frame_height)
                balance_history = person_balance_history[person_id]
                balance_history.append(balance_score)
                if len(balance_history) > BALANCE_HISTORY_LENGTH:
                    balance_history.pop(0)
                person_balance_history[person_id] = balance_history

                current_time = time.time()
                fall, confidence, direction = detect_fall(knee_angles, current_time, landmarks)
                fall_severity = "N/A"
                if fall:
                    prev_hip_y = prev_persons.get(person_id, {}).get('hip_y', person_data['hip_y'])
                    frame_interval = 1 / original_fps
                    fall_speed = calculate_fall_speed(prev_hip_y*frame_height, person_data['hip_y']*frame_height, frame_interval)
                    fall_severity = classify_fall_severity(fall_speed, landmarks, frame_height)
                    if not fall_video_recording:
                        print(f"Detected Falling（ID:{person_id}，Severity:{fall_severity}），start to saving...")
                        fall_video_writer, video_path = init_fall_video_writer(frame_width, frame_height, original_fps, current_timestamp)
                        for buf_frame in frame_buffer:
                            fall_video_writer.write(buf_frame)
                        fall_video_recording = True
                        fall_video_frame_count = 0

                # Draw person info
                head = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                draw.text((head.x*frame_width-20, head.y*frame_height-30), f"ID:{person_id}", fill=(0,255,255), font=get_available_font(30))
                draw_real_time_angles(draw, knee_angles, coronal_angles, offset_y=person_id*120)
                balance_offset_x = frame_width - 300
                balance_offset_y = (person_id-1)*200
                draw.text((balance_offset_x, 30+balance_offset_y), f"Balance: {balance_score}", fill=(0,255,0), font=get_available_font(24))
                draw_balance_trend(draw, balance_history, (balance_offset_x, 50+balance_offset_y, frame_width-50, 150+balance_offset_y))

                if fall:
                    draw_fall_warning(draw, confidence, direction, severity=fall_severity, person_id=person_id)
                    draw_training_suggestions(pil_img, direction)

                write_detection_data(current_timestamp, person_id, knee_angles, coronal_angles, balance_score, fall, direction, fall_severity)

            for person_data in current_persons.values():
                draw_landmarks(pil_img, person_data['landmarks'])

            if current_persons:
                first_person_landmarks = next(iter(current_persons.values()))['landmarks']
                draw_angle_diagrams(pil_img, knee_angles, coronal_angles)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if fall_video_recording:
                fall_video_writer.write(frame)
                fall_video_frame_count += 1
                if fall_video_frame_count >= original_fps * 2:
                    fall_video_writer.release()
                    print(f"Fall Video saved：{video_path}")
                    fall_video_recording = False

            cv2.imshow("Fall Detection", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                if fall_video_recording and fall_video_writer:
                    fall_video_writer.release()
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video processing Done! The data has been saved into：", CSV_FILE_PATH)

def process_camera():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION[1])
    if not cap.isOpened():
        print("Cannot open camera！")
        return

    original_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frame_width = CAMERA_RESOLUTION[0]
    frame_height = CAMERA_RESOLUTION[1]
    delay_ms = int(1000 / original_fps)
    print(f"Camera On！Resolution Ratio：{frame_width}×{frame_height}，FPS：{original_fps:.1f}")

    global last_fall_time, fall_detected, fall_start_time
    last_fall_time = 0
    fall_detected = False
    fall_start_time = 0
    prev_persons = {}
    person_balance_history = defaultdict(list)
    frame_buffer = []
    fall_video_writer = None
    fall_video_recording = False
    fall_video_frame_count = 0

    init_data_export()

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera Connection failed！")
                break

            frame = cv2.flip(frame, 1)
            frame_buffer.append(frame)
            if len(frame_buffer) > FALL_VIDEO_FRAME_BUFFER:
                frame_buffer.pop(0)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            current_timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            current_landmarks_list = []
            if results.pose_landmarks:
                current_landmarks_list = [results.pose_landmarks] if isinstance(results.pose_landmarks, landmark_pb2.NormalizedLandmarkList) else []
            current_persons = track_persons(prev_persons, current_landmarks_list)
            prev_persons = {p_id: {'hip_x': p['hip_x'], 'hip_y': p['hip_y']} for p_id, p in current_persons.items()}

            for person_id, person_data in current_persons.items():
                landmarks = person_data['landmarks']

                knee_angles = {
                    "left": calculate_knee_angle(landmarks, "left"),
                    "right": calculate_knee_angle(landmarks, "right"),
                }
                coronal_angles = {
                    "left": calculate_coronal_plane_angle(landmarks, "left"),
                    "right": calculate_coronal_plane_angle(landmarks, "right"),
                }
                balance_score = calculate_balance_score(landmarks, frame_width, frame_height)
                balance_history = person_balance_history[person_id]
                balance_history.append(balance_score)
                if len(balance_history) > BALANCE_HISTORY_LENGTH:
                    balance_history.pop(0)
                person_balance_history[person_id] = balance_history

                current_time = time.time()
                fall, confidence, direction = detect_fall(knee_angles, current_time, landmarks)
                fall_severity = "N/A"
                if fall:
                    prev_hip_y = prev_persons.get(person_id, {}).get('hip_y', person_data['hip_y'])
                    frame_interval = 1 / original_fps
                    fall_speed = calculate_fall_speed(prev_hip_y*frame_height, person_data['hip_y']*frame_height, frame_interval)
                    fall_severity = classify_fall_severity(fall_speed, landmarks, frame_height)
                    if not fall_video_recording:
                        print(f"Falling Detected（ID:{person_id}，Severity:{fall_severity}），start to saving...")
                        fall_video_writer, video_path = init_fall_video_writer(frame_width, frame_height, original_fps, current_timestamp)
                        for buf_frame in frame_buffer:
                            fall_video_writer.write(buf_frame)
                        fall_video_recording = True
                        fall_video_frame_count = 0

                head = landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                draw.text((head.x*frame_width-20, head.y*frame_height-30), f"ID:{person_id}", fill=(0,255,255), font=get_available_font(30))
                draw_real_time_angles(draw, knee_angles, coronal_angles, offset_y=person_id*120)
                balance_offset_x = frame_width - 300
                balance_offset_y = (person_id-1)*200
                draw.text((balance_offset_x, 30+balance_offset_y), f"Balance: {balance_score}", fill=(0,255,0), font=get_available_font(24))
                draw_balance_trend(draw, balance_history, (balance_offset_x, 50+balance_offset_y, frame_width-50, 150+balance_offset_y))

                if fall:
                    draw_fall_warning(draw, confidence, direction, severity=fall_severity, person_id=person_id)
                    draw_training_suggestions(pil_img, direction)

                write_detection_data(current_timestamp, person_id, knee_angles, coronal_angles, balance_score, fall, direction, fall_severity)

            for person_data in current_persons.values():
                draw_landmarks(pil_img, person_data['landmarks'])

            if current_persons:
                first_person_landmarks = next(iter(current_persons.values()))['landmarks']
                draw_angle_diagrams(pil_img, knee_angles, coronal_angles)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            if fall_video_recording:
                fall_video_writer.write(frame)
                fall_video_frame_count += 1
                if fall_video_frame_count >= original_fps * 2:
                    fall_video_writer.release()
                    print(f"Falling Video Saved：{video_path}")
                    fall_video_recording = False

            cv2.imshow("Real-time Fall Detection", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord('q'):
                if fall_video_recording and fall_video_writer:
                    fall_video_writer.release()
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Camera detection Done! The data has been saved into：", CSV_FILE_PATH)



if __name__ == "__main__":
    print("Please Choose Detecting Mode：")
    print("1. Local Video Detection")
    print("2. Real-Time Camera Detection")
    choice = input("Input Option (1/2)：")

    if choice == "1":
        video_path = input("Please Enter Video Path：")
        process_video(video_path)
    elif choice == "2":
        process_camera()
    else:
        print("Unavailable option！")
