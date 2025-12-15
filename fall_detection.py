import imageio
import mediapipe as mp
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
import time
import math
import platform

SLOW_FACTOR = 2

def load_font(size):
    '''Cross-platform font loader'''
    try:
        if platform.system() == 'Darwin':
            return ImageFont.truetype("/System/Library/Fonts/PingFang.ttc",size)
        return ImageFont.truetype("arial.ttc", size)
    except:
        return ImageFont.load_default()

# Initialize MediaPipe pose detection
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Fall detection parameters
FALL_ANGLE_THRESHOLD = 45
FALL_DURATION_THRESHOLD = 1.0
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

# Training suggestions for four fall types
TRAINING_SUGGESTIONS = {
    "forward_fall": [
        "1. Reminder training: Stand with feet shoulder-width apart",
        "2. Slowly rise onto tiptoes to the highest point",
        "3. Pause for 2 seconds then slowly lower",
        "4. Forward step: Take a step forward",
        "5. Shift weight to front leg and slowly squat"
    ],
    "left_fall": [
        "1. Crossover step training",
        "2. Side lunge: Take a step to the side",
        "3. Shift weight to the side leg",
        "4. Slowly squat down",
        "5. Alternate between left and right legs"
    ],
    "right_fall": [
        "1. Crossover step training",
        "2. Side lunge: Take a step to the side",
        "3. Shift weight to the side leg",
        "4. Slowly squat down",
        "5. Alternate between left and right legs"
    ],
    "backward_fall": [
        "1. Single leg back lift: Stand holding a chair",
        "2. Slowly lift one leg backward",
        "3. Core anti-rotation training",
        "4. Sitting or standing, hold a light object",
        "5. Slowly rotate torso to one side"
    ],
    "sitting_fall": [
        "1. Seated rotation training",
        "2. Sit on a chair, hold a light object",
        "3. Slowly rotate to both sides",
        "4. Wall squat training",
        "5. Use leg strength to slowly stand up"
    ]
}

def calculate_knee_angle(landmarks, leg_side):
    """Calculate knee angle (sagittal plane)"""
    if leg_side == "left":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    else:
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
        return None

    hip_to_knee = [knee.x - hip.x, knee.y - hip.y]
    knee_to_ankle = [ankle.x - knee.x, ankle.y - knee.y]
    
    dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[1] * knee_to_ankle[1]
    hip_to_knee_length = math.sqrt(hip_to_knee[0]**2 + hip_to_knee[1]** 2)
    knee_to_ankle_length = math.sqrt(knee_to_ankle[0]**2 + knee_to_ankle[1]** 2)
    
    if hip_to_knee_length * knee_to_ankle_length == 0:
        return None
        
    angle = math.degrees(math.acos(dot_product / (hip_to_knee_length * knee_to_ankle_length)))
    return angle

def calculate_coronal_plane_angle(landmarks, leg_side):
    """Calculate knee coronal plane angle"""
    if leg_side == "left":
        hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
    else:
        hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
        knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
        ankle = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE]

    if hip.visibility < 0.5 or knee.visibility < 0.5 or ankle.visibility < 0.5:
        return None

    # Coronal plane angle calculation (x-z plane)
    hip_to_knee = [knee.x - hip.x, 0, knee.z - hip.z]
    knee_to_ankle = [ankle.x - knee.x, 0, ankle.z - knee.z]
    
    cross_product = hip_to_knee[1] * knee_to_ankle[2] - hip_to_knee[2] * knee_to_ankle[1]
    dot_product = hip_to_knee[0] * knee_to_ankle[0] + hip_to_knee[2] * knee_to_ankle[2]
    
    hip_to_knee_length = math.sqrt(hip_to_knee[0]**2 + hip_to_knee[2]** 2)
    knee_to_ankle_length = math.sqrt(knee_to_ankle[0]**2 + knee_to_ankle[2]** 2)
    
    if hip_to_knee_length * knee_to_ankle_length == 0:
        return None
        
    angle = math.degrees(math.atan2(cross_product, dot_product))
    return abs(angle)

def classify_fall_direction(landmarks, knee_angles):
    """Classify fall direction"""
    if landmarks[mp_pose.PoseLandmark.LEFT_HIP].visibility < 0.5 or landmarks[mp_pose.PoseLandmark.RIGHT_HIP].visibility < 0.5:
        return "unknown_direction"
    
    # Get key point positions
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP]
    left_knee = landmarks[mp_pose.PoseLandmark.LEFT_KNEE]
    right_knee = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE]
    
    # Calculate hip center point
    hip_center_x = (left_hip.x + right_hip.x) / 2
    hip_center_y = (left_hip.y + right_hip.y) / 2
    
    # Calculate knee center point
    knee_center_x = (left_knee.x + right_knee.x) / 2
    knee_center_y = (left_knee.y + right_knee.y) / 2
    
    # Calculate torso vector
    torso_vector = [knee_center_x - hip_center_x, knee_center_y - hip_center_y]
    
    # Calculate torso angle (relative to vertical)
    torso_angle = math.degrees(math.atan2(torso_vector[0], torso_vector[1]))
    
    # Calculate coronal plane angles
    left_coronal_angle = calculate_coronal_plane_angle(landmarks, "left")
    right_coronal_angle = calculate_coronal_plane_angle(landmarks, "right")
    
    # Determine fall direction
    if torso_angle > 45:  # Forward fall
        return "forward_fall"
    elif torso_angle < -45:  # Backward fall
        return "backward_fall"
    elif left_coronal_angle is not None and left_coronal_angle > 30:  # Left fall
        return "left_fall"
    elif right_coronal_angle is not None and right_coronal_angle > 30:  # Right fall
        return "right_fall"
    elif knee_angles['left'] is not None and knee_angles['right'] is not None and \
         knee_angles['left'] < 90 and knee_angles['right'] < 90:  # Sitting fall
        return "sitting_fall"
    else:
        return "unknown_direction"

def detect_fall(knee_angles, current_time, landmarks):
    """Fall detection function"""
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
                direction = classify_fall_direction(landmarks, knee_angles)
                return True, min(1.0, fall_duration / FALL_DURATION_THRESHOLD), direction
    else:
        fall_detected = False
    
    return False, 0.0, "no_fall"

def draw_realtime_angles(image, knee_angles, coronal_angles):
    """Show angles at top left"""
    draw = ImageDraw.Draw(image)
    font = load_font(32)

    # Prevent None
    lk = knee_angles.get("left") or 0
    rk = knee_angles.get("right") or 0
    lc = coronal_angles.get("left") or 0
    rc = coronal_angles.get("right") or 0

    text = (
        f"L-Knee: {lk:.1f}°   R-Knee: {rk:.1f}°\n"
        f"L-Coronal: {lc:.1f}°   R-Coronal: {rc:.1f}°"
    )

    draw.rectangle([(0, 0), (650, 90)], fill=(0, 0, 0, 180))
    draw.text((10, 15), text, fill=(255, 255, 255), font=font)

def draw_fall_warning(image, confidence, direction):
    """Draw fall warning"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    warning_height = int(height * 0.15 * confidence)
    draw.rectangle([(0, 0), (width, warning_height)], fill=(255, 0, 0))

    font = load_font(int(min(width, height) * 0.08 * confidence))

    text = f"Fall Warning! ({direction})"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    draw.text(
        (width * 0.25, warning_height * 0.25),
        text,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0)
    )

def draw_training_suggestions(image, direction):
    """Draw training suggestions"""
    if direction not in TRAINING_SUGGESTIONS:
        return
        
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Draw background (increased height to accommodate larger font)
    suggestion_height = 250  # Increased from 200 to 250
    draw.rectangle([(0, height-suggestion_height), (width, height)], fill=(0, 0, 0, 128))

    title_font = load_font(38)
    item_font = load_font(28)

    # Draw title
    title = f"{direction.replace('_', ' ').title()} Training Suggestions:"
    draw.text((10, height-suggestion_height+10), title, fill=(255, 255, 255), font=title_font)
    
    # Draw suggestions (adjusted spacing for larger font)
    for i, suggestion in enumerate(TRAINING_SUGGESTIONS[direction]):
        draw.text((30, height-suggestion_height+60+i*38), suggestion, fill=(255, 255, 255), font=item_font)  # Changed from 50+i*30 to 60+i*40

def draw_angle_diagrams(image, knee_angles, coronal_angles):
    """Draw sagittal and coronal plane angle diagrams"""
    draw = ImageDraw.Draw(image)
    width, height = image.size
    
    # Draw sagittal plane diagram
    sagittal_x = width - 200
    sagittal_y = height - 200
    draw.ellipse([(sagittal_x-50, sagittal_y-50), (sagittal_x+50, sagittal_y+50)], outline=(255,255,255))
    
    # Draw leg lines (sagittal plane)
    left_knee_angle = knee_angles['left'] if knee_angles['left'] is not None else 180
    right_knee_angle = knee_angles['right'] if knee_angles['right'] is not None else 180
    
    # Left leg sagittal
    left_leg_x = sagittal_x - 50 * math.sin(math.radians(left_knee_angle/2))
    left_leg_y = sagittal_y + 50 * math.cos(math.radians(left_knee_angle/2))
    draw.line([(sagittal_x, sagittal_y), (left_leg_x, left_leg_y)], fill=(255,0,0), width=3)
    
    # Right leg sagittal
    right_leg_x = sagittal_x + 50 * math.sin(math.radians(right_knee_angle/2))
    right_leg_y = sagittal_y + 50 * math.cos(math.radians(right_knee_angle/2))
    draw.line([(sagittal_x, sagittal_y), (right_leg_x, right_leg_y)], fill=(0,255,0), width=3)
    
    # Draw coronal plane diagram
    coronal_x = width - 200
    coronal_y = height - 400
    draw.ellipse([(coronal_x-50, coronal_y-50), (coronal_x+50, coronal_y+50)], outline=(255,255,255))
    
    # Draw leg lines (coronal plane)
    left_coronal_angle = coronal_angles['left'] if coronal_angles['left'] is not None else 0
    right_coronal_angle = coronal_angles['right'] if coronal_angles['right'] is not None else 0
    
    # Left leg coronal
    left_leg_x = coronal_x - 50 * math.sin(math.radians(left_coronal_angle))
    left_leg_y = coronal_y + 50 * math.cos(math.radians(left_coronal_angle))
    draw.line([(coronal_x, coronal_y), (left_leg_x, left_leg_y)], fill=(255,0,0), width=3)
    
    # Right leg coronal
    right_leg_x = coronal_x + 50 * math.sin(math.radians(right_coronal_angle))
    right_leg_y = coronal_y + 50 * math.cos(math.radians(right_coronal_angle))
    draw.line([(coronal_x, coronal_y), (right_leg_x, right_leg_y)], fill=(0,255,0), width=3)
    
    # Add labels
    font = load_font(28)
    
    draw.text((width-250, height-250), "Sagittal Angles", fill=(255,255,255), font=font)
    draw.text((width-250, height-450), "Coronal Angles", fill=(255,255,255), font=font)

def draw_landmarks(image, landmarks):
    """Custom draw landmarks and connections"""
    draw = ImageDraw.Draw(image)
    image_width, image_height = image.size
    
    for landmark in KEY_LANDMARKS:
        if landmarks[landmark].visibility > 0.5:
            x = landmarks[landmark].x * image_width
            y = landmarks[landmark].y * image_height
            draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=(255, 0, 0))
    
    connections = mp_pose.POSE_CONNECTIONS
    for connection in connections:
        start_idx = connection[0]
        end_idx = connection[1]
        
        if (landmarks[start_idx].visibility > 0.5 and 
            landmarks[end_idx].visibility > 0.5):
            start_point = (landmarks[start_idx].x * image_width, 
                          landmarks[start_idx].y * image_height)
            end_point = (landmarks[end_idx].x * image_width,
                         landmarks[end_idx].y * image_height)
            draw.line([start_point, end_point], fill=(0, 0, 255), width=2)

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Cannot open the video：", video_path)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    print("The video has loaded! FPS =", fps)

    global last_fall_time, fall_detected, fall_start_time
    last_fall_time = 0
    fall_detected = False

    with mp_pose.Pose(min_detection_confidence=0.5,
                      min_tracking_confidence=0.5) as pose:

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Process Done.")
                break

            time.sleep(1/ fps * SLOW_FACTOR)

            # ---- Step 1: Mediapipe Pose Inference ----
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)

            if not results.pose_landmarks:
                cv2.imshow("Fall Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            landmarks = results.pose_landmarks.landmark

            # ---- Step 2: Calculate knee angle ----
            knee_angles = {
                "left": calculate_knee_angle(landmarks, "left"),
                "right": calculate_knee_angle(landmarks, "right"),
            }

            # ---- Step 3: Calculate coronal plane angle ----
            coronal_angles = {
                "left": calculate_coronal_plane_angle(landmarks, "left"),
                "right": calculate_coronal_plane_angle(landmarks, "right"),
            }

            # ---- Step 4: Fall detection ----
            current_time = time.time()
            fall, confidence, direction = detect_fall(knee_angles, current_time, landmarks)

            # ---- Step 5: Using PiL drawing everything ----
            pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            draw_landmarks(pil_img, landmarks)
            draw_angle_diagrams(pil_img, knee_angles, coronal_angles)

            draw_realtime_angles(pil_img, knee_angles, coronal_angles)

            if fall:
                draw_fall_warning(pil_img, confidence, direction)
                draw_training_suggestions(pil_img, direction)

            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            # ---- Step 6: Show plots ----
            cv2.imshow("Fall Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Video Done！")

if __name__ == "__main__":
    video_path = "/Users/liyining/CVproj/Test1.MOV"  # Replace with your video path
    process_video(video_path)
