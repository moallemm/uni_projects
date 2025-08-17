import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore", message="h5py is running against HDF5")
import cv2
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict
from ultralytics import YOLO
import pyttsx3
import time

class VisionAssistant:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load models with consistent methods
        self.scene_classifier = tf.keras.models.load_model('best_model.keras')
        self.indoor_detector = YOLO('indoor_yolov8n.pt')
        self.outdoor_detector = YOLO('outdoor_yolov8n.pt')
        
        # MiDaS setup
        self.depth_estimator = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=False)
        state_dict = torch.load('midas_v21_small_256.pt', map_location=torch.device('cpu'))
        self.depth_estimator.load_state_dict(state_dict)
        self.depth_estimator.eval()
        
        # TTS setup
        self.tts_engine = pyttsx3.init()
        self.last_tts_time = 0
        self.tts_cooldown = 5  # seconds between TTS updates
        
        # Device configuration
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_estimator.to(self.device)
        
        # Depth calibration
        self.depth_scale = 5.0
        self.frame_count = 0

    def classify_scene(self, frame):
        """Classify scene using consistent preprocessing with training/testing."""
        from keras.preprocessing import image as keras_image

        # Convert BGR (OpenCV) to RGB (Keras expects RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for consistent behavior with keras_image.load_img
        img_pil = keras_image.array_to_img(rgb_frame)

        # Resize using Keras' utility (PIL-based)
        img_resized = keras_image.smart_resize(np.array(img_pil), (224, 224))

        # Normalize
        img_array = img_resized / 255.0
        img_array_exp = np.expand_dims(img_array, axis=0)

        # Predict
        prediction = self.scene_classifier.predict(img_array_exp)[0][0]
        return 'indoor' if prediction <= 0.5 else 'outdoor'

    def detect_objects(self, frame, scene_type):
        model = self.indoor_detector if scene_type == 'indoor' else self.outdoor_detector
        results = model(frame)
        
        objects = []
        for result in results:
            for box in result.boxes:
                objects.append({
                    'class': result.names[int(box.cls[0])],
                    'bbox': box.xyxy[0].tolist()
                })
        return objects

    def estimate_depth(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            prediction = self.depth_estimator(img_tensor)

        depth_map = prediction.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        depth_map -= depth_map.min()
        depth_map /= (depth_map.max() + 1e-8)

        # Invert: to convert "inverse depth" → actual relative depth
        depth_map = 1.0 - depth_map

        # Resize to match original frame
        depth_map = cv2.resize(depth_map, (frame.shape[1], frame.shape[0]))

        return depth_map

    def get_object_distance(self, bbox, depth_map):
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure valid bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)

        # Extract central 40% region of the box (to avoid noisy edges)
        box_w = x2 - x1
        box_h = y2 - y1

        cx1 = int(x1 + box_w * 0.3)
        cy1 = int(y1 + box_h * 0.3)
        cx2 = int(x2 - box_w * 0.3)
        cy2 = int(y2 - box_h * 0.3)

        if cx1 >= cx2 or cy1 >= cy2:
            return 5.0  # fallback for very small boxes

        center_region = depth_map[cy1:cy2, cx1:cx2]

        if center_region.size == 0:
            return 5.0

        # Use 30th percentile to reduce noise from outliers (tweakable)
        central_depth = np.percentile(center_region, 30)

        distance = central_depth * self.depth_scale
        return np.clip(distance, 0.05, 10.0)  # Clamp to 5cm–10m
    
    def generate_description(self, scene_type, objects, depth_map):
        description = f"We are {'indoors' if scene_type == 'indoor' else 'outdoors'}. "
        
        object_counts = defaultdict(int)
        closest_info = {'object': None, 'distance': float('inf'), 'position': None}
        object_positions = []
        
        for obj in objects:
            class_name = obj['class']
            object_counts[class_name] += 1
            
            # Get object center position (normalized 0-1 from left to right)
            x1, y1, x2, y2 = obj['bbox']
            center_x = (x1 + x2) / 2 / depth_map.shape[1]
            center_y = (y1 + y2) / 2 / depth_map.shape[0]
            
            distance = self.get_object_distance(obj['bbox'], depth_map)
            object_positions.append((class_name, distance, center_x, center_y))
            
            if distance < closest_info['distance']:
                closest_info = {
                    'object': class_name,
                    'distance': distance,
                    'position': center_x
                }
        
        # Count objects description
        if object_counts:
            description += "There are "
            items = list(object_counts.items())
            for i, (obj, count) in enumerate(items):
                if i > 0:
                    description += ", " + ("and " if i == len(items)-1 else "")
                description += f"{count} {obj}{'s' if count > 1 else ''}"
            description += ". "
        
        # Closest object description
        if closest_info['object']:
            dist = closest_info['distance']
            qualifier = "very close" if dist < 1.5 else (
                    "close" if dist < 3 else 
                    "moderately far" if dist < 6 else 
                    "far")
            
            position = closest_info['position']
            direction = "straight ahead" if 0.4 <= position <= 0.6 else (
                    "to your left" if position < 0.4 else 
                    "to your right")
            
            description += (f"The closest is {closest_info['object']} at about "
                        f"{dist:.1f} meters {qualifier}, located {direction}. ")
        
        # Add navigation suggestions
        if objects:
            # Group objects by direction
            left_objects = [obj for obj in object_positions if obj[2] < 0.4]
            center_objects = [obj for obj in object_positions if 0.4 <= obj[2] <= 0.6]
            right_objects = [obj for obj in object_positions if obj[2] > 0.6]
            
            # Calculate average distances for each direction
            def avg_distance(group):
                if not group: return float('inf')
                return sum(obj[1] for obj in group) / len(group)
            
            left_avg = avg_distance(left_objects)
            center_avg = avg_distance(center_objects)
            right_avg = avg_distance(right_objects)
            
            # Find the clearest path (direction with fewest objects or most space)
            direction_scores = {
                'left': 1/(left_avg + 0.1) * len(left_objects),
                'center': 1/(center_avg + 0.1) * len(center_objects),
                'right': 1/(right_avg + 0.1) * len(right_objects)
            }
            
            clearest_direction = min(direction_scores, key=direction_scores.get)
            
            # Generate suggestion
            if clearest_direction == 'left':
                description += "The path to your left appears more open. Consider moving left."
            elif clearest_direction == 'right':
                description += "The path to your right appears more open. Consider moving right."
            else:
                if len(center_objects) == 0:
                    description += "The center path appears clear. You can move straight ahead."
                else:
                    description += "You may continue straight ahead, but be mindful of objects in your path."
        
        return description

    def process_frame(self, frame):
        self.frame_count += 1
        
        # Skip frames for performance (process every nth frame)
        if self.frame_count % 3 != 0:
            return None, frame
        
        scene_type = self.classify_scene(frame)
        objects = self.detect_objects(frame, scene_type)
        depth_map = self.estimate_depth(frame)
        description = self.generate_description(scene_type, objects, depth_map)
        
        # Visualize results
        display_frame = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            distance = self.get_object_distance(obj['bbox'], depth_map)
            label = f"{obj['class']} ({distance:.2f}m)"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
        return description, display_frame

    def process_video(self, video_source=0):
        """Process video from file (provide path) or webcam (default 0)"""
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error opening video source {video_source}")
            return
        
        # Get video properties for output file if needed
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output window
        cv2.namedWindow("Vision Assistant", cv2.WINDOW_NORMAL)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            description, processed_frame = self.process_frame(frame)
            
            # Display processed frame
            cv2.imshow("Vision Assistant", processed_frame)
            
            # Speak description (with cooldown to avoid overlapping speech)
            current_time = time.time()
            if description and (current_time - self.last_tts_time) > self.tts_cooldown:
                self.tts_engine.say(description)
                self.tts_engine.runAndWait()
                self.last_tts_time = current_time
            
            # Exit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    assistant = VisionAssistant(debug_mode=True)
    
    # For webcam (default camera)
    assistant.process_video()
    
    # For video file
    # assistant.process_video("path/to/your/video.mp4")