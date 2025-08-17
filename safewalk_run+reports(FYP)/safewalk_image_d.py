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
import glob
from keras.preprocessing import image

class VisionAssistant:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.debug_dir = "debug_output"
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Load models
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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.depth_estimator.to(self.device)
        self.depth_scale = 5.0

    def classify_scene(self, frame):
        """Classify scene as indoor/outdoor with confidence"""
        from keras.preprocessing import image as keras_image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = keras_image.array_to_img(rgb_frame)
        img_resized = keras_image.smart_resize(np.array(img_pil), (224, 224))
        img_array = img_resized / 255.0
        prediction = self.scene_classifier.predict(np.expand_dims(img_array, axis=0))[0][0]
        confidence = abs(prediction - 0.5) * 2  # Convert to 0-1 confidence
        return ('indoor', confidence) if prediction <= 0.5 else ('outdoor', confidence)

    def detect_objects(self, frame, scene_type):
        """Detect objects with confidence filtering"""
        model = self.indoor_detector if scene_type == 'indoor' else self.outdoor_detector
        results = model(frame, conf=0.5)  # Increased confidence threshold
        return [{
            'class': result.names[int(box.cls[0])],
            'bbox': box.xyxy[0].tolist(),
            'confidence': box.conf[0].item()
        } for result in results for box in result.boxes]

    def estimate_depth(self, frame):
        """Create depth map with error handling"""
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            prediction = self.depth_estimator(img_tensor)

        depth_map = prediction.squeeze().cpu().numpy()
        depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
        return cv2.resize(1.0 - depth_map, (frame.shape[1], frame.shape[0]))

    def get_object_distance(self, bbox, depth_map):
        """Calculate object distance with safety checks"""
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
        
        if x1 >= x2 or y1 >= y2:
            return float('inf')
            
        center_region = depth_map[y1:y2, x1:x2]
        return np.percentile(center_region, 30) * self.depth_scale if center_region.size > 0 else float('inf')

    def generate_description(self, scene_type, scene_confidence, objects, depth_map):
        """Generate descriptive feedback in requested format"""
        description_parts = []
        
        # 1. Scene information
        confidence_term = "clearly" if scene_confidence > 0.7 else "appear to be"
        description_parts.append(f"You are {confidence_term} {scene_type}.")
        
        # 2. Object detection details
        if objects:
            # Group objects by direction and type
            object_groups = defaultdict(list)
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) / 2 / depth_map.shape[1]
                distance = self.get_object_distance(obj['bbox'], depth_map)
                
                direction = "left" if center_x < 0.4 else \
                          "right" if center_x > 0.6 else \
                          "center"
                object_groups[(obj['class'], direction)].append(distance)
            
            # Build object description
            obj_descriptions = []
            for (obj_type, direction), distances in object_groups.items():
                avg_dist = sum(distances)/len(distances)
                dist_term = "very close" if avg_dist < 1.5 else \
                          "close" if avg_dist < 3 else \
                          "moderately far" if avg_dist < 6 else \
                          "far away"
                
                plural = 's' if len(distances) > 1 else ''
                obj_descriptions.append(
                    f"{len(distances)} {obj_type}{plural} {dist_term} to your {direction}"
                )
            
            description_parts.append("I can see " + ", ".join(obj_descriptions) + ".")
            
            # 3. Navigation recommendation
            left_obstacles = [d for (t, d), dists in object_groups.items() 
                            if d == "left" for d in dists]
            center_obstacles = [d for (t, d), dists in object_groups.items() 
                              if d == "center" for d in dists]
            right_obstacles = [d for (t, d), dists in object_groups.items() 
                             if d == "right" for d in dists]
            
            clearance = {
                'left': min(left_obstacles) if left_obstacles else float('inf'),
                'center': min(center_obstacles) if center_obstacles else float('inf'),
                'right': min(right_obstacles) if right_obstacles else float('inf')
            }
            
            best_path = min(clearance, key=clearance.get)
            
            if clearance[best_path] > 3:
                description_parts.append("The path ahead appears clear.")
            else:
                obstacle_type = next((t for (t, d), dists in object_groups.items() 
                                   if d == best_path and min(dists) == clearance[best_path]), "object")
                description_parts.append(
                    f"Recommended path: Move slightly to your {best_path} to avoid the {obstacle_type}.")
        else:
            description_parts.append("No significant objects detected. Path is clear.")
        
        return " ".join(description_parts)

    def process_image(self, image_path):
        """Process single image with enhanced descriptive output"""
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error loading image: {image_path}")
            return

        # Process the image
        scene_type, scene_conf = self.classify_scene(frame)
        objects = self.detect_objects(frame, scene_type)
        depth_map = self.estimate_depth(frame)
        description = self.generate_description(scene_type, scene_conf, objects, depth_map)

        # Visualization
        display_frame = frame.copy()
        for obj in objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            distance = self.get_object_distance(obj['bbox'], depth_map)
            label = f"{obj['class']} {distance:.1f}m"
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(display_frame, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        print("\n=== Environment Description ===")
        print(description)
        
        self.tts_engine.say(description)
        self.tts_engine.runAndWait()
        
        cv2.imshow('Vision Assistant', display_frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    assistant = VisionAssistant(debug_mode=True)
    assistant.process_image("test_images/downloadees.jpeg")