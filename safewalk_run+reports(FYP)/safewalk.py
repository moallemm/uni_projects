import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import warnings
warnings.filterwarnings("ignore")
import cv2
import numpy as np
import torch
import tensorflow as tf
from collections import defaultdict
from ultralytics import YOLO
import pyttsx3
import time
import threading
from IPython.display import display, clear_output
import matplotlib.pyplot as plt

class VisionAssistant:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        self.description_interval = 8  # seconds between each spoken output
        self.last_description_time = 0  # last time a description was spoken

        try:
            print("Loading scene classifier...")
            self.scene_classifier = tf.keras.models.load_model('best_model.keras')

            print("Loading object detectors...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.indoor_detector = YOLO('indoor_yolov8n.pt').to(device)
            self.outdoor_detector = YOLO('outdoor_yolov8n.pt').to(device)

            print("Loading depth estimator...")
            self.depth_estimator = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', pretrained=False)
            state_dict = torch.load('midas_v21_small_256.pt', map_location=torch.device('cpu'))
            self.depth_estimator.load_state_dict(state_dict)
            self.depth_estimator.eval().to(device)

        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")

        # Text-to-speech
        self.tts_engine = pyttsx3.init()
        self.tts_queue = []

        # Frame and timing
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()

        # Start TTS thread
        self.tts_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self.tts_thread.start()
        print("Vision Assistant initialized successfully!")

    def _tts_worker(self):
        while True:
            if self.tts_queue:
                message = self.tts_queue.pop(0)
                try:
                    # Use a fresh engine every time
                    engine = pyttsx3.init()
                    engine.say(message)
                    engine.runAndWait()
                    engine.stop()
                except Exception as e:
                    if self.debug_mode:
                        print(f"[TTS error] {str(e)}")
            time.sleep(0.1)


    def async_say(self, message):
        self.tts_queue.append(message)

    def classify_scene(self, frame):
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = tf.image.resize(rgb_frame/255.0, (224, 224))
            prediction = self.scene_classifier.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
            confidence = abs(prediction - 0.5) * 2
            return ('indoor', confidence) if prediction <= 0.5 else ('outdoor', confidence)
        except:
            return 'indoor', 0.5

    def detect_objects(self, frame, scene_type):
        try:
            model = self.indoor_detector if scene_type == 'indoor' else self.outdoor_detector
            results = model(frame, imgsz=320, conf=0.4, verbose=False)
            objects = []
            for result in results:
                for box in result.boxes:
                    objects.append({
                        'class': result.names[int(box.cls[0])],
                        'bbox': box.xyxy[0].tolist(),
                        'confidence': box.conf[0].item()
                    })
            return objects
        except:
            return []

    def estimate_depth(self, frame):
        try:
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to('cuda' if torch.cuda.is_available() else 'cpu')

            with torch.no_grad():
                prediction = self.depth_estimator(img_tensor)

            depth_map = prediction.squeeze().cpu().numpy()
            depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-8)
            return cv2.resize(1.0 - depth_map, (frame.shape[1], frame.shape[0]))
        except:
            return np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

    def get_object_distance(self, bbox, depth_map):
        try:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(depth_map.shape[1], x2), min(depth_map.shape[0], y2)
            if x1 >= x2 or y1 >= y2:
                return float('inf')
            region = depth_map[y1:y2, x1:x2]
            if region.size == 0:
                return float('inf')
            return np.percentile(region, 30) * 5.0
        except:
            return float('inf')

    def generate_description(self, scene_type, scene_confidence, objects, depth_map):
        description = []

        location = "indoors" if scene_type == 'indoor' else "outdoors"
        confidence = "very clearly" if scene_confidence > 0.8 else \
                     "clearly" if scene_confidence > 0.6 else \
                     "appears to be"
        description.append(f"You are {confidence} {location}.")

        if not objects:
            description.append("No significant objects detected.")
        else:
            obj_groups = defaultdict(list)
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) / 2 / depth_map.shape[1]
                distance = self.get_object_distance(obj['bbox'], depth_map)
                direction = "left" if center_x < 0.4 else "right" if center_x > 0.6 else "center"
                obj_groups[(obj['class'], direction)].append(distance)

            obj_descriptions = []
            for (obj_type, direction), distances in obj_groups.items():
                avg_dist = sum(distances) / len(distances)
                dist_desc = "very close" if avg_dist < 1 else \
                            "close" if avg_dist < 3 else \
                            "moderately far" if avg_dist < 6 else \
                            "far away"
                plural = 's' if len(distances) > 1 else ''
                obj_descriptions.append(
                    f"{len(distances)} {obj_type}{plural} {dist_desc} to your {direction}"
                )
            description.append("I can see " + ", ".join(obj_descriptions) + ".")

            left, center, right = [], [], []
            for obj in objects:
                x1, y1, x2, y2 = obj['bbox']
                center_x = (x1 + x2) / 2 / depth_map.shape[1]
                distance = self.get_object_distance(obj['bbox'], depth_map)
                if center_x < 0.4:
                    left.append(distance)
                elif center_x > 0.6:
                    right.append(distance)
                else:
                    center.append(distance)

            def clearance(dist):
                return float('inf') if not dist else min(dist)

            clearances = {
                'left': clearance(left),
                'center': clearance(center),
                'right': clearance(right)
            }

            best = min(clearances, key=clearances.get)
            if clearances[best] > 3:
                description.append("The path ahead appears clear. You can move forward safely.")
            else:
                if best == 'center':
                    description.append("Caution: Objects ahead. Proceed carefully forward.")
                else:
                    description.append(f"Recommended path: Move slightly to your {best} to avoid obstacles.")
        return " ".join(description)

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % 3 != 0:
            return None, frame

        try:
            scene_type, scene_confidence = self.classify_scene(frame)
            objects = self.detect_objects(frame, scene_type)
            depth_map = self.estimate_depth(frame)
            description = self.generate_description(scene_type, scene_confidence, objects, depth_map)

            display_frame = frame.copy()
            for obj in objects:
                x1, y1, x2, y2 = map(int, obj['bbox'])
                distance = self.get_object_distance(obj['bbox'], depth_map)
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0,255,0), 2)
                label = f"{obj['class']} {distance:.1f}m"
                cv2.putText(display_frame, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            if self.frame_count % 10 == 0:
                self.fps = 10 / (time.time() - self.last_time)
                self.last_time = time.time()

            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

            return description, display_frame

        except Exception as e:
            if self.debug_mode:
                print(f"Processing error: {str(e)}")
            return None, frame

    def run_realtime(self, camera_index=0):
        cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if not cap.isOpened():
            print(f"Error opening camera {camera_index}")
            return

        print("Starting real-time vision assistant...")
        max_frames = 3000
        frame_counter = 0

        try:
            while frame_counter < max_frames:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break

                description, processed_frame = self.process_frame(frame)
                #cv2.imshow("Vision Assistant", processed_frame)
                #if cv2.waitKey(1) & 0xFF == ord('q'):
                #    break

                # Show in notebook
                rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                plt.imshow(rgb)
                plt.axis('off')
                clear_output(wait=True)
                display(plt.gcf())
                plt.close()

                # Speak every 8 seconds
                now = time.time()
                if description and (now - self.last_description_time) > self.description_interval:
                    if self.debug_mode:
                        print("Description:", description)
                    self.async_say(description)
                    self.last_description_time = now

                frame_counter += 1

        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Vision assistant stopped.")

# Run the assistant
if __name__ == '__main__':
    assistant = VisionAssistant(debug_mode=True)
    assistant.run_realtime()
