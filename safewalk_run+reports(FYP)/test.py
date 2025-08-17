import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import matplotlib.pyplot as plt

class SceneClassifierDebugger:
    def __init__(self):
        # Load model exactly as in test script
        self.model = tf.keras.models.load_model('best_model.keras')
        self.img_size = (224, 224)
        
        # Create debug directories
        self.debug_dir = "classification_debug"
        os.makedirs(self.debug_dir, exist_ok=True)
        
    def process_image_pipeline(self, img_path):
        """Pipeline processing method"""
        # OpenCV loading
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Failed to load {img_path}")
            return None
            
        # Pipeline preprocessing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb_frame, self.img_size)
        normalized = resized / 255.0
        input_tensor = np.expand_dims(normalized, axis=0)
        
        # Save pipeline-processed image
        pipeline_img = (normalized * 255).astype(np.uint8)
        cv2.imwrite(f"{self.debug_dir}/pipeline_{os.path.basename(img_path)}", 
                   cv2.cvtColor(pipeline_img, cv2.COLOR_RGB2BGR))
        
        return input_tensor

    def process_image_test(self, img_path):
        """Test script processing method"""
        # Keras loading
        img = keras_image.load_img(img_path, target_size=self.img_size)
        img_array = keras_image.img_to_array(img) / 255.0
        input_tensor = np.expand_dims(img_array, axis=0)
        
        # Save test-processed image
        test_img = (img_array * 255).astype(np.uint8)
        cv2.imwrite(f"{self.debug_dir}/test_{os.path.basename(img_path)}", 
                   cv2.cvtColor(test_img, cv2.COLOR_RGB2BGR))
        
        return input_tensor

    def compare_classification(self, img_path):
        """Run both methods and compare results"""
        # Get both versions
        pipeline_input = self.process_image_pipeline(img_path)
        test_input = self.process_image_test(img_path)
        
        if pipeline_input is None:
            return
            
        # Get predictions
        pipeline_pred = self.model.predict(pipeline_input, verbose=0)[0][0]
        test_pred = self.model.predict(test_input, verbose=0)[0][0]
        
        # Calculate difference
        diff = np.abs(pipeline_input - test_input).mean()
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show pipeline processed
        ax1.imshow(pipeline_input[0])
        ax1.set_title(f"Pipeline Processed\nPred: {pipeline_pred:.4f}")
        ax1.axis('off')
        
        # Show test processed
        ax2.imshow(test_input[0])
        ax2.set_title(f"Test Processed\nPred: {test_pred:.4f}")
        ax2.axis('off')
        
        plt.suptitle(f"Input Difference: {diff:.2e}", y=0.95)
        plt.savefig(f"{self.debug_dir}/compare_{os.path.basename(img_path)}")
        plt.close()
        
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Pipeline prediction: {pipeline_pred:.4f}")
        print(f"Test prediction: {test_pred:.4f}")
        print(f"Mean absolute difference: {diff:.2e}")
        print("="*50)
        
        return pipeline_pred, test_pred, diff

if __name__ == '__main__':
    debugger = SceneClassifierDebugger()
    
    # Test on problematic images
    test_images = [
        "problem_image1.jpg",
        "problem_image2.jpg",
        # Add more problematic images
    ]
    
    for img in test_images:
        debugger.compare_classification(img)