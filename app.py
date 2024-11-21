import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image

st.set_page_config(
    page_title="Banana Shelf Life Detector",
    layout="centered"
)

st.title("Banana Shelf Life Detector 🍌")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Add camera input widget
st.write("### Take a picture of a banana")
image = st.camera_input("Click to capture")

if image:
    # Convert image to OpenCV format
    bytes_data = image.getvalue()
    img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    
    with st.spinner('Analyzing banana...'):
        # Detect bananas
        results = model(img, classes=[46])  # 46 is banana class
        
        if len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            for box in boxes:
                # Get confidence
                conf = float(box.conf[0])
                
                if conf > 0.3:  # 30% confidence threshold
                    # Get coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # Extract banana region
                    banana_roi = img[y1:y2, x1:x2]
                    
                    # Analyze colors for shelf life
                    hsv = cv2.cvtColor(banana_roi, cv2.COLOR_BGR2HSV)
                    
                    # Define color ranges
                    green_mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
                    yellow_mask = cv2.inRange(hsv, (20, 50, 50), (35, 255, 255))
                    brown_mask = cv2.inRange(hsv, (10, 50, 20), (20, 255, 255))
                    
                    # Calculate percentages
                    total_pixels = banana_roi.shape[0] * banana_roi.shape[1]
                    green_percent = np.sum(green_mask > 0) / total_pixels
                    yellow_percent = np.sum(yellow_mask > 0) / total_pixels
                    brown_percent = np.sum(brown_mask > 0) / total_pixels
                    
                    # Determine shelf life
                    if green_percent > 0.3:
                        shelf_life = 2
                        color = (0, 255, 0)
                    elif yellow_percent > 0.3 and brown_percent < 0.2:
                        shelf_life = 1
                        color = (0, 255, 255)
                    else:
                        shelf_life = 0
                        color = (0, 0, 255)
                    
                    # Draw box
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    
                    # Add text
                    text = f"Shelf Life: {shelf_life} days"
                    cv2.putText(img, text, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display results
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Banana")
            
            # Show metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Green %", f"{green_percent*100:.1f}%")
            with col2:
                st.metric("Yellow %", f"{yellow_percent*100:.1f}%")
            with col3:
                st.metric("Brown %", f"{brown_percent*100:.1f}%")
            
            # Show shelf life prediction
            st.success(f"🍌 Shelf Life Prediction: {shelf_life} days")
            st.info(f"📊 Detection Confidence: {conf*100:.1f}%")
            
        else:
            st.warning("No banana detected. Please try again!")

st.markdown("---")
st.markdown("### Instructions:")
st.markdown("""
1. Allow camera access when prompted
2. Point camera at a banana
3. Click the camera button to take a picture
4. Wait for analysis
""")