import cv2
import base64
import requests
import numpy as np

# Flask server URL
url = "http://localhost:5000/predict"

# Start the webcam capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to 48x48 (as expected by the model)
    gray_resized = cv2.resize(gray, (48, 48))
    
    # Encode the frame as base64
    _, buffer = cv2.imencode('.jpg', gray_resized)
    base64_image = base64.b64encode(buffer).decode('utf-8')
    
    # Prepare the data to send in the POST request
    data = {"image": base64_image}
    
    # Send the image to Flask API for prediction
    response = requests.post(url, json=data)
    result = response.json()
    
    # Extract prediction label and accuracy
    if 'label' in result:
        label = result['label']
        accuracy = result['accuracy']
        cv2.putText(frame, f"{label}: {accuracy}%", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the live feed with prediction
    cv2.imshow("Live Sign Language Detection", frame)
    
    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
