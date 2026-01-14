import cv2
import os
import numpy as np

def train_model(known_faces_dir):
    # Initialize face detector and recognizer
    # We use the built-in Haarcascade for detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # LBPH Face Recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    ids = []
    names = {} # Map ID -> Name
    current_id = 0

    print(f"Training model on faces in '{known_faces_dir}'...")

    # Walk through the directory
    # Structure can be:
    #   known_faces/Elon Musk.jpg  (Single file)
    #   known_faces/Elon Musk/1.jpg (Folder)
    
    for root, dirs, files in os.walk(known_faces_dir):
        for filename in files:
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                path = os.path.join(root, filename)
                
                # Determine name from folder name or filename
                if root == known_faces_dir:
                    # File is directly in known_faces, use filename as name
                    name = os.path.splitext(filename)[0]
                else:
                    # File is in a subfolder, use folder name as name
                    name = os.path.basename(root)

                # Assign a unique ID to each name
                if name not in [n for n in names.values()]:
                    names[current_id] = name
                    this_id = current_id
                    current_id += 1
                else:
                    # Find the existing ID for this name
                    for key, val in names.items():
                        if val == name:
                            this_id = key
                            break
                
                # Read image
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                # --- ACCURACY IMPROVEMENT: Histogram Equalization ---
                # This normalizes lighting across all training images
                img = cv2.equalizeHist(img)

                # Detect faces
                detected_faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5)

                for (x, y, w, h) in detected_faces:
                    roi = img[y:y+h, x:x+w]
                    # Standardize ROI size for better recognition
                    roi = cv2.resize(roi, (200, 200))
                    faces.append(roi)
                    ids.append(this_id)
                    
                print(f"  Processed: {filename} -> {name}")

    if len(faces) == 0:
        print("Error: No faces could be used for training.")
        return None, None, None
        
    recognizer.train(faces, np.array(ids))
    print("Training complete.")
    
    return recognizer, names, face_cascade

def main():
    # Get the absolute path of the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Construct the full path to known_faces
    known_faces_dir = os.path.join(script_dir, "known_faces")
    
    print(f"Looking for faces in: {known_faces_dir}")
    
    # Train the model
    recognizer, names, face_cascade = train_model(known_faces_dir)
    
    if recognizer is None:
        return

    # Initialize Webcam
    cap = None
    for index in [0, 1]:
        print(f"Attempting to open camera index {index}...")
        temp_cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if temp_cap.isOpened():
            # Read a test frame
            ret, frame = temp_cap.read()
            if ret and np.sum(frame) > 0:
                print(f"Success with camera index {index}!")
                cap = temp_cap
                break
            else:
                print(f"Camera index {index} opened but returned invalid/black frame.")
                temp_cap.release()
        else:
            print(f"Could not open camera index {index}.")

    if cap is None:
        print("Error: Could not find a working webcam.")
        return

    print("Starting Recognition... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame.")
            break

        if np.sum(frame) == 0:
            # Skip black frames
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # --- ACCURACY IMPROVEMENT: Histogram Equalization (Live) ---
        gray = cv2.equalizeHist(gray)
        
        # Detect faces in current frame
        # High accuracy detection parameters
        faces_detected = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.05, 
            minNeighbors=6
        )

        for (x, y, w, h) in faces_detected:
            # Region of interest
            roi_gray = gray[y:y+h, x:x+w]
            # Standardize ROI size to match training
            roi_gray = cv2.resize(roi_gray, (200, 200))
            
            # Predict
            try:
                id_, confidence = recognizer.predict(roi_gray)
                
                # Confidence threshold (Distance): Lower is better
                if confidence < 75: 
                    name = names[id_]
                    conf_text = f"{int(max(0, 100 - confidence))}%"
                else:
                    name = "Unknown"
                    conf_text = ""
            except Exception as e:
                name = "Unknown"
                conf_text = ""

            # Draw
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            display_name = f"{name} ({conf_text})" if conf_text else name
            cv2.putText(frame, display_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Face Recognition (High Accuracy)', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
