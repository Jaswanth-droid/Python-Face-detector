# Python-Face-detector
This is a python face detection model that uses computer vision and machine learning models to identify the facial features of a person and detect their name .
A high-accuracy face recognition system built with Python and OpenCV. This project uses **Haar Cascades** for face detection and the **LBPH (Local Binary Patterns Histograms)** algorithm for facial recognition. It supports real-time recognition via webcam and is optimized for both Windows/Linux and Mobile (Pydroid 3).

##  Key Features

*   **Real-Time Detection:** Fast face detection using OpenCV Haarcascades.
*   **LBPH Recognition:** Local Binary Patterns Histograms for robust facial signature matching.
*   **Auto-Training:** Automatically scans the `known_faces` directory and trains the model on startup.
*   **Enhanced Accuracy:** 
    *   **Histogram Equalization:** Normalizes lighting conditions for better recognition in varied environments.
    *   **Standardized ROI:** Resizes regions of interest to 200x200 pixels for consistent performance.
*   **Confidence Scoring:** Provides a percentage-based confidence level for each recognized face.
*   **Mobile Support:** Includes a dedicated version for Android via Pydroid 3
