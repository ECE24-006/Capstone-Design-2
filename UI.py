import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import joblib
import numpy as np
import requests
import base64
from skimage.morphology import skeletonize
import os

# Load trained model and scaler
rf = joblib.load("./random_forest_92.pkl")
scaler = joblib.load("./scaler_rf_knn.pkl")

# Extract features for classification


def extract_features(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    image = cv2.resize(image, (300, 100))
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    skeleton = skeletonize(binary > 0)

    h, w = binary.shape
    aspect_ratio = h / w
    ink_density = np.sum(binary > 0) / (h * w)
    stroke_width = np.std(skeleton)

    return [aspect_ratio, ink_density, stroke_width]

# Signature detection via Roboflow API


def detect_signature_with_roboflow(image_path):
    with open(image_path, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode("utf-8")

    url = "https://detect.roboflow.com/signature_detector-lvtel-favej/1?api_key=wnqhD3ebj7crllMwV4Zq"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    response = requests.post(url, data=encoded_image, headers=headers)
    result = response.json()

    return len(result.get("predictions", [])) > 0

# Handle file upload and prediction


def upload_image():
    file_path = filedialog.askopenfilename(title="Select Signature Image")
    if file_path:
        filename = os.path.basename(file_path)
        filename_label.config(text=f"File: {filename}")

    if file_path:
        # Step 1: Signature detection
        has_signature = detect_signature_with_roboflow(file_path)
        if not has_signature:
            image = Image.open(file_path)
            image = image.resize((300, 100))
            image_tk = ImageTk.PhotoImage(image)
            img_label.config(image=image_tk)
            img_label.image = image_tk
            result_label.config(
                text="Prediction: No signature detected", fg="orange")
            messagebox.showwarning(
                "No Signature Found", "❌ No signature was detected in the image. Please try another.")
            return

        # Step 2: Display image
        image = Image.open(file_path)
        image = image.resize((300, 200))
        image_tk = ImageTk.PhotoImage(image)
        img_label.config(image=image_tk)
        img_label.image = image_tk

        # Step 3: Feature extraction and classification
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        features = extract_features(opencv_image)
        features_scaled = scaler.transform([features])
        prediction = rf.predict(features_scaled)[0]
        confidence = rf.predict_proba(features_scaled)[0][prediction] * 100

        if prediction == 1 and confidence < 70:
            label = "Forged ❌"
            result_label.config(text=f"Prediction: {label}", fg="red")
        else:
            label = "Genuine ✅" if prediction == 1 else "Forged ❌"
            result_label.config(
                text=f"Prediction: {label}", fg="green" if prediction == 1 else "red")


# GUI setup
root = tk.Tk()
root.title("Signature Verification System")
root.geometry("500x600")
root.config(bg="#f0f0f0")

font_title = ("Arial", 18, "bold")
font_labels = ("Arial", 14)
font_button = ("Arial", 12, "bold")

title_label = tk.Label(root, text="Signature Verification",
                      font=font_title, bg="#f0f0f0", fg="#4A90E2")
title_label.pack(pady=20)

upload_btn = tk.Button(root, text="Upload Signature", command=upload_image,
                      font=font_button, bg="#4A90E2", fg="white", relief="flat", height=2, width=20)
upload_btn.pack(pady=20)

filename_label = tk.Label(root, text="", font=(
    "Arial", 12), bg="#f0f0f0", fg="gray")
filename_label.pack(pady=5)

img_label = tk.Label(root, bg="#f0f0f0")
img_label.pack(pady=20)

result_label = tk.Label(root, text="Prediction: ",
                        font=font_labels, bg="#f0f0f0")
result_label.pack(pady=10)

root.mainloop()
