import os
import cv2
import numpy as np
import gradio as gr
from extract_frames import extract_frames
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("model/violence_model.h5")

# Prediction function
def predict_violence(video):
    if video is None:
        return "Please upload a video."

    # Gradio passes a dictionary in newer versions
    video_path = video if isinstance(video, str) else video.name
    frames = extract_frames(video_path)
    frames = np.expand_dims(frames, axis=0)  # Shape: (1, 30, 64, 64, 3)
    prediction = model.predict(frames)[0][0]  # Get scalar output

    label = "Violence/Accident Detected" if prediction > 0.5 else "Non-Violence"
    #percentage = round(prediction * 100, 2) if label == "Violence" else round((1 - prediction) * 100, 2)
    return label

# Gradio Interface
interface = gr.Interface(
    fn=predict_violence,
    inputs=gr.Video(label="Upload Video"),
    outputs=gr.Textbox(label="Detection"),
    title="Violence and Detection",
    description="Upload a video to detect whether it contains violence or accident or non-violence."
)

# Launch interface
if __name__ == "__main__":
    interface.launch()
