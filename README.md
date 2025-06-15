# 🚨 Smart Surveillance for Violence and Accident Detection using CNN and RNN

This project presents a deep learning-based real-time surveillance system that can detect **violence** and **accidents** in 2D video sequences using a hybrid **CNN + RNN (LSTM)** model. It aims to provide intelligent monitoring solutions for public safety in environments such as smart cities, public transport, and sensitive zones.

---

## 🔍 Overview

- 📹 Extracts frames from video input
- 🧠 Uses Convolutional Neural Networks (CNNs) to extract spatial features
- 🔁 Utilizes Long Short-Term Memory (LSTM) networks to capture temporal patterns
- ✅ Classifies sequences as *Violence* or *Accident*
- 🌐 Real-time testing via an interactive **Gradio interface**

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy, Pandas
- Gradio (for user interface)
- Matplotlib / Seaborn (for visualizations)

---

## 📁 Project Structure

extract_frames.py – Extracts frames from video files

data_processing.py – Preprocesses frame data for model input

train_model.py – Trains CNN + LSTM model

test_model_interface.py – Gradio-based interface for testing the model

model/ – Directory for saving trained models

dataset/ – Contains raw and processed datasets


## 📊 Results
High accuracy achieved on standard video surveillance datasets

Successfully distinguishes between normal and violent/accident actions

Works in real-time with stable performance on test videos

## 🔮 Future Enhancements
Extend detection to more action types like theft, fire, or falls

Deploy the model on edge devices like Raspberry Pi

Add SMS/email-based alert notifications for detected incidents

