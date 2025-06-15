import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from extract_frames import extract_frames
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split

# Directories
violence_dir = "dataset/Violence"
non_violence_dir = "dataset/NonViolence"

X, y = [], []

# Load dataset
def load_dataset():
    for label, folder in enumerate([non_violence_dir, violence_dir]):
        for video in os.listdir(folder):
            video_path = os.path.join(folder, video)
            frames = extract_frames(video_path)
            X.append(frames)
            y.append(label)
    return np.array(X), np.array(y)

print("📦 Loading dataset...")
X, y = load_dataset()
print("✅ Dataset loaded:", X.shape, y.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
print("⚙️ Building model...")
model = Sequential([
    TimeDistributed(Conv2D(32, (3,3), activation='relu'), input_shape=(30, 64, 64, 3)),
    TimeDistributed(MaxPooling2D(2,2)),
    TimeDistributed(Flatten()),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train
print("🚀 Training model...")
model.fit(X_train, y_train, epochs=10, batch_size=8, validation_split=0.1)

# Evaluate
print("📊 Evaluating model...")
y_pred_prob = model.predict(X_test).flatten()
y_pred = (y_pred_prob > 0.5).astype("int")

# Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n✅ Accuracy:  {accuracy * 100:.2f}%")
print(f"🎯 Precision: {precision:.2f}")
print(f"🔄 Recall:    {recall:.2f}")
print(f"📌 F1-Score:  {f1:.2f}")

# Classification report
print("\n📃 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=["Non-Violence", "Violence"]))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Non-Violence", "Violence"], yticklabels=["Non-Violence", "Violence"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.savefig("roc_curve.png")
plt.close()

# Save model
print("💾 Saving model...")
os.makedirs("model", exist_ok=True)
model.save("model/violence_model.h5")
print("✅ Model saved to model/violence_model.h5")
