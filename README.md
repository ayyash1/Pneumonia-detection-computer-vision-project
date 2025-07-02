# 🫁 Pneumonia Detection using Teachable Machine + Streamlit

A lightweight and effective chest X-ray classifier built using **Teachable Machine (no-code)** and deployed using **Streamlit**. The model predicts whether an uploaded X-ray image shows signs of **Pneumonia** or is **Normal**.

---

## 🔧 Tools Used
- **Model Training**: [Teachable Machine](https://teachablemachine.withgoogle.com/)
- **Frameworks**: TensorFlow, Keras
- **Deployment**: Streamlit
- **Languages**: Python

---

## 📊 Model Performance

| Metric     | Score   |
|------------|---------|
| Accuracy   | 98%   |
| Precision  | 98.1%   |
| Recall     | 95.2%   |
| F1-Score   | 96.6%   |

📌 Visuals included:
- Confusion matrix  
- Accuracy/loss per epoch graphs

---

## 📁 Folder Structure
```bash
PneumoniaClassification/
├── Code/
│ ├── app.py #Streamlit app
│ ├── main.py #Prediction script
│ ├── keras_model.h5 #Trained model
│ ├── labels.txt #Class names
├── metrics/ #Evaluation visuals
├── requirements.txt #Python dependencies
└── README.md #This file
```

---

## ⚙️ Setup & Run

```bash
# Step 1: Clone the repo
git clone https://github.com/yourusername/pneumonia-detection-teachable.git
cd pneumonia-detection-teachable

# Step 2: Create virtual environment (optional)
python -m venv env
source env/bin/activate   # Windows: .\env\Scripts\activate

# Step 3: Install required packages
pip install -r requirements.txt

# Step 4: Run the Streamlit app
streamlit run Code/app.py
```
---
## 📦 requirements.txt
```bash
tensorflow==2.12.1
keras==2.12.0
opencv-python
numpy
Pillow
streamlit
```
---
## ▶️ How It Works
1.Upload an X-ray image in the Streamlit UI

2.The model processes the image

3.You'll get a prediction (Pneumonia/Normal) with a confidence score

---
## ✅ Next Improvements
- Add Grad-CAM for explainability
- Support more medical image types
- Convert model to TFLite for mobile usage
- Train on larger, real-world datasets
---
## 👤 Author
- Ahamed Ayyash
- Computer Engineering Student | Passionate about AI for Healthcare

📬LinkedIn: Ayyash Fous
💡Open for collaboration or internship opportunities

---
## 📜 License
MIT License — free to use, modify, and share with credit.
