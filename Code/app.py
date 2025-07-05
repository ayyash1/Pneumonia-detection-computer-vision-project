import streamlit as st
from keras.models import load_model
from PIL import Image

from utils import classify, set_background

set_background('./PneumoniaClassification/2.jpeg')    

st.set_page_config(page_title="Pneumonia Detection with AI", layout="centered")

# Main content container
with st.container():
    st.markdown('<div class="transparent-box">', unsafe_allow_html=True)

    st.title('🔬 Pneumonia Detection using Deep Learning')
    st.header('Please upload a chest X-ray image')
    st.markdown("This model predicts whether the uploaded X-ray image indicates **Pneumonia** or is **Normal**. Built with Keras and trained via Teachable Machine.")

    file = st.file_uploader("Upload a chest X-ray image", type=["jpg", "jpeg", "png"])
    if file:
        st.image(file, caption="Uploaded X-ray", use_container_width=True)
        st.success("Ready for prediction!")
        # Add your model prediction code here

    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar content
with st.sidebar:
    st.markdown("## 🩺 AI Health Scanner")
    st.markdown("---")

    st.markdown("### **Working**")
    st.markdown("""
**Step 1:** Upload a chest X-ray image  
**Step 2:** The model analyzes the image  
**Step 3:** It predicts:
- 🟢 **Normal**
- 🔴 **Pneumonia**
    """)

    st.markdown("### **Model Info**")
    st.markdown("""
- **Model:** `keras_model.h5`  
- **Input Size:** `224x224 RGB`  
- **Trained via:** [Teachable Machine](https://teachablemachine.withgoogle.com/)
    """)

    st.markdown("### 👤 **Developed By**")
    st.markdown("""
**Ayyash Fous**  
Computer Vision Engineer  
[GitHub Repo](https://github.com/ayyash1/Pneumonia-detection-computer-vision-project.git)
    """)

    st.markdown("---")
    st.info("⚠️ This is a prototype for educational use only.")


#################################################################################


#load classifier
model = load_model('./PneumoniaClassification/Code/keras_model.h5')

#load class names
with open('./PneumoniaClassification/Code/labels.txt', 'r') as f:
    class_names = [line.strip() for line in f.readlines()]
    f.close()

#display images`
with st.spinner("Analyzing X-ray..."):
    if file is not None:
        image = Image.open(file).convert('RGB')

        #classify image
        class_name, conf_score = classify(image, model, class_names)

        #print classificaiton
        clean_class_name = class_name.strip()

        st.write("## {}".format(clean_class_name))  
        st.write("### score: {}%".format(int(conf_score * 1000) / 10))
