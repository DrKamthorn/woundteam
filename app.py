import streamlit as st
from img_classification import teachable_machine_classification
from PIL import Image, ImageOps

st.title("AI/ML คาดการณ์การหายของแผลกดทับ")
st.header("หน่วยเยี่ยมบ้าน ศูนย์การแพทย์กาญจนาภิเษก")
st.text("อัปโหลดภาพแผลกดทับที่ต้องการประเมินด้วย AI/ML")

uploaded_file = st.file_uploader("Choose a photo ...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded pressure wound.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    label = teachable_machine_classification(image, 'keras_model.h5')
    if label == 1:
        st.write("สภาพแผลน่าจะมีโอกาสหายได้")
    else:
        st.write("สภาพแผลรุนแรง ยากต่อการหาย/ภาพไม่อยู่ใน Scope")