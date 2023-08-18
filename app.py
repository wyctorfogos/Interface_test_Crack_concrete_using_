import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

TF_MODEL_FILE_PATH = "./models/modelo_treinado_224x224_ONLY_TRANSFER_LEARNING_MobileNetv2_softmax_4_categorias.h5"
model = tf.keras.models.load_model(TF_MODEL_FILE_PATH)

category_dict = {'Armature Exposure': 0, 'Concrete Cracks': 1, 'Infiltration': 2, 'Normal': 3}

def obter_categoria(valor_procurado):
    for chave, valor in category_dict.items():
        if valor == valor_procurado:
            return chave
    return None

def predict_image(image_data):
    image = cv2.cvtColor(np.array(image_data), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    output = model.predict(image)
    predicted_class = obter_categoria(int(np.argmax(np.round(output)[0])))
    return predicted_class

st.set_page_config(page_title='Seu Título', page_icon=':emoji:')
logo = 'logo_bitavel.png'
st.image(logo, width=150)
st.title("Crack concrete anomaly detector AI")
st.markdown("Select the image to be used and classified.")
file = st.file_uploader("Choose the image!", type=["jpg", "png"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if st.button("Submit"):
        with st.spinner(text="Loading model ...."):
            pred = predict_image(image)
            st.write("Classe predita: ")
            st.write(pred)

