import torch
from transformers import BertForSequenceClassification, AutoTokenizer
import streamlit as st

LABELS = ['neutral', 'happiness', 'sadness', 'enthusiasm', 'fear', 'anger', 'disgust']

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
    model = BertForSequenceClassification.from_pretrained('Aniemore/rubert-tiny2-russian-emotion-detection')
    return tokenizer, model

tokenizer, model = load_model()

def predict_emotion(text: str) -> str:
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    predicted = torch.argmax(predicted, dim=1).numpy()
    return LABELS[predicted[0]]

def predict_emotions(text: str) -> dict:
    inputs = tokenizer(text, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**inputs)
    predicted = torch.nn.functional.softmax(outputs.logits, dim=1)
    emotions_list = {}
    for i in range(len(predicted.detach().numpy()[0].tolist())):
        emotions_list[LABELS[i]] = predicted.detach().numpy()[0].tolist()[i]
    return emotions_list

# Интерфейс Streamlit
st.title("Анализ эмоций в тексте")
st.write("Привет! Напиши предложение, и я оценю выраженные в нём эмоции.")

with st.chat_message('assistant'):
    st.write('Привет, пиши предложение и я оценю выраженные эмоции')

prompt = st.chat_input('Введите текст для анализа...')
if prompt:
    with st.chat_message('user'):
        st.write(prompt)
    with st.chat_message('assistant'):
        result = predict_emotions(prompt)
        st.write('Ваше предложение имеет следующие эмоции:')
        st.write(result)
        
        # Показываем основную эмоцию
        main_emotion = predict_emotion(prompt)
        st.success(f"Основная эмоция: **{main_emotion}**")
