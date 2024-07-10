# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 15:48:47 2024

@author: user
"""

import streamlit as st
import pdfplumber
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
import string
import xml.etree.ElementTree as ET

# 確認是否已經下載所需的NLTK資源，否則進行下載
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # 將文本轉換為小寫
    text = text.lower()
    
    # 去除標點符號
    translator = str.maketrans("", "", string.punctuation)
    text = text.translate(translator)
    
    # 分詞
    words = word_tokenize(text)
    
    # 去除停用詞
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    return ' '.join(filtered_words)

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

def load_mesh_terms(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    mesh_terms = set()
    for descriptor in root.findall(".//DescriptorRecord/DescriptorName/String"):
        mesh_terms.add(descriptor.text.lower())
    return mesh_terms

# 讀取本地 MeSH 資料文件
mesh_file_path = "desc2023.xml"  # 確保這個文件在你的 GitHub 存儲庫中
mesh_terms = load_mesh_terms(mesh_file_path)

# Streamlit App
st.title("醫學關鍵字提取")

# 上傳文件
uploaded_files = st.file_uploader("上傳你的文本或PDF文件", type=["txt", "pdf"], accept_multiple_files=True)

if uploaded_files:
    texts = []
    for uploaded_file in uploaded_files:
        if uploaded_file.type == "application/pdf":
            # 提取PDF中的文本
            text = extract_text_from_pdf(uploaded_file)
        else:
            # 提取文本文件中的文本
            text = uploaded_file.read().decode("utf-8")
        texts.append(text)
    
    # 預處理每篇文章
    processed_texts = [preprocess_text(text) for text in texts]
    
    # 使用TfidfVectorizer來計算TF-IDF值
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # 取得特徵名稱（關鍵字）
    feature_names = vectorizer.get_feature_names_out()
    
    # 顯示每篇文章的醫學關鍵字及其TF-IDF值
    for i, text in enumerate(processed_texts):
        st.subheader(f"Text {i+1}:")
        tfidf_scores = zip(feature_names, tfidf_matrix[i].toarray()[0])
        sorted_scores = sorted(tfidf_scores, key=lambda x: x[1], reverse=True)
        for keyword, score in sorted_scores:
            if score > 0 and keyword in mesh_terms:
                st.write(f"Medical Keyword: {keyword}, Score: {score:.4f}")
        st.write("\n")
else:
    st.write("請上傳一個或多個文本或PDF文件。")
