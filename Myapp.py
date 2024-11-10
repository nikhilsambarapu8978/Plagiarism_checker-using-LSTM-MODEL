import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import hashlib
import os

# Load the model
model = tf.keras.models.load_model('quantized_model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# File to store uploaded text hashes
hashes_file = 'uploaded_texts_hashes.txt'

# Load existing hashes if the file exists
if os.path.exists(hashes_file):
    with open(hashes_file, 'r') as file:
        uploaded_texts_hash = set(line.strip() for line in file)
else:
    uploaded_texts_hash = set()

# Dictionary to store file hashes and their corresponding texts
file_hash_mapping = {}

# Function to preprocess text
def preprocess_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=200)  # Assuming max sequence length is 200
    return padded

# Function to hash text for plagiarism check
def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

# Function to calculate Jaccard similarity
def jaccard_similarity(text1, text2):
    set1 = set(text1.split())
    set2 = set(text2.split())
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 0

# Set custom CSS styles
st.markdown(
    """
    <style>
    body {
        background-color: #f0f8ff;
    }
    .stButton>button {
        background-color: #007BFF;
        color: white;
    }
    .stButton>button:hover {
        background-color: #0056b3;
    }
    .stTextInput>div>input {
        background-color: white;
        border: 1px solid #007BFF;
        border-radius: 5px;
    }
    .stFileUploader {
        background-color: #e6f7ff;
    }
    .stMarkdown {
        color: #333333;
    }
    .stSuccess {
        color: #155724;
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
    .stError {
        color: #721c24;
        background-color: #f8d7da;
        border-color: #f5c6cb;
    }
    .header {
        background-color: #007BFF;
        color: white;
        padding: 10px;
        text-align: center;
        font-size: 24px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Navigation bar
st.markdown('<div class="header">Plagiarism Checker App</div>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Plagiarism Checker")
st.sidebar.markdown("Upload your text files and check for plagiarism.")

# Frontend using Streamlit
st.title("Plagiarism Checker ")

uploaded_files = st.file_uploader("Upload text files", type="txt", accept_multiple_files=True)

# Initialize a session state to store previously uploaded files
if 'previous_files' not in st.session_state:
    st.session_state.previous_files = []

if uploaded_files:
    st.header("Uploaded Files")
    
    for uploaded_file in uploaded_files:
        # Read the uploaded file
        text = uploaded_file.read().decode('utf-8')
        
        st.subheader(f"Content of {uploaded_file.name}:")
        st.write(text)

        # Hash the text to check for re-upload
        text_hash = hash_text(text)
        
        # Check if the hash already exists
        if text_hash in uploaded_texts_hash:
            st.error(f"Warning: The text from {uploaded_file.name} has already been uploaded. It is considered plagiarized.")
        else:
            # Add the hash to the set of uploaded texts
            uploaded_texts_hash.add(text_hash)
            
            # Preprocess the text
            processed_text = preprocess_text(text)
            
            # Make a prediction
            prediction = model.predict(processed_text)
            
            # Display the result
            label = np.argmax(prediction, axis=1)  # Assuming the model outputs class probabilities
            
            if label[0] == 1:  # Assuming 1 indicates plagiarized
                st.error(f"Warning: The text from {uploaded_file.name} appears to be plagiarized.")
            else:
                st.success(f"The text from {uploaded_file.name} appears to be original.")

            # Update the file with the new hash
            with open(hashes_file, 'a') as file:
                file.write(text_hash + '\n')
            
            # Map the file name to its content for further checks
            file_hash_mapping[uploaded_file.name] = text
            
            # Save the uploaded file name to session state
            st.session_state.previous_files.append(uploaded_file.name)

    # Check for plagiarism across uploaded files
    st.header("Plagiarism Analysis")
    
    for file_name1, text1 in file_hash_mapping.items():
        for file_name2, text2 in file_hash_mapping.items():
            if file_name1 != file_name2:
                similarity = jaccard_similarity(text1, text2)
                if similarity > 0.2:  # Adjust threshold as needed
                    plagiarism_percentage = similarity * 100
                    st.warning(f"Potential plagiarism detected between '{file_name1}' and '{file_name2}': {plagiarism_percentage:.2f}% similarity.")

# Display previously uploaded files
st.sidebar.header("Previously Uploaded Files")
if st.session_state.previous_files:
    for file_name in st.session_state.previous_files:
        st.sidebar.markdown(f"- {file_name}")
else:
    st.sidebar.markdown("No previous files uploaded.")

# Footer
st.sidebar.markdown("### About this App")
st.sidebar.markdown("This application checks for plagiarism by comparing uploaded text files.")
st.sidebar.markdown("This application is made by Batch 15")
