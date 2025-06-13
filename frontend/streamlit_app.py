import streamlit as st
# from annotated_text import annotated_text
from datetime import datetime, time, timedelta
import os
import json
import random

import torch
import numpy as np

# from utils import *
# from model.architectures import *
# from model.model_loader import load_model

import requests

def load_sample_data():
    with open('frontend/datas/samples.json', 'r') as f:
        data = json.load(f)
    return random.choice(data['entries'])

def main():
    st.title("Hacker News Upvote Predictor 🚀")

    # Initialize session state for form values
    if 'title' not in st.session_state:
        st.session_state.title = ""
    if 'author' not in st.session_state:
        st.session_state.author = ""
    if 'url' not in st.session_state:
        st.session_state.url = ""
    if 'date' not in st.session_state:
        st.session_state.date = datetime.now().date()
    if 'time_str' not in st.session_state:
        st.session_state.time_str = (datetime.now()+timedelta(hours=1)).strftime("%H:%M")
    if 'user_prediction' not in st.session_state:
        st.session_state.user_prediction = None

    input_container = st.container()
    with input_container:
        st.write("Enter post details below to predict the number of upvotes.")

        # Add sample button at the top
        if st.button("Load Sample Data"):
            sample = load_sample_data()
            # Parse the timestamp from the sample
            sample_time = datetime.strptime(sample['time'], "%Y-%m-%d %H:%M:%S")
            st.session_state.title = sample['title']
            st.session_state.author = sample['by']
            st.session_state.url = sample['url']
            st.session_state.date = sample_time.date()
            st.session_state.time_str = sample_time.strftime("%H:%M")

        title = st.text_input("Post Title", value=st.session_state.title)
        author = st.text_input("Author", value=st.session_state.author)
        url = st.text_input("URL Link Attached", value=st.session_state.url)
        date = st.date_input("Post Date", value=st.session_state.date)
        time_str = st.text_input("Post Time (HH:MM)", value=st.session_state.time_str)
        
        # Add user prediction field
        user_prediction = st.number_input("Your Upvote Prediction", min_value=0, step=1, value=st.session_state.user_prediction)

        st.write(f"You selected: {date} @ {time_str}")

        if st.button("Predict"):
            try:
                entered_time = datetime.strptime(time_str, "%H:%M").time()
                st.success(f"You selected: {date} @ {time_str}")
                # Combine into a single datetime object to convert to unix timestamp
                combined_datetime = f"{date} {time_str}:00"
                
                # Convert to datetime object
                dt = datetime.strptime(combined_datetime, "%Y-%m-%d %H:%M:%S")
                unix_timestamp = int(dt.timestamp())

                input_data = {
                    "title": title,
                    "url": url,
                    "user": author,
                    "timestamp": unix_timestamp
                }
            except ValueError:
                st.error("Please enter time in 24-hour HH:MM format.")
            
            try:
                backend_url = os.getenv("BACKEND_URL", "http://localhost:8888")
                response = requests.post(f"{backend_url}/predict", json=input_data)

                if response.status_code == 200:
                    prediction = response.json()["predicted_score"]
                    st.success(f"Predicted Upvotes: {prediction}")
                else:
                    st.error(f"Error from server: {response.text}")
            except Exception as e:
                st.error(f"Failed to connect to FastAPI: {e}")
main()