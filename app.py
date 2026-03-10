import streamlit as st
import pandas as pd
import plotly.express as px
from chatbot import chatbot_response

# Page config
st.set_page_config(page_title="WhatsApp Style Chatbot", layout="wide")

# Sidebar
st.sidebar.title("Options")
show_charts = st.sidebar.checkbox("📊 Show emotion statistics")

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []

# Input
user_input = st.text_input("💬 Enter your message:")

if st.button("Send"):
    if user_input.strip():
        emotion, reply = chatbot_response(user_input)
        st.session_state.history.append({
            "user": user_input,
            "bot": reply,
            "emotion": emotion
        })
    else:
        st.warning("Please enter a message.")

# Chat display in WhatsApp style
st.subheader("💬 Chat")
for chat in reversed(st.session_state.history):
    # Assign color per emotion
    color_bot = "#ECE5DD"  # default bot bubble
    if chat['emotion'] == "joy": color_bot = "#FFF176"      # yellow
    elif chat['emotion'] == "sadness": color_bot = "#90CAF9" # blue
    elif chat['emotion'] == "anger": color_bot = "#EF9A9A"   # red
    elif chat['emotion'] == "fear": color_bot = "#B39DDB"    # purple
    elif chat['emotion'] == "surprise": color_bot = "#FFCC80" # orange
    elif chat['emotion'] == "neutral": color_bot = "#ECE5DD"

    # Emoji for emotion
    emoji = ""
    if chat['emotion'] == "joy": emoji = "😄"
    elif chat['emotion'] == "sadness": emoji = "😢"
    elif chat['emotion'] == "anger": emoji = "😡"
    elif chat['emotion'] == "fear": emoji = "😱"
    elif chat['emotion'] == "surprise": emoji = "😮"
    elif chat['emotion'] == "neutral": emoji = "😐"

    # User message
    st.markdown(f"""
        <div style='text-align: right; margin:5px'>
            <span style='background-color:#DCF8C6; padding:10px; border-radius:15px; display:inline-block; max-width:70%'>
                {chat['user']}
            </span>
        </div>
    """, unsafe_allow_html=True)

    # Bot message
    st.markdown(f"""
        <div style='text-align: left; margin:5px'>
            <span style='background-color:{color_bot}; padding:10px; border-radius:15px; display:inline-block; max-width:70%'>
                {chat['bot']} {emoji}
            </span>
        </div>
    """, unsafe_allow_html=True)

# Sidebar statistics
if show_charts and st.session_state.history:
    st.sidebar.subheader("📊 Emotion Statistics")
    df = pd.DataFrame(st.session_state.history)
    emotion_counts = df['emotion'].value_counts().reset_index()
    emotion_counts.columns = ['Emotion', 'Count']
    fig = px.pie(emotion_counts, names='Emotion', values='Count', title="Emotions Detected")
    st.sidebar.plotly_chart(fig)