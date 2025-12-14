import streamlit as st
import pandas as pd


import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import plotly.express as px
import plotly.graph_objects as go

# Download NLTK VADER lexicon
nltk.download("vader_lexicon")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="🏢 Mall Feedback",
    layout="wide",
    page_icon="✨"
)

# =========================
# Custom CSS: Sleek & Modern
# =========================
st.markdown("""
<style>
/* Background & Font */
.main {
    background-color: #f4f6fa;
    font-family: "Segoe UI", "Helvetica Neue", sans-serif;
}

/* Headings */
h1,h2,h3,h4{
    color:#1c2833;
    font-weight:700;
    margin-bottom:0.5rem;
}

/* User info emphasis */
.bold-label{
    font-size:18px;
    font-weight:800;
    color:#2c3e50;
}

/* Chat bubbles */
.chat-bubble {
    padding:16px 20px;
    border-radius:24px;
    max-width:80%;
    margin-bottom:12px;
    font-size:16px;
    line-height:1.6;
    transition: all 0.2s ease-in-out;
}
.bot {
    background-color:#f0f4ff;
    color:#2c3e50;
    margin-right:auto;
    border-left:6px solid #4c82f7;
    box-shadow:0 3px 10px rgba(0,0,0,0.08);
}
.user {
    background-color:#4c82f7;
    color:white;
    margin-left:auto;
    border-right:6px solid #3b6de0;
    box-shadow:0 3px 10px rgba(0,0,0,0.12);
}

/* Metric cards */
.metric-card {
    background:white;
    border-radius:18px;
    padding:28px;
    text-align:center;
    box-shadow:0 10px 24px rgba(0,0,0,0.08);
    margin-bottom:24px;
    transition: transform 0.25s ease-in-out;
}
.metric-card:hover{
    transform: translateY(-6px) scale(1.02);
}
.metric-value{
    font-size:30px;
    font-weight:800;
    color:#2c3e50;
}
.metric-label{
    font-size:15px;
    color:#7f8c8d;
    margin-top:10px;
}

/* Survey container */
.survey-container{
    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0 10px 24px rgba(0,0,0,0.08);
    height:100%;
    overflow-y:auto;
}

/* User info card */
.user-info-card{
    background:white;
    padding:28px;
    border-radius:18px;
    margin-bottom:24px;
    box-shadow:0 8px 18px rgba(0,0,0,0.07);
}

/* Buttons */
button.stButton>button{
    width:100%;
    padding:14px 0;
    border-radius:12px;
    font-weight:700;
    font-size:16px;
    transition: background 0.2s ease-in-out;
}
button.stButton>button:hover{
    opacity:0.92;
    transform: scale(1.02);
}

/* Thank You Fullscreen Overlay */
@keyframes sparkle {
  0% { opacity: 0; transform: scale(0.5) rotate(0deg);} 
  50% { opacity: 1; transform: scale(1.2) rotate(180deg);} 
  100% { opacity: 0; transform: scale(0.5) rotate(360deg);} 
}
@keyframes fadeout {
  from { opacity: 1; }
  to { opacity: 0; visibility: hidden; }
}
.overlay {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 100%;
  background: rgba(255,255,255,0.96);
  z-index: 9999;
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  animation: fadeout 0.8s ease forwards;
  animation-delay: 2s;
}
.sparkle {
  position: absolute;
  font-size: 32px;
  animation: sparkle 2s infinite;
  opacity: 0;
}
.thankyou {
  font-size: 50px;
  font-weight: 900;
  color: #2c3e50;
  text-align: center;
  margin-bottom: 15px;
}
.subnote {
  font-size: 22px;
  color: #7f8c8d;
}
</style>
""", unsafe_allow_html=True)

# =========================
# Initialize Session State
# =========================
if "user_info" not in st.session_state:
    st.session_state.user_info = {}
if "qid" not in st.session_state:
    st.session_state.qid = 0
if "responses" not in st.session_state:
    st.session_state.responses = {}

# =========================
# Questions & Ratings
# =========================
questions = {
    0: "Are you satisfied with the cleanliness of the mall?",
    1: "Are you happy with the variety of shops in the mall?",
    2: "Are you satisfied with the billing counter service?",
    3: "Do you find the mall staff helpful?",
    4: "Are you happy with the food court options?",
    5: "Would you recommend this mall to others?"
}
rating_map = {"😊 Good": 5, "😐 Average": 3, "😞 Worst": 1}

# =========================
# Layout: Dashboard & Chatbot
# =========================
left, right = st.columns([2,1])

# -------------------------
# LEFT: Dashboard
# -------------------------
with left:
    st.markdown("### 🧑 USER DETAILS")
    
    # User Form
    if not st.session_state.user_info:
        with st.form("user_form", clear_on_submit=True):
            name = st.text_input("👤 Name")
            age = st.number_input("📅 Age", min_value=10, max_value=100, step=1)
            gender = st.selectbox("⚧ Gender", ["Male", "Female", "Other"])
            gmail = st.text_input("📧 Email")
            submitted = st.form_submit_button("✅ Save & Start Chat")
            if submitted:
                if name and gmail:
                    st.session_state.user_info = {"Name": name, "Age": age, "Gender": gender, "Email": gmail}
                    st.rerun()
                else:
                    st.error("Please fill in all required fields.")

    # User Info Card
    if st.session_state.user_info:
        st.markdown("<div class='user-info-card'>", unsafe_allow_html=True)
        st.markdown("#### 👤 User Information")
        for k,v in st.session_state.user_info.items():
            st.markdown(f"<span class='bold-label'>{k}:</span> {v}", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Feedback Analytics
    if st.session_state.responses:
        st.markdown("### 📊 Feedback Analytics")
        df = pd.DataFrame([{"Question": questions[q], "Response": resp[0], "Rating": resp[1]} for q,resp in st.session_state.responses.items()])
        for k,v in st.session_state.user_info.items():
            df[k] = v

        sia = SentimentIntensityAnalyzer()
        df["Sentiment"] = df["Response"].apply(lambda x: sia.polarity_scores(x)["compound"])
        sentiment_score = df["Sentiment"].mean() * 100

        # Metric Cards
        col1,col2,col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>👥 1,250</div><div class='metric-label'>Total Users</div></div>", unsafe_allow_html=True)
        with col2:
            avg_rating = df["Rating"].mean()
            st.markdown(f"<div class='metric-card'><div class='metric-value'>⭐ {avg_rating:.1f}/5</div><div class='metric-label'>Average Rating</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>💡 {sentiment_score:.0f}</div><div class='metric-label'>Positive Sentiment Score</div></div>", unsafe_allow_html=True)

        # Charts: Gauge & Donut
        col1,col2 = st.columns(2)
        with col1:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=sentiment_score,
                title={'text': "Sentiment Score", 'font': {'size': 20,'color':'#2c3e50'}},
                delta={'reference':50,'increasing':{'color':"#27ae60"}, 'decreasing':{'color':'#e74c3c'}},
                gauge={'axis':{'range':[0,100]}, 'bar':{'color':"#4c82f7"},
                       'steps':[{'range':[0,40],'color':'#e74c3c'},
                                {'range':[40,70],'color':'#f1c40f'},
                                {'range':[70,100],'color':'#2ecc71'}]}
            ))
            fig_gauge.update_layout(height=320, margin=dict(l=20,r=20,t=50,b=20), font={'family':'Segoe UI','color':'#2c3e50'})
            st.plotly_chart(fig_gauge, use_container_width=True)
        with col2:
            rating_counts = df["Response"].value_counts().reset_index()
            rating_counts.columns=["Response","Count"]
            fig_donut = px.pie(
                rating_counts,
                names="Response",
                values="Count",
                hole=0.55,
                color="Response",
                color_discrete_map={"😊 Good":"#2ecc71","😐 Average":"#f1c40f","😞 Worst":"#e74c3c"}
            )
            fig_donut.update_traces(textinfo="percent+label", pull=[0.08]*len(rating_counts),
                                     marker=dict(line=dict(color='#f4f6fa',width=2)))
            fig_donut.update_layout(title={'text':'Feedback Distribution','x':0.5}, height=320,
                                    margin=dict(l=20,r=20,t=40,b=20), showlegend=True)
            st.plotly_chart(fig_donut,use_container_width=True)

        # Detailed Ratings Bar Chart
        st.markdown("### 📋 DETAILED RATING")
        fig_bar = px.bar(df, y="Question", x="Rating", orientation="h",
                         color="Rating", text="Rating",
                         color_continuous_scale=["#e74c3c","#f1c40f","#2ecc71"],
                         hover_data={"Response":True,"Sentiment":True})
        fig_bar.update_traces(texttemplate='%{text:.1f}', textposition="outside",
                              marker_color=df["Rating"].apply(lambda x:"#2ecc71" if x>=4 else ("#f1c40f" if x==3 else "#e74c3c")))
        fig_bar.update_layout(xaxis_title="Rating (1-5)", yaxis_title="",
                              height=450, margin=dict(l=20,r=20,t=40,b=20),
                              plot_bgcolor='white', paper_bgcolor='white',
                              font=dict(family="Segoe UI", color="#2c3e50"))
        st.plotly_chart(fig_bar,use_container_width=True)

        # CSV Export
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("📥 Download Feedback CSV", data=csv, file_name=f"mall_feedback_{st.session_state.user_info['Name']}.csv", mime="text/csv")

# -------------------------
# RIGHT: Chatbot
# -------------------------
with right:
    st.markdown("<div class='survey-container'>", unsafe_allow_html=True)
    st.markdown("### 🤖 FEEDBACK CHATBOT")

    qid = st.session_state.qid
    st.markdown("---")
    st.markdown("<div class='chat-bubble bot'>Hi there! Let's analyze your feedback together. 💬</div>", unsafe_allow_html=True)

    # Previous Q&A
    for i in range(qid):
        st.markdown(f"<div class='chat-bubble bot'>🤖 {questions[i]}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-bubble user'>🧑 {st.session_state.responses[i][0]}</div>", unsafe_allow_html=True)

    # Current Question
    if st.session_state.user_info and qid < len(questions):
        qtext = questions[qid]
        st.markdown(f"<div class='chat-bubble bot'>🤖 {qtext}</div>", unsafe_allow_html=True)
        col1,col2,col3 = st.columns(3)
        if col1.button("😊 Good", key=f"good_{qid}"):
            st.session_state.responses[qid] = ("😊 Good", rating_map["😊 Good"])
            st.session_state.qid += 1
            st.rerun()
        if col2.button("😐 Average", key=f"avg_{qid}"):
            st.session_state.responses[qid] = ("😐 Average", rating_map["😐 Average"])
            st.session_state.qid += 1
            st.rerun()
        if col3.button("😞 Worst", key=f"worst_{qid}"):
            st.session_state.responses[qid] = ("😞 Worst", rating_map["😞 Worst"])
            st.session_state.qid += 1
            st.rerun()
    elif st.session_state.user_info:
        st.markdown("<div class='chat-bubble bot'>🎉 Thank you! You've completed the survey.</div>", unsafe_allow_html=True)

        # Fullscreen Sparkle + Auto-hide after 2s
        st.markdown("""
        <div class='overlay'>
          <!-- Sparkles -->
          <div class='sparkle' style='top:10%; left:20%; animation-delay:0s;'>✨</div>
          <div class='sparkle' style='top:30%; left:70%; animation-delay:0.5s;'>✨</div>
          <div class='sparkle' style='top:60%; left:40%; animation-delay:1s;'>✨</div>
          <div class='sparkle' style='top:80%; left:80%; animation-delay:1.5s;'>✨</div>
          <div class='sparkle' style='top:50%; left:10%; animation-delay:2s;'>✨</div>

          <!-- Thank You Popup -->
          <div class='thankyou'>🎉 THANK YOU FOR YOUR FEEDBACK 🎉</div>
          <div class='subnote'>Your insights help us improve and serve you better 💖</div>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("</div>", unsafe_allow_html=True)



