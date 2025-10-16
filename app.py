import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


# ---------- LOAD AND TRAIN MODEL ----------
@st.cache_resource
def train_model():
    data = pd.read_csv("session_data.csv")

    label_encoder = LabelEncoder()
    data['winner_encoded'] = label_encoder.fit_transform(data['session_winner'])

    features = ['inning_number', 'session_number', 'runs_scored', 'wickets_fallen']
    X = data[features]
    y = data['winner_encoded']

    clf = RandomForestClassifier(
        n_estimators=200,
        max_features='sqrt',
        max_depth=16,
        min_samples_leaf=5,
        bootstrap=True,
        random_state=42,
        oob_score=True
    )
    clf.fit(X, y)
    return clf, label_encoder


clf, label_encoder = train_model()

# ---------- STREAMLIT UI ----------
st.title("🏏 Test Match Session Winner Predictor")
st.write("Predict who dominated a session in a Test match using runs and wickets.")

col1, col2 = st.columns(2)
with col1:
    inning_number = st.number_input("Innings Number", min_value=1, max_value=4, value=1)
    session_number = st.number_input("Session Number", min_value=1, max_value=3, value=1)
with col2:
    runs_scored = st.number_input("Runs Scored in Session", min_value=0, max_value=300, value=120)
    wickets_fallen = st.number_input("Wickets Fallen in Session", min_value=0, max_value=10, value=2)

if st.button("Predict"):
    sample = np.array([[inning_number, session_number, runs_scored, wickets_fallen]])
    prediction = clf.predict(sample)
    probabilities = clf.predict_proba(sample)

    predicted_label = label_encoder.inverse_transform(prediction)[0]
    final_prediction = ''
    if predicted_label == 1:
        final_prediction = 'Batting Team'
    elif predicted_label == -1:
        final_prediction = 'Bowling Team'
    else:
        final_prediction = 'This session is shared'
    st.subheader(f"Predicted Session Winner: **{final_prediction}**")

    st.write("### Confidence Levels:")
    prob_df = pd.DataFrame({
        'Outcome': label_encoder.classes_,
        'Confidence': np.round(probabilities[0], 3)
    })
    st.bar_chart(prob_df.set_index('Outcome'))

st.markdown("---")
st.caption("Built by Kp using Streamlit and RandomForestClassifier")
