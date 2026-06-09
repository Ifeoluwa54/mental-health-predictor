import streamlit as st
import joblib
import numpy as np
import pandas as pd



model = joblib.load('mental_health_model.pkl')
encoders = joblib.load('encoders.pkl')
le_target = joblib.load('le_target.pkl')


st.set_page_config(page_title = "Mental Health Predictor", page_icon = "🧠", layout = "wide" )


st.markdown("""
    <style>
        .main {
            background-color: #f0f4f8;
            color: #2c3e50;
        }
        .title {
            font-size: 48px;
            font-weight: bold;
            color: #2c3e50;
            text-align: center;
            padding: 20px 0;
        }
        .subtitle {
            font-size: 18px;
            color: #555;
            text-align: center;
            margin-bottom: 30px;
        }
        .warning-box {
            background-color: #fff3cd;
            border-left: 5px solid #ffc107;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 25px;
        }
        .predict-btn button {
            background-color: #2c3e50;
            color: white;
            font-size: 18px;
            padding: 12px 40px;
            border-radius: 10px;
            width: 100%;
        }
        .result-box {
            padding: 25px;
            border-radius: 12px;
            text-align: center;
            font-size: 24px;
            font-weight: bold;
            margin-top: 20px;
        }
            label {
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }
        
        .stSelectbox label, .stNumberInput label {
            color: #2c3e50 !important;
            font-weight: 600 !important;
        }
        .stTabs [data-baseweb="tab"] {
    color: #2c3e50 !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    }
    
    /* Active tab */
    .stTabs [aria-selected="true"] {
        color: #3498db !important;
        border-bottom: 3px solid #3498db !important;
    }
    
    /* Section heading */
    h3 {
        color: #2c3e50 !important;
    }

    </style>
""", unsafe_allow_html=True)


st.markdown('<div class="title">🧠 Mental Health Treatment Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle" style="color: #2c3e50;">A machine learning tool to predict mental health treatment likelihood in the tech industry</div>', unsafe_allow_html=True)
st.markdown("""
    <div style='background-color: #fff3cd; border-left: 5px solid #ffc107; 
    padding: 15px; border-radius: 8px; margin-bottom: 25px; color: #856404;'>
    ⚠️ <b>Disclaimer:</b> This is not a clinical diagnosis. Please consult a qualified 
    mental health professional if you have concerns.
    </div>
""", unsafe_allow_html=True)


tab1, tab2 = st.tabs(["📖 About & Research", "🔮 Predict"])

with tab1:
    st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <h2 style='color: #2c3e50;'>Mental Health in the Tech Industry</h2>
            <p style='color: #555; font-size: 16px;'>Why this tool exists and what the research says</p>
        </div>
    """, unsafe_allow_html=True)

    # Three stat cards at the top
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #667eea, #764ba2); padding: 25px; border-radius: 12px; text-align: center; color: white;'>
                <h1>55%</h1>
                <p>of tech workers report escalating substance use due to workplace stress</p>
                <small>APN Mental Health in Tech Report, 2023</small>
            </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #f093fb, #f5576c); padding: 25px; border-radius: 12px; text-align: center; color: white;'>
                <h1>5x</h1>
                <p>more likely to be depressed than the average worker in other industries</p>
                <small>BIMA Tech Industry Report</small>
            </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
            <div style='background: linear-gradient(135deg, #4facfe, #00f2fe); padding: 25px; border-radius: 12px; text-align: center; color: white;'>
                <h1>66%</h1>
                <p>higher stress levels in tech compared to the rest of the workforce</p>
                <small>BIMA Tech Industry Report</small>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # About section
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
            <div style='background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);'>
                <h3 style='color: #2c3e50;'>🔍 About This Tool</h3>
                <p style='color: #555; line-height: 1.8;'>
                This tool was built using a real survey dataset collected by 
                <b>Kaggle</b> —
                <br><br>
                The dataset captures workplace factors, personal background, and 
                attitudes toward mental health across tech professionals globally. 
                A machine learning model was trained on this data to identify 
                patterns that predict whether someone is likely to seek mental 
                health treatment.
                </p>
            </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown("""
            <div style='background: white; padding: 25px; border-radius: 12px; box-shadow: 0 2px 10px rgba(0,0,0,0.08);'>
                <h3 style='color: #2c3e50;'>📊 Key Industry Findings</h3>
                <ul style='color: #555; line-height: 2;'>
                    <li>2 in 5 tech workers are at <b>high risk of burnout</b></li>
                    <li>62% feel emotionally and physically <b>drained</b> by their jobs</li>
                    <li>Only <b>40%</b> of big tech companies have good work-life balance</li>
                    <li>34% of tech employees say mental health resources affect their <b>decision to stay</b> at a company</li>
                    <li>77% of workers reported <b>work-related stress</b> in the last month</li>
                </ul>
                <small style='color: #999;'>Sources: Yerboo Burnout Index, Jumar Tech Survey 2024, APA 2023</small>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Call to action
    st.markdown("""
        <div style='background: linear-gradient(135deg, #2c3e50, #3498db); padding: 30px; border-radius: 12px; text-align: center; color: white;'>
            <h3>Ready to check your likelihood of seeking treatment?</h3>
            <p>Head over to the <b>🔮 Predict</b> tab and fill in your details.</p>
        </div>
    """, unsafe_allow_html=True)







with tab2:
    st.markdown("### 📋 Tell us about yourself and your workplace")
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("🎂 Age", min_value=18, max_value=100, value=25)
        gender = st.selectbox("⚧ Gender", ['Female', 'Male', 'Others'])
        country = st.selectbox("🌍 Country", ['United States', 'Canada', 'United Kingdom', 'Bulgaria', 'France', 'Portugal', 'Netherlands', 'Switzerland', 'Poland', 'Australia', 'Germany', 'Russia', 'Mexico', 'Brazil', 'Slovenia', 'Costa Rica', 'Austria', 'Ireland', 'India', 'South Africa', 'Italy', 'Sweden', 'Colombia', 'Latvia', 'Romania', 'Belgium', 'New Zealand', 'Zimbabwe', 'Spain', 'Finland', 'Uruguay', 'Israel', 'Bosnia and Herzegovina', 'Hungary', 'Singapore', 'Japan', 'Nigeria', 'Croatia', 'Norway', 'Thailand', 'Denmark', 'Bahamas, The', 'Greece', 'Moldova', 'Georgia', 'China', 'Czech Republic', 'Philippines'])
        state = st.selectbox("📍 State (US only)", ['IL', 'IN', 'unknown', 'TX', 'TN', 'MI', 'OH', 'CA', 'CT', 'MD', 'NY', 'NC', 'MA', 'IA', 'PA', 'WA', 'WI', 'UT', 'NM', 'OR', 'FL', 'MN', 'MO', 'AZ', 'CO', 'GA', 'DC', 'NE', 'WV', 'OK', 'KS', 'VA', 'NH', 'KY', 'AL', 'NV', 'NJ', 'SC', 'VT', 'SD', 'ID', 'MS', 'RI', 'WY', 'LA', 'ME'])
        family_history = st.selectbox("👨‍👩‍👧 Family History of Mental Health", ['No', 'Yes'])
        work_interfere = st.selectbox("💼 Does mental health interfere with work?", ['Often', 'Rarely', 'Never', 'Sometimes'])
    
    with col2:
        benefits = st.selectbox("🏥 Does employer provide mental health benefits?", ['Yes', "Don't know", 'No'])
        care_options = st.selectbox("💊 Are care options available?", ['Not sure', 'No', 'Yes'])
        wellness_program = st.selectbox("🧘 Wellness program available?", ['No', "Don't know", 'Yes'])
        seek_help = st.selectbox("🤝 Does employer encourage seeking help?", ['Yes', "Don't know", 'No'])
        anonymity = st.selectbox("🔒 Is anonymity protected?", ['Yes', "Don't know", 'No'])
        leave = st.selectbox("🏖️ How easy is it to take medical leave?", ['Somewhat easy', "Don't know", 'Somewhat difficult', 'Very difficult', 'Very easy'])
    
    with col3:
        mental_health_consequence = st.selectbox("⚖️ Fear of mental health consequence at work?", ['No', 'Maybe', 'Yes'])
        coworkers = st.selectbox("👥 Comfortable discussing with coworkers?", ['Some of them', 'No', 'Yes'])
        mental_health_interview = st.selectbox("🎤 Would you mention mental health in an interview?", ['No', 'Yes', 'Maybe'])
        mental_vs_physical = st.selectbox("⚖️ Is mental health taken as seriously as physical?", ['Yes', "Don't know", 'No'])
        obs_consequence = st.selectbox("👁️ Have you observed consequences for others?", ['No', 'Yes'])
        


st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🔮 Predict", use_container_width=True)

if predict_clicked:
    # Step 1 — collect all inputs into a dictionary
    input_data = {
        'Age': age,
        'Gender': gender,
        'Country': country,
        'state': state,
        'family_history': family_history,
        'work_interfere': work_interfere,
        'benefits': benefits,
        'care_options': care_options,
        'wellness_program': wellness_program,
        'seek_help': seek_help,
        'anonymity': anonymity,
        'leave': leave,
        'mental_health_consequence': mental_health_consequence,
        'coworkers': coworkers,
        'mental_health_interview': mental_health_interview,
        'mental_vs_physical': mental_vs_physical,
        'obs_consequence': obs_consequence
    }

    # Step 2 — encode the inputs
    input_df = pd.DataFrame([input_data])
    for col in input_df.columns:
        if col in encoders:
            known_classes = set(encoders[col].classes_)
            input_df[col] = input_df[col].apply(
                lambda x: encoders[col].transform([x])[0] if x in known_classes else -1
            )

    # Step 3 — make prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0]
    result = le_target.inverse_transform(prediction)[0]

    # Step 4 — display result
    st.markdown("<br>", unsafe_allow_html=True)
    if result == 'Yes':
        st.markdown(f"""
            <div style='background: linear-gradient(135deg, #f5576c, #f093fb); 
                        padding: 30px; border-radius: 12px; text-align: center; color: white;'>
                <h2>🔴 Likely to Seek Treatment</h2>
                <p style='font-size: 18px;'>Based on your responses, you are likely to seek mental health treatment.</p>
                <h3>Confidence: {round(probability[1] * 100, 1)}%</h3>
                <p>Please remember this is not a clinical diagnosis. Consider speaking with a mental health professional.</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style='background: linear-gradient(135deg,  #f7971e, #ffd200); 
                        padding: 30px; border-radius: 12px; text-align: center; color: #2c3e50;'>
                <h2>🟡 Not Likely to Seek Treatment</h2>
                <p style='font-size: 18px;'>Based on your responses, you may not seek mental health treatment — 
but that does not mean you don't need it.</p>
                <h3>Confidence: {round(probability[0] * 100, 1)}%</h3>
               <p>Barriers like stigma, cost, or workplace culture may be preventing you from seeking help. 
Consider speaking with a mental health professional regardless of this result.</p>
            </div>
        """, unsafe_allow_html=True)