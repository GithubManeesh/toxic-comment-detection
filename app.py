import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
import pandas as pd
import nltk

# Download NLTK data if needed
try:
    stop_words = set(stopwords.words("english"))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words("english"))

# Page configuration
st.set_page_config(
    page_title="Toxic Comment Detector",
    page_icon="🛡️",
    layout="wide"
)

# Initialize session state for results
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'last_input' not in st.session_state:
    st.session_state.last_input = ""
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'last_probabilities' not in st.session_state:
    st.session_state.last_probabilities = None

# Load model and vectorizer
@st.cache_resource
def load_models():
    model = joblib.load("toxicity_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

try:
    model, vectorizer = load_models()
    model_loaded = True
except:
    model_loaded = False
    st.error("⚠️ Model files not found. Please ensure 'toxicity_model.pkl' and 'tfidf_vectorizer.pkl' are in the correct directory.")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Function to perform analysis
def analyze_comment():
    if st.session_state.user_input and st.session_state.user_input.strip():
        st.session_state.analysis_done = True
        st.session_state.last_input = st.session_state.user_input
        cleaned = clean_text(st.session_state.user_input)
        vector = vectorizer.transform([cleaned])
        prediction = model.predict(vector)[0]
        probabilities = model.predict_proba(vector)

    st.session_state.last_prediction = prediction
    st.session_state.last_probabilities = probabilities

# Title and header
st.title("🛡️ Toxic Comment Detection System")
st.markdown("Detect cyberbullying and harmful comments using Machine Learning")

# Create tabs
tab1, tab2, tab3 = st.tabs(["📝 Analyze", "📊 Statistics", "ℹ️ About"])

with tab1:
    if not model_loaded:
        st.error("⚠️ Cannot analyze comments because model files are missing.")
        st.info("Please make sure 'toxicity_model.pkl' and 'tfidf_vectorizer.pkl' are in the same folder as this app.")
    else:
        # Input section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Enter a comment to analyze")
            user_input = st.text_area(
                "Type your comment below:",
                height=120,
                placeholder="Example: You are a stupid idiot...",
                label_visibility="collapsed",
                key="user_input"
            )
            
            # Analyze button
            st.button("🔍 Analyze Comment", type="primary", use_container_width=True, on_click=analyze_comment)
        
        with col2:
            st.subheader("📝 Example Comments")
            
            # Non-clickable examples in a nice box
            with st.container(border=True):
                st.markdown("**Safe Comment:**")
                st.caption("'Thank you for your help! I really appreciate it.'")
                
                st.markdown("**Toxic Comment (Insult):**")
                st.caption("'You are so stupid and worthless!'")
                
                st.markdown("**Toxic Comment (Threat):**")
                st.caption("'I will find you and hurt you badly!'")
                
                st.markdown("**Neutral Comment:**")
                st.caption("'What time is the meeting tomorrow?'")
        
        st.divider()
        
        # Prediction section
        if st.session_state.analysis_done and st.session_state.last_prediction is not None:
            user_input = st.session_state.last_input
            prediction = st.session_state.last_prediction
            probabilities = st.session_state.last_probabilities
            
            with st.spinner("🔬 Analyzing comment..."):
                # Process the comment
                cleaned = clean_text(user_input)
                
                labels = [
                    "Toxic",
                    "Severe Toxic",
                    "Obscene",
                    "Threat",
                    "Insult",
                    "Identity Hate"
                ]
                
                scores = []
                for prob in probabilities: 
                    scores.append(prob[0][1])

                # Create results dataframe
                results = pd.DataFrame({
                    "Category": labels,
                    "Status": ["🔴 Detected" if x == 1 else "🟢 Not Detected" for x in prediction],
                    "Value": prediction
                })
                
                # Results header
                st.subheader("📊 Analysis Results")
                
                # Show original and cleaned text
                with st.expander("View original and processed text"):
                    col_orig, col_clean = st.columns(2)
                    with col_orig:
                        st.markdown("**Original:**")
                        st.info(user_input)
                    with col_clean:
                        st.markdown("**Processed (after cleaning):**")
                        if cleaned:
                            st.code(cleaned, language="text")
                        else:
                            st.code("No words left after cleaning (all were stopwords or special characters)", language="text")
                
                # Overall verdict in a prominent container
                if sum(prediction) == 0:
                    with st.container(border=True):
                        st.success("### ✅ SAFE COMMENT")
                        st.markdown("No toxic content detected in this comment.")
                else:
                    toxic_count = sum(prediction)
                    toxic_categories = [labels[i] for i, val in enumerate(prediction) if val == 1]
                    with st.container(border=True):
                        st.error("### ⚠️ TOXIC COMMENT DETECTED")
                        st.markdown(f"Found **{toxic_count}** toxic categor{'y' if toxic_count == 1 else 'ies'}:")
                        for cat in toxic_categories:
                            st.markdown(f"- 🔴 **{cat}**")
                
                st.divider()
                
                # Detailed results in columns
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.subheader("📋 Detailed Analysis")
                    st.subheader("📊 Toxicity Probability")
                    for label, score in zip(labels, scores):
                        st.write(f"{label}: {round(score*100,2)}%")
                        st.progress(score)
                    
                    # Create a dataframe for display
                    display_df = results[['Category', 'Status']].copy()
                    
                    # Display dataframe
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=250,
                        hide_index=True,
                        column_config={
                            "Category": "Toxicity Category",
                            "Status": "Detection Status"
                        }
                    )
                
                with col_res2:
                    st.subheader("📊 Summary")
                    
                    # Metrics
                    toxic_count = sum(prediction)
                    safe_count = 6 - toxic_count
                    
                    # Create metrics in columns
                    met_col1, met_col2 = st.columns(2)
                    with met_col1:
                        st.metric(
                            label="Toxic Categories",
                            value=toxic_count,
                            delta=None
                        )
                    with met_col2:
                        st.metric(
                            label="Safe Categories",
                            value=safe_count,
                            delta=None
                        )
                    
                    # Progress bar for toxicity level
                    st.markdown("**Toxicity Level:**")
                    toxicity_percentage = (toxic_count / 6) * 100
                    st.progress(
                        toxicity_percentage / 100,
                        text=f"{toxicity_percentage:.0f}% Toxic"
                    )
                    
                    # Simple category status
                    st.markdown("**Category Status:**")
                    
                    # Create a grid for category status
                    status_cols = st.columns(2)
                    for i, (cat, status) in enumerate(zip(labels, prediction)):
                        col_idx = i % 2
                        with status_cols[col_idx]:
                            if status == 1:
                                st.markdown(f"🔴 **{cat}**")
                            else:
                                st.markdown(f"🟢 {cat}")
        
        elif st.session_state.analysis_done and not st.session_state.user_input:
            st.warning("⚠️ Please enter a comment first.")

with tab2:
    st.subheader("📊 Detection Statistics")
    
    # Sample statistics (in a real app, these would come from a database)
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    
    with col_stat1:
        st.metric(
            label="Total Analyses",
            value="1,234",
            delta="+56 today"
        )
    with col_stat2:
        st.metric(
            label="Safe Comments",
            value="856",
            delta="69%"
        )
    with col_stat3:
        st.metric(
            label="Toxic Comments",
            value="378",
            delta="31%"
        )
    with col_stat4:
        st.metric(
            label="Accuracy",
            value="94%",
            delta="+2%"
        )
    
    st.divider()
    
    # Category distribution
    st.subheader("Toxicity Distribution by Category")
    
    # Sample data
    category_data = pd.DataFrame({
        'Category': ['Toxic', 'Severe Toxic', 'Obscene', 'Threat', 'Insult', 'Identity Hate'],
        'Count': [150, 45, 98, 23, 112, 34]
    })
    
    # Simple bar chart
    st.bar_chart(
        category_data.set_index('Category'),
        use_container_width=True,
        height=400
    )
    
    # Show raw data
    with st.expander("View raw statistics"):
        st.dataframe(
            category_data,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Category": "Toxicity Category",
                "Count": "Number of Detections"
            }
        )

with tab3:
    st.subheader("ℹ️ About This Project")
    
    st.markdown("""
    ### 🎯 Project Overview
    This **Toxic Comment Detection System** uses Machine Learning to identify harmful content in text. 
    It can detect multiple types of toxicity in comments to help create safer online spaces.
    
    ### 📋 Toxicity Categories
    """)
    
    # Category descriptions in an organized layout
    cat_col1, cat_col2 = st.columns(2)
    
    with cat_col1:
        with st.container(border=True):
            st.markdown("**🔴 Toxic**")
            st.caption("General toxic, rude, or disrespectful content")
        
        with st.container(border=True):
            st.markdown("**🔴 Severe Toxic**")
            st.caption("Extremely toxic, hateful, or violent content")
        
        with st.container(border=True):
            st.markdown("**🔴 Obscene**")
            st.caption("Obscene, vulgar, or profane language")
    
    with cat_col2:
        with st.container(border=True):
            st.markdown("**🔴 Threat**")
            st.caption("Threatening or violent content towards others")
        
        with st.container(border=True):
            st.markdown("**🔴 Insult**")
            st.caption("Insulting, demeaning, or offensive language")
        
        with st.container(border=True):
            st.markdown("**🔴 Identity Hate**")
            st.caption("Hate speech based on race, gender, religion, etc.")
    
    st.divider()
    
    # How it works section
    st.subheader("⚙️ How It Works")
    
    step1, step2, step3, step4 = st.columns(4)
    with step1:
        with st.container(border=True):
            st.markdown("**1️⃣ Input**")
            st.caption("Enter or paste your comment")
    
    with step2:
        with st.container(border=True):
            st.markdown("**2️⃣ Process**")
            st.caption("Text is cleaned and converted to numbers")
    
    with step3:
        with st.container(border=True):
            st.markdown("**3️⃣ Analyze**")
            st.caption("ML model predicts toxicity levels")
    
    with step4:
        with st.container(border=True):
            st.markdown("**4️⃣ Results**")
            st.caption("Get instant feedback on all 6 categories")
    
    st.divider()
    
    # Additional info in columns
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.info("""
        **🔒 Privacy & Security**
        - ✅ All analysis is done locally
        - ✅ No data is stored or shared
        - ✅ Your comments remain private
        - ✅ No internet connection required
        """)
    
    with info_col2:
        st.success("""
        **🚀 Features**
        - ✅ Real-time toxicity detection
        - ✅ 6 different toxicity categories
        - ✅ Easy-to-use interface
        - ✅ Instant results
        - ✅ Free to use
        """)

