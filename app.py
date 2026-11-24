import streamlit as st
import pickle

# ==========================================
# 1. Load Trained Model and Vectorizer
# ==========================================

with open('bernoulli_nb.pkl', 'rb') as file:
    model = pickle.load(file)

with open('vectorizer.pkl', 'rb') as file:
    cv = pickle.load(file)

# ==========================================
# 2. Streamlit Page Configuration
# ==========================================

st.set_page_config(
    page_title="SMS Spam Detector",
    page_icon="📩",
    layout="centered"
)

# Application title
st.title("📩 SMS Spam Detection")

st.write(
    "Detect whether an SMS message is Spam "
    "or Not Spam using Bernoulli Naive Bayes."
)

st.divider()

# ==========================================
# 3. User Input Form
# ==========================================

with st.form("spam_detection_form"):

    # SMS message input field
    sms = st.text_area(
        "Enter your SMS message:",
        placeholder="Type your SMS message here...",
        height=150
    )

    # Submit button
    submit = st.form_submit_button(
        "Predict SMS",
        use_container_width=True
    )

# ==========================================
# 4. Prediction
# ==========================================

if submit:

    # Check if the input is empty
    if not sms.strip():
        st.warning("Please enter an SMS message.")

    else:
        # Apply the same text preprocessing
        # used during model training
        # Example: transformed_sms = transform_text(sms)

        transformed_sms = sms

        # Convert SMS into numerical features
        vector_input = cv.transform(
            [transformed_sms]
        )

        # Predict SMS category
        prediction = model.predict(vector_input)[0]

        # ==================================
        # 5. Display Prediction Result
        # ==================================

        if prediction == 1:
            st.error("🚨 Spam SMS Detected!")

        else:
            st.success("✅ Not Spam SMS (Ham)")

st.divider()

st.caption(
    "Machine Learning Model: Bernoulli Naive Bayes"
)