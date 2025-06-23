# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import easyocr
from datetime import datetime
from deepface import DeepFace

# ========== Functions ==========

@st.cache_data
def load_ocr_reader():
    return easyocr.Reader(['en'])

def extract_dob(text_lines):
    for line in text_lines:
        if "DOB" in line or "Birth" in line:
            # Extract DOB using simple pattern
            digits = ''.join(c if c.isdigit() or c == '/' else '' for c in line)
            if len(digits) >= 8:
                try:
                    return datetime.strptime(digits, "%d/%m/%Y")
                except:
                    continue
    return None

def calculate_age(dob):
    today = datetime.now()
    return (today - dob).days // 365

def is_blurry(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance < 100, variance

# ========== UI ==========

st.title("🧾 Age & Identity Verification System")

st.markdown("Upload a simulated Aadhar image and a selfie to verify age and identity.")

# Upload Aadhar Card
aadhar_file = st.file_uploader("Upload Aadhar Card (JPEG/PNG)", type=["jpg", "jpeg", "png"])
selfie_file = st.file_uploader("Upload Selfie Image (JPEG/PNG)", type=["jpg", "jpeg", "png"])

if aadhar_file and selfie_file:
    # Load images
    aadhar_img = Image.open(aadhar_file).convert("RGB")
    selfie_img = Image.open(selfie_file).convert("RGB")
    
    st.image([aadhar_img, selfie_img], caption=["Aadhar Image", "Selfie Image"], width=300)

    # Convert PIL to OpenCV format
    aadhar_cv = cv2.cvtColor(np.array(aadhar_img), cv2.COLOR_RGB2BGR)
    selfie_cv = cv2.cvtColor(np.array(selfie_img), cv2.COLOR_RGB2BGR)

    # Run OCR
    st.subheader("🔍 Extracting DOB from Aadhar")
    reader = load_ocr_reader()
    result = reader.readtext(np.array(aadhar_img), detail=0)
    st.write("OCR Text:", result)

    dob = extract_dob(result)
    if dob:
        age = calculate_age(dob)
        st.success(f"Extracted DOB: {dob.strftime('%d-%m-%Y')} (Age: {age})")
    else:
        st.error("Could not extract DOB")

    # Face Matching
    st.subheader("👥 Face Matching")
    try:
        analysis = DeepFace.verify(aadhar_cv, selfie_cv, enforce_detection=True)
        score = analysis['distance']
        threshold = 0.6
        confidence = max(0, round((1 - score / threshold) * 100, 2))
        match = analysis["verified"]

        st.write(f"🔢 Match Confidence Score: {confidence}%")
        st.success("✅ Face Match!") if match else st.error("❌ Face Mismatch!")

    except Exception as e:
        st.error(f"Face detection failed: {e}")

    # Blurriness Check
    st.subheader("🕵️‍♂️ Image Quality Check")
    blurry, score = is_blurry(selfie_cv)
    st.write(f"Sharpness Score: {round(score, 2)}")
    if blurry:
        st.warning("⚠️ Selfie seems blurry. Try again with better lighting or focus.")

    # Final Result
    st.subheader("📋 Final Decision")
    if dob and age >= 18 and match and not blurry:
        st.success("🎉 Verified: Person is above 18 and ID matches selfie.")
    else:
        st.error("❗ Verification Failed. Check age, face match, or image quality.")

