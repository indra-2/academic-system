import os
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"

import streamlit as st
import matplotlib.pyplot as plt
from model import analyze_student

st.title("🎓 Academic Intelligence System")

# ===== INPUT =====
st.sidebar.header("Enter Data")

sem1 = st.sidebar.slider("Sem1", 0.0, 10.0, 7.0)
sem2 = st.sidebar.slider("Sem2", 0.0, 10.0, 7.0)
sem3 = st.sidebar.slider("Sem3", 0.0, 10.0, 7.0)
sem4 = st.sidebar.slider("Sem4", 0.0, 10.0, 7.0)

math = st.sidebar.slider("Math", 0, 100, 60)
physics = st.sidebar.slider("Physics", 0, 100, 60)
english = st.sidebar.slider("English", 0, 100, 60)

# ===== BUTTON =====
if st.sidebar.button("Analyze"):

    result = analyze_student(
        [sem1, sem2, sem3, sem4],
        [math, physics, english]
    )

    # ===== OUTPUT =====
    st.subheader("Results")

    col1, col2, col3 = st.columns(3)
    col1.metric("CGPA", result["cgpa"])
    col2.metric("Uncertainty", result["unc"])
    col3.metric("Percentile", f"{result['percentile']}%")

    st.write("### Risk:", result["risk"])
    st.write("### Weak Subjects:", result["weak"])

    st.write("### Recommendations")
    for r in result["rec"]:
        st.write("-", r)

    # ===== CHART =====
    st.write("### Performance Chart")

    fig, ax = plt.subplots()
    ax.bar(["CGPA"], [result["cgpa"]])
    ax.set_ylim(0,10)
    st.pyplot(fig)
