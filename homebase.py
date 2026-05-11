import streamlit as st

st.title("🚀 My Code Hub")

page = st.sidebar.selectbox("Pick program:", ["Trading Signals", "Sauna Log", "Notes"])

if page == "Trading Signals":
    st.header("AI Trading")
    if st.button("Run Backtest"): 
        st.success("✅ Trading signals ready! (Add your Python here)")
        st.metric("Sharpe", 1.23)

elif page == "Sauna Log":
    st.header("Heat Sessions")
    mins = st.slider("Duration (min)", 10, 40, 25)
    temp = st.slider("Temp °F", 110, 160, 125)
    if st.button("Log it"):
        st.balloons()
        st.write(f"✅ Logged: {mins}min @ {temp}°F")

else:
    st.header("Notes")
    st.text_area("Quick notes...")
