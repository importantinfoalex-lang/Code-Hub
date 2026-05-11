import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Trading Lab", page_icon="📈", layout="wide")

st.title("AI Trading Lab")
st.caption("Search a company, run a signal model, and review the backtest")

ticker_map = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "NVIDIA": "NVDA",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "Alphabet": "GOOGL",
    "Meta": "META",
    "SPY ETF": "SPY",
    "QQQ ETF": "QQQ",
    "IWM ETF": "IWM"
}

with st.sidebar:
    st.header("Search")
    company = st.selectbox("Company / ETF", list(ticker_map.keys()))
    ticker = ticker_map[company]
    st.write("Ticker:", ticker)

    st.header("Settings")
    start = st.date_input("Start date")
    end = st.date_input("End date")
    threshold = st.slider("Buy threshold", 0.50, 0.70, 0.55, 0.01)
    fee = st.slider("Transaction cost per trade", 0.0, 0.01, 0.001, 0.0005)
    cooldown = st.slider("Cooldown bars after a trade", 0, 10, 2)
    run = st.button("Run model")

@st.cache_data
def load_data(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    return df.dropna()

def make_features(df):
    out = df.copy()
    out["ret1"] = out["Close"].pct_change()
    out["ret5"] = out["Close"].pct_change(5)
    out["ma10"] = out["Close"].rolling(10).mean()
    out["ma20"] = out["Close"].rolling(20).mean()
    out["vol10"] = out["Close"].pct_change().rolling(10).std()
    out["trend"] = out["ma10"] / out["ma20"] - 1
    out["target"] = (out["Close"].shift(-1) > out["Close"]).astype(int)
    return out.dropna()

def apply_cooldown(raw_signal, cooldown):
    final_signal = []
    in_position = 0
    cool = 0
    for sig in raw_signal:
        if cool > 0:
            final_signal.append(in_position)
            cool -= 1
            continue
        if sig == 1 and in_position == 0:
            in_position = 1
            cool = cooldown
        elif sig == 0 and in_position == 1:
            in_position = 0
            cool = cooldown
        final_signal.append(in_position)
    return final_signal

if run:
    df = load_data(ticker, start, end)

    if df.empty or len(df) < 60:
        st.error("Not enough data for a reliable model.")
    else:
        feat = make_features(df)
        cols = ["ret1", "ret5", "trend", "vol10"]
        X = feat[cols]
        y = feat["target"]

        split = int(len(feat) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        feat["prob_up"] = model.predict_proba(X)[:, 1]
        feat["raw_signal"] = (feat["prob_up"] > threshold).astype(int)
        feat["signal"] = apply_cooldown(feat["raw_signal"].tolist(), cooldown)

        feat["position"] = feat["signal"].shift(1).fillna(0)
        feat["trade_change"] = feat["signal"].diff().fillna(0)
        feat["trade_cost"] = np.where(feat["trade_change"] != 0, fee, 0)

        feat["strategy_ret"] = feat["position"] * feat["Close"].pct_change() - feat["trade_cost"]
        feat["buy_hold_ret"] = feat["Close"].pct_change()

        feat["strategy_equity"] = (1 + feat["strategy_ret"].fillna(0)).cumprod()
        feat["buy_hold_equity"] = (1 + feat["buy_hold_ret"].fillna(0)).cumprod()

        buys = feat[feat["trade_change"] == 1]
        sells = feat[feat["trade_change"] == -1]

        test_probs = model.predict_proba(X_test)[:, 1]
        test_preds = (test_probs > threshold).astype(int)
        acc = accuracy_score(y_test, test_preds)

        latest = feat.iloc[-1]
        signal = "BUY" if latest["prob_up"] > threshold else "CASH"

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Test accuracy", f"{acc:.2%}")
        k2.metric("Latest up probability", f"{latest['prob_up']:.2%}")
        k3.metric("Current signal", signal)
        k4.metric("Last close", f"${latest['Close']:.2f}")

        tab1, tab2, tab3, tab4 = st.tabs(["Price", "Backtest", "Trades", "Data"])

        with tab1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=feat.index, y=feat["Close"], mode="lines", name="Price"))
            fig.add_trace(go.Scatter(
                x=buys.index, y=buys["Close"], mode="markers", name="Buy",
                marker=dict(color="green", size=10, symbol="triangle-up")
            ))
            fig.add_trace(go.Scatter(
                x=sells.index, y=sells["Close"], mode="markers", name="Sell",
                marker=dict(color="red", size=10, symbol="triangle-down")
            ))
            fig.update_layout(height=600, legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.line_chart(feat[["strategy_equity", "buy_hold_equity"]])
            st.write("Strategy includes transaction costs and cooldown logic.")

        with tab3:
            trade_log = feat.loc[feat["trade_change"] != 0, ["Close", "prob_up", "signal", "trade_cost"]].copy()
            trade_log["action"] = np.where(trade_log["signal"] == 1, "BUY", "SELL")
            st.dataframe(trade_log[["action", "Close", "prob_up", "trade_cost"]].sort_index(ascending=False))

        with tab4:
            st.dataframe(feat.tail(20)[["Close", "prob_up", "raw_signal", "signal", "strategy_equity", "buy_hold_equity"]])

else:
    st.info("Choose a company in the sidebar, set dates, then tap Run model.")
