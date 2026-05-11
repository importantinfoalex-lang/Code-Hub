import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from datetime import datetime

try:
    from alpha_vantage.timeseries import TimeSeries
except Exception:
    TimeSeries = None

st.set_page_config(page_title="Adaptive Paper Trader", page_icon="📈", layout="wide")

st.title("Adaptive Paper Trader")
st.caption("Walk-forward learning, virtual cash, real market data")

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
    st.header("Portfolio")
    chosen = st.multiselect(
        "Choose assets",
        list(ticker_map.keys()),
        default=["Apple", "Microsoft", "SPY ETF"]
    )
    tickers = [ticker_map[x] for x in chosen]

    st.header("Simulation")
    start = st.date_input("Start date")
    end = st.date_input("End date")
    starting_cash = st.number_input("Starting cash", min_value=100.0, value=10000.0, step=100.0)
    threshold = st.slider("Buy threshold", 0.50, 0.70, 0.55, 0.01)
    fee = st.slider("Transaction cost per trade", 0.0, 0.01, 0.001, 0.0005)
    cooldown = st.slider("Cooldown bars after a trade", 0, 10, 2)
    refresh_secs = st.slider("Auto refresh seconds", 5, 300, 30, 5)
    train_window = st.slider("Walk-forward train window (bars)", 60, 500, 120, 10)
    test_window = st.slider("Walk-forward test window (bars)", 20, 200, 40, 5)
    alpha_key = st.text_input("Alpha Vantage API key (optional)", type="password")
    run = st.button("Run / Refresh")

@st.cache_data(ttl=60)
def load_yfinance(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)
    return df.dropna()

@st.cache_data(ttl=60)
def load_alpha_vantage(ticker, api_key):
    if TimeSeries is None or not api_key:
        return pd.DataFrame()
    ts = TimeSeries(key=api_key, output_format="pandas")
    data, _ = ts.get_daily_adjusted(symbol=ticker, outputsize="full")
    data = data.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "5. adjusted close": "Adj Close",
        "6. volume": "Volume"
    })
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    return data[["Open", "High", "Low", "Close", "Volume"]].dropna()

def load_data(ticker, start, end, alpha_key=None):
    df = load_yfinance(ticker, start, end)
    source = "yfinance"
    if df.empty and alpha_key:
        df = load_alpha_vantage(ticker, alpha_key)
        source = "alpha_vantage"
        if not df.empty:
            df = df.loc[(df.index >= pd.to_datetime(start)) & (df.index <= pd.to_datetime(end))]
    return df, source

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

def walk_forward_probs(X, y, train_window, test_window):
    probs = pd.Series(index=X.index, dtype=float)
    train_scores = []
    test_scores = []
    model_versions = []
    pos = 0

    while pos + train_window + test_window <= len(X):
        X_train = X.iloc[pos:pos + train_window]
        y_train = y.iloc[pos:pos + train_window]
        X_test = X.iloc[pos + train_window:pos + train_window + test_window]
        y_test = y.iloc[pos + train_window:pos + train_window + test_window]

        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)

        train_pred = model.predict(X_train)
        test_prob = model.predict_proba(X_test)[:, 1]
        test_pred = (test_prob > 0.5).astype(int)

        probs.iloc[pos + train_window:pos + train_window + test_window] = test_prob
        train_scores.append(accuracy_score(y_train, train_pred))
        test_scores.append(accuracy_score(y_test, test_pred))
        model_versions.append({
            "segment": len(model_versions) + 1,
            "train_start": X_train.index[0],
            "train_end": X_train.index[-1],
            "test_start": X_test.index[0],
            "test_end": X_test.index[-1],
            "train_acc": train_scores[-1],
            "test_acc": test_scores[-1]
        })
        pos += test_window

    probs = probs.ffill().bfill()
    return probs, pd.DataFrame(model_versions), train_scores, test_scores

def simulate_asset(feat, threshold, cooldown, fee, starting_cash, signal_strength):
    feat = feat.copy()
    feat["raw_signal"] = (feat["prob_up"] > threshold).astype(int)
    feat["signal"] = apply_cooldown(feat["raw_signal"].tolist(), cooldown)

    cash = starting_cash
    shares = 0.0
    equity_curve = []
    trade_rows = []
    bars_since_trade = 999

    prev_signal = 0
    entry_price = None

    for idx, row in feat.iterrows():
        price = float(row["Close"])
        signal = int(row["signal"])
        trade_cost = 0.0
        target_alloc = max(0.0, min(1.0, signal_strength.loc[idx]))
        target_cash = starting_cash * target_alloc

        if shares == 0 and signal == 1 and bars_since_trade > cooldown and cash > 0:
            buy_cash = min(cash, target_cash if target_cash > 0 else cash)
            if buy_cash > 0:
                trade_cost = buy_cash * fee
                spend = buy_cash - trade_cost
                shares = spend / price
                cash -= buy_cash
                entry_price = price
                trade_rows.append([idx, "BUY", price, shares, cash, shares * price + cash, row["prob_up"], trade_cost])
                bars_since_trade = 0

        elif shares > 0:
            hit_stop = entry_price is not None and price <= entry_price * 0.95
            hit_take = entry_price is not None and price >= entry_price * 1.10
            exit_signal = signal == 0 and prev_signal == 1

            if hit_stop or hit_take or exit_signal:
                trade_value = shares * price
                trade_cost = trade_value * fee
                cash += trade_value - trade_cost
                trade_rows.append([idx, "SELL", price, shares, cash, cash, row["prob_up"], trade_cost])
                shares = 0.0
                entry_price = None
                bars_since_trade = 0

        equity = cash + shares * price
        equity_curve.append(equity)
        bars_since_trade += 1
        prev_signal = signal

    feat["equity"] = equity_curve
    feat["drawdown"] = feat["equity"] / feat["equity"].cummax() - 1
    return feat, pd.DataFrame(trade_rows, columns=["date", "action", "price", "shares", "cash", "equity", "prob_up", "fee"])

def max_drawdown(equity):
    roll = equity.cummax()
    dd = equity / roll - 1
    return float(dd.min()), dd

if run:
    if not tickers:
        st.error("Pick at least one asset.")
    else:
        results = {}
        latest_probs = {}
        sources = {}
        wf_summaries = {}
        trade_logs = {}

        for ticker in tickers:
            df, source = load_data(ticker, start, end, alpha_key if alpha_key else None)
            sources[ticker] = source
            if df.empty or len(df) < max(train_window + test_window + 20, 120):
                continue

            feat = make_features(df)
            cols = ["ret1", "ret5", "trend", "vol10"]
            X = feat[cols].dropna()
            y = feat.loc[X.index, "target"]

            wf_probs, wf_summary, train_scores, test_scores = walk_forward_probs(X, y, train_window, test_window)
            feat = feat.loc[wf_probs.index].copy()
            feat["prob_up"] = wf_probs

            latest_probs[ticker] = float(feat["prob_up"].iloc[-1])
            wf_summaries[ticker] = wf_summary

            strength = feat["prob_up"].rolling(5, min_periods=1).mean().fillna(0)
            feat, trade_log = simulate_asset(
                feat, threshold, cooldown, fee, starting_cash / max(1, len(tickers)), strength
            )

            results[ticker] = feat
            trade_logs[ticker] = trade_log

        if not results:
            st.error("Not enough data for the selected assets.")
        else:
            all_index = sorted(set().union(*[df.index for df in results.values()]))
            portfolio = pd.DataFrame(index=all_index)

            latest_probs_series = pd.Series(latest_probs)
            p = latest_probs_series.clip(lower=0.0)
            weights = p / p.sum() if p.sum() > 0 else pd.Series(np.ones(len(p)) / len(p), index=p.index)

            for ticker, feat in results.items():
                portfolio[ticker] = feat["equity"].reindex(all_index).ffill().fillna(starting_cash / len(results))

            portfolio["portfolio_equity"] = 0.0
            for ticker in results:
                portfolio["portfolio_equity"] += portfolio[ticker] * weights.get(ticker, 0)

            bh = pd.DataFrame(index=all_index)
            for ticker, feat in results.items():
                bh[ticker] = feat["Close"].reindex(all_index).ffill()
            bh["basket_bh"] = starting_cash * (bh.mean(axis=1) / bh.mean(axis=1).iloc[0])

            portfolio["drawdown"] = portfolio["portfolio_equity"] / portfolio["portfolio_equity"].cummax() - 1
            max_dd, _ = max_drawdown(portfolio["portfolio_equity"])

            ending_equity = float(portfolio["portfolio_equity"].iloc[-1])
            total_return = ending_equity / starting_cash - 1
            bh_end = float(bh["basket_bh"].iloc[-1])
            bh_return = bh_end / starting_cash - 1

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Ending equity", f"${ending_equity:,.2f}", f"{total_return:.2%}")
            c2.metric("Basket buy & hold", f"${bh_end:,.2f}", f"{bh_return:.2%}")
            c3.metric("Max drawdown", f"{max_dd:.2%}")
            c4.metric("Model refresh", datetime.now().strftime("%H:%M:%S"))

            st.write("Data source used:", ", ".join([f"{k}: {v}" for k, v in sources.items()]))

            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Portfolio", "Assets", "Signals", "Trades", "Walk-forward"])

            with tab1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=portfolio.index, y=portfolio["portfolio_equity"], mode="lines", name="Portfolio"))
                fig.add_trace(go.Scatter(x=bh.index, y=bh["basket_bh"], mode="lines", name="Basket buy & hold"))
                fig.update_layout(height=550, legend=dict(orientation="h"))
                st.plotly_chart(fig, use_container_width=True)

                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=portfolio.index, y=portfolio["drawdown"], mode="lines", name="Drawdown", line=dict(color="red")))
                fig2.update_layout(height=280, showlegend=False)
                st.plotly_chart(fig2, use_container_width=True)

            with tab2:
                for ticker, feat in results.items():
                    st.subheader(ticker)
                    st.line_chart(feat[["Close", "equity"]])

            with tab3:
                sig_df = pd.DataFrame([
                    {"ticker": t, "latest_prob": latest_probs[t], "source": sources.get(t, "")}
                    for t in results.keys()
                ]).sort_values("latest_prob", ascending=False)
                st.dataframe(sig_df)

            with tab4:
                combined = []
                for t, log in trade_logs.items():
                    if not log.empty:
                        temp = log.copy()
                        temp["ticker"] = t
                        combined.append(temp)
                if combined:
                    st.dataframe(pd.concat(combined).sort_values("date", ascending=False))
                else:
                    st.info("No trades were generated.")

            with tab5:
                for ticker, summary in wf_summaries.items():
                    st.subheader(ticker)
                    st.dataframe(summary)
else:
    st.info("Pick your assets, set your simulation values, then tap Run / Refresh.")
