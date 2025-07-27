from flask import Flask, jsonify, request
from flask_cors import CORS 
from stable_baselines3 import DQN
from stock_trading_env import StockTradingEnv  # Copy your class to a .py file
import yfinance as yf
import numpy as np
import pandas as pd

app = Flask(__name__)
CORS(app)  

# Load model once
model = DQN.load("dqn_trading_model")

def get_data(ticker="AAPL", start="2022-01-01", end="2023-01-01"):
    data = yf.download(ticker, start=start, end=end)
    return data[['Close']]

@app.route('/predict', methods=['GET'])
def predict():
    ticker = request.args.get('ticker', 'AAPL')
    data = get_data(ticker)
    if data.empty or len(data) < 10:
        return jsonify({"error": "Insufficient or empty stock data for ticker."}), 400

    env = StockTradingEnv(data)
    obs = env.reset()
    done = False
    history = []

    action_map = {0: "hold", 1: "buy", 2: "sell"}
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if env.current_step >= len(data):
            price = data['Close'].iloc[-1]
        else:
            price = data['Close'].iloc[env.current_step]

        net_worth = env.balance + env.shares_held * price
        history.append({
            "step": env.current_step,
            "price": float(price),
            "action": int(action),
            "action_name": action_map[int(action)],
            "net_worth": float(net_worth)
        })

    #  Add summary statistics
    initial_balance = env.initial_balance
    final_balance = history[-1]["net_worth"]
    profit = final_balance - initial_balance
    profit_percent = (profit / initial_balance) * 100

    return jsonify({
        "history": history,
        "summary": {
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "profit": profit,
            "profit_percent": profit_percent
        }
    })


if __name__ == '__main__':
    app.run(debug=True)
