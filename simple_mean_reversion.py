#!/usr/bin/env python3
"""
Test Strategy for Poloniex BTC Trading
Checks BTC price and places buy order if price < $100k USD
"""

##Test Paper Trade


import os
import sys
import time
import json
import pandas as pd
import numpy as np
from decimal import Decimal

# Add the parent directory to the path to import our modules
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "../../"))
sys.path.insert(0, PROJECT_ROOT)

from constants import set_constants, get_constants
from exchange_api_spot.user import get_client_exchange
from utils import (
    get_line_number,
    update_key_and_insert_error_log,
    generate_random_string,
    get_precision_from_real_number,
    get_arg
)
## General log ##
from logger import logger_access, logger_error
from logger import setup_logger_global


def bb(df, period=20, num_std=2):
    """
    Calculate Bollinger Bands
    Returns: (upper_band, lower_band, middle_band)
    """
    close = df['close']
    rolling_mean = close.rolling(window=period).mean()
    rolling_std = close.rolling(window=period).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band, rolling_mean


def rsi(df, period=14):
    """
    Calculate RSI
    Returns: Series of RSI values
    """
    close = df['close']
    delta = close.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)

    roll_up = pd.Series(up, index=close.index).rolling(window=period).mean()
    roll_down = pd.Series(down, index=close.index).rolling(window=period).mean()

    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi

class MeanReversionBBRSIStrategy:
    """
    Mean Reversion Strategy using Bollinger Bands + RSI:
    - Buy if price < Lower Band AND RSI(5) < 30
    - Sell if price > Upper Band AND RSI(5) > 70
    """

    def __init__(self, api_key="", secret_key="", passphrase="", session_key=""):
        # Identifiers
        self.run_key = generate_random_string()
        self.session_key = session_key

        # Strategy params
        self.symbol = "BTC"
        self.quote = "USDT"
        self.trade_value = 200              # Trade value in quote currency
        self.bb_period = 20
        self.bb_std = 2
        self.rsi_period = 5
        self.interval = "15m"
        self.limit = self.bb_period * 2
        self.sleep = 60

        # Position tracking
        self.lastest_pos = 0    # 1 = long, -1 = short, 0 = flat

        # Config logger
        self.exchange = "binance"
        self.class_name = self.__class__.__name__
        strategy_log_name = f'{self.symbol}_{self.exchange}_{self.class_name}'
        self.logger_strategy = setup_logger_global(
            strategy_log_name, strategy_log_name + '.log'
        )
        self.stop_flag = False

        try:
            account_info = {
                "api_key": api_key,
                "secret_key": secret_key,
                "passphrase": passphrase,
            }

            self.client = get_client_exchange(
                exchange_name="binance",
                acc_info=account_info,
                symbol=self.symbol,
                quote=self.quote,
                session_key=session_key,
            )

            self.logger_strategy.info(
                f"client initialized successfully for {self.symbol}/{self.quote}"
            )
        except Exception as e:
            self.logger_strategy.error(f"❌ Failed to initialize client: {e}")
            raise

    def get_current_price(self):
        """Get current price of the symbol."""
        try:
            price_data = self.client.get_price()
            if price_data and 'price' in price_data:
                return float(price_data['price'])
            else:
                self.logger_strategy.error("Invalid price response")
                return None
        except Exception as e:
            self.logger_strategy.error(f"Error getting price: {e}")
            return None

    def get_candles(self):
        """Fetch OHLCV candles and return as DataFrame."""
        candles_response = self.client.get_candles(
            base=self.symbol,
            quote=self.quote,
            interval=self.interval,
            limit=self.limit
        )

        candles = candles_response.get("candle", [])
        df = pd.DataFrame(
            candles,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades',
                    'taker_base', 'taker_quote', 'ignore']
        )
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        return df

    def get_signal(self):
        """Return signal based on BB + RSI."""
        df = self.get_candles()
        if df.empty or len(df) < max(self.bb_period, self.rsi_period):
            return 0

        # Get BB + RSI
        upper_band, lower_band, _ = bb(df, period=self.bb_period, num_std=self.bb_std)
        rsi_series = rsi(df, period=self.rsi_period)

        curr_price = self.get_current_price()
        curr_upper = upper_band.iloc[-1]
        curr_lower = lower_band.iloc[-1]
        curr_rsi = rsi_series.iloc[-1]

        self.logger_strategy.info(
            f"Price={curr_price}, BB=({curr_lower:.2f}, {curr_upper:.2f}), RSI={curr_rsi:.2f}"
        )

        if curr_price is None or np.isnan(curr_price) or np.isnan(curr_rsi):
            return 0
        
        alpha_factor = (curr_rsi - 50) / 10_000
        curr_lower = curr_lower * (1 + alpha_factor)
        curr_upper = curr_upper * (1 + alpha_factor)

        # Entry / Exit rules
        if curr_price < curr_lower and curr_rsi < 30:
            return 1  # BUY
        elif curr_price > curr_upper and curr_rsi > 70:
            return -1  # SELL
        return 0

    def get_quantity(self, price):
        return self.trade_value / price if price else 0

    def place_order(self, side, quantity, price=None, order_type="MARKET"):
        try:
            if side == "BUY":
                balance = self.check_account_balance(is_base=False)
                if balance < self.trade_value:
                    self.logger_strategy.warning(f"Insufficient {self.quote} balance")
                    return False
            elif side == "SELL":
                balance = self.check_account_balance(is_base=True)
                if balance * price < self.trade_value:
                    self.logger_strategy.warning(f"Insufficient {self.symbol} balance")
                    return False

            order_result = self.client.place_order(
                side_order=side,
                quantity=quantity,
                order_type=order_type,
                force='GTC'
            )

            if order_result and order_result.get('code') == 0:
                order_id = order_result.get('data', {}).get('orderId', 'N/A')
                self.lastest_pos = 1 if side == "BUY" else -1
                self.logger_strategy.info(
                    f"✅ {side} order placed successfully, ID={order_id}, Qty={quantity:.6f}"
                )
                return True
            else:
                self.logger_strategy.error(f"❌ Failed to place order: {order_result}")
                return False

        except Exception as e:
            self.logger_strategy.error(f"Error placing order: {e}")
            return False

    def logic(self):
        signal = self.get_signal()
        self.logger_strategy.info(f"Signal: {signal}")

        if signal == 0 or signal == self.lastest_pos:
            return False

        price = self.get_current_price()
        quantity = self.get_quantity(price)

        if signal == 1:
            return self.place_order("BUY", quantity, price)
        elif signal == -1:
            return self.place_order("SELL", quantity, price)
        return False

    def run_strategy(self):
        self.logger_strategy.info("🚀 Starting BB + RSI Mean Reversion Strategy...")
        try:
            while not self.stop_flag:
                executed = self.logic()
                if executed:
                    self.logger_strategy.info("Trade executed successfully")
                else:
                    self.logger_strategy.info("No trade executed this cycle")
                time.sleep(self.sleep)
        except Exception as e:
            self.logger_strategy.error(f"Strategy error: {e} _ {e.__traceback__.tb_lineno}")
            return False

def main():
    """
    Main function to run the strategy
    """
    params = get_constants()
    SESSION_ID     = params.get("SESSION_ID", "")
    EXCHANGE       = params.get("EXCHANGE", "")
    API_KEY        = params.get("API_KEY", "")
    SECRET_KEY     = params.get("SECRET_KEY", "")
    PASSPHRASE     = params.get("PASSPHRASE", "")
    STRATEGY_NAME  = params.get("STRATEGY_NAME", "")
    PAPER_MODE     = params.get("PAPER_MODE", True)     
    
    if not API_KEY or not SECRET_KEY:
        logger_access.info("❌ API credentials are required")
        return
    
    if not SESSION_ID:
        logger_access.info("❌ Session key is required")
        return
    
    logger_access.info("✅ Environment variables loaded successfully")
    logger_access.info(f"🔑 Session Key: {SESSION_ID}")
    logger_access.info("BTC Test Strategy starting...")

    try:
        # Initialize strategy
        strategy = SimpleStrategy(
            api_key=API_KEY,
            secret_key=SECRET_KEY,
            passphrase=PASSPHRASE,
            session_key=SESSION_ID,
        )
        
        logger_access.info("Test Strategy initialized successfully")

        # Run the strategy loop (internal while True)
        strategy.run_strategy()

    except KeyboardInterrupt:
        logger_access.info("\n🛑 Strategy stopped by user")
        
    except Exception as e:
        logger_access.error(f"❌ Fatal error: {e}")
        update_key_and_insert_error_log(
            generate_random_string(),
            "BTC",
            get_line_number(),
            "BINANCE",
            "test-strategy.py",
            f"Fatal error: {e}"
        )

if __name__ == "__main__":
    main()
