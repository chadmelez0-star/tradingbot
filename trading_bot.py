# ==========================================
# BINANCE TRADING BOT - GÜVENLİ BAŞLANGIÇ
# ==========================================

import os
import time
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
from datetime import datetime
import requests

# ==========================================
# API AYARLARI - BUNLARI DEĞİŞTİRİN
# ==========================================

API_KEY = '6GzbcJqV9UGJhbAPvYYo2rwjB3LE2ogqAiQMs5iXu5J6rNHtVMNzuZkyTAd7DhQF'  # YENİ API KEY YAZIN
API_SECRET = 'lPcReuw8USNLuqh23A9qok9mgsVrhctyg47uVDhx7Hh51UL7n03fOt8OV4x0Qj69'  # YENİ SECRET YAZIN
TEST_MODE = True  # Test modu - Gerçek para kullanılmaz

# ==========================================
# RİSK YÖNETİMİ
# ==========================================

class RiskManager:
    def __init__(self):
        self.max_position_size = 20  # Her işlem için portföyün %20'si
        self.max_daily_trades = 5    # Günde max 5 işlem
        self.stop_loss_percent = 3   # %3 zararda sat
        self.take_profit_percent = 6  # %6 karda sat
        self.daily_loss_limit = 10   # Günlük %10 zararda dur
        
        self.today_trades = 0
        self.daily_pnl = 0
        
    def can_trade(self):
        if self.today_trades >= self.max_daily_trades:
            return False, "Günlük işlem limiti doldu"
        if self.daily_pnl <= -self.daily_loss_limit:
            return False, "Günlük zarar limiti aşıldı"
        return True, "OK"
    
    def calculate_position_size(self, total_balance, current_price):
        risk_amount = total_balance * (self.max_position_size / 100)
        quantity = risk_amount / current_price
        return round(quantity, 6)

# ==========================================
# TEKNİK ANALİZ
# ==========================================

class TechnicalAnalyzer:
    @staticmethod
    def calculate_rsi(prices, period=14):
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum()/period
        down = -seed[seed < 0].sum()/period
        rs = up/down if down != 0 else 0
        rsi = np.zeros_like(prices)
        rsi[:period] = 100. - 100./(1. + rs)
        
        for i in range(period, len(prices)):
            delta = deltas[i-1]
            upval = delta if delta > 0 else 0.
            downval = -delta if delta < 0 else 0.
            up = (up*(period-1) + upval)/period
            down = (down*(period-1) + downval)/period
            rs = up/down if down != 0 else 0
            rsi[i] = 100. - 100./(1. + rs)
        
        return rsi[-1]
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
        exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        histogram = macd - signal_line
        
        return {
            'macd': macd.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }
    
    @staticmethod
    def calculate_bollinger_bands(prices, period=20, std_dev=2):
        sma = pd.Series(prices).rolling(window=period).mean()
        std = pd.Series(prices).rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return {
            'upper': upper_band.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower_band.iloc[-1],
            'current_price': prices[-1]
        }
    
    @staticmethod
    def generate_signal(rsi, macd, bb, current_price):
        signals = []
        
        if rsi < 30:
            signals.append('BUY')
        elif rsi > 70:
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            signals.append('BUY')
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        if current_price <= bb['lower']:
            signals.append('BUY')
        elif current_price >= bb['upper']:
            signals.append('SELL')
        else:
            signals.append('HOLD')
        
        buy_count = signals.count('BUY')
        sell_count = signals.count('SELL')
        
        if buy_count >= 2:
            return 'BUY'
        elif sell_count >= 2:
            return 'SELL'
        return 'HOLD'

# ==========================================
# ANA BOT
# ==========================================

class BinanceTradingBot:
    def __init__(self):
        self.client = Client(API_KEY, API_SECRET, testnet=TEST_MODE)
        self.risk_manager = RiskManager()
        self.analyzer = TechnicalAnalyzer()
        self.symbol = 'BTCUSDT'
        self.timeframe = Client.KLINE_INTERVAL_15MINUTE
        self.last_signal = None
        self.position = None
        self.entry_price = 0
        self.current_price = 0
        
    def send_notification(self, message):
        print(f"[BİLDİRİM] {message}")
        
    def get_historical_data(self, limit=50):
        try:
            klines = self.client.get_klines(
                symbol=self.symbol,
                interval=self.timeframe,
                limit=limit
            )
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            df['close'] = df['close'].astype(float)
            return df['close'].values
        except Exception as e:
            self.send_notification(f"❌ Veri hatası: {e}")
            return None
    
    def get_account_balance(self):
        try:
            account = self.client.get_account()
            usdt_balance = next((float(b['free']) for b in account['balances'] 
                               if b['asset'] == 'USDT'), 0)
            btc_balance = next((float(b['free']) for b in account['balances'] 
                              if b['asset'] == 'BTC'), 0)
            return usdt_balance, btc_balance
        except Exception as e:
            self.send_notification(f"❌ Bakiye hatası: {e}")
            return 0, 0
    
    def place_buy_order(self, quantity):
        try:
            if TEST_MODE:
                self.send_notification(f"🧪 TEST: {quantity} BTC alımı simüle edildi")
                return True
            
            order = self.client.order_market_buy(
                symbol=self.symbol,
                quantity=quantity
            )
            self.send_notification(f"✅ ALIM: {quantity} BTC")
            return True
        except Exception as e:
            self.send_notification(f"❌ Alım hatası: {e}")
            return False
    
    def place_sell_order(self, quantity):
        try:
            if TEST_MODE:
                self.send_notification(f"🧪 TEST: {quantity} BTC satımı simüle edildi")
                return True
            
            order = self.client.order_market_sell(
                symbol=self.symbol,
                quantity=quantity
            )
            self.send_notification(f"✅ SATIM: {quantity} BTC")
            return True
        except Exception as e:
            self.send_notification(f"❌ Satım hatası: {e}")
            return False
    
    def check_stop_loss_take_profit(self):
        if not self.position or self.entry_price == 0:
            return False
        
        change_percent = ((self.current_price - self.entry_price) / self.entry_price) * 100
        
        if self.position == 'LONG':
            if change_percent <= -self.risk_manager.stop_loss_percent:
                self.send_notification(f"🛑 STOP LOSS: %{change_percent:.2f}")
                return True
            elif change_percent >= self.risk_manager.take_profit_percent:
                self.send_notification(f"🎯 TAKE PROFIT: %{change_percent:.2f}")
                return True
        return False
    
    def run_analysis(self):
        print("📊 Analiz başlıyor...")
        prices = self.get_historical_data()
        if prices is None:
            print("❌ Fiyat verisi alınamadı!")
            return
        if len(prices) < 30:
            print(f"❌ Yetersiz veri: {len(prices)} mum")
            return
        print(f"✅ {len(prices)} fiyat verisi alındı")
        
        self.current_price = prices[-1]
        
        rsi = self.analyzer.calculate_rsi(prices)
        macd = self.analyzer.calculate_macd(prices)
        bb = self.analyzer.calculate_bollinger_bands(prices)
        
        signal = self.analyzer.generate_signal(rsi, macd, bb, self.current_price)
        
        usdt_balance, btc_balance = self.get_account_balance()
        total_balance = usdt_balance + (btc_balance * self.current_price)
        
        status_msg = f"""
{'='*50}
🤖 BOT DURUM RAPORU - {datetime.now().strftime('%H:%M:%S')}
{'='*50}
💰 Toplam Bakiye: ${total_balance:.2f}
   ├─ USDT: ${usdt_balance:.2f}
   └─ BTC: {btc_balance:.6f} (${btc_balance * self.current_price:.2f})

📈 {self.symbol}: ${self.current_price:.2f}

📊 İNDİKATÖRLER:
   ├─ RSI: {rsi:.2f} ({'Düşük(Al)' if rsi < 30 else 'Yüksek(Sat)' if rsi > 70 else 'Normal'})
   ├─ MACD: {macd['histogram']:.4f} ({'Yükseliş' if macd['histogram'] > 0 else 'Düşüş'})
   └─ Bollinger: Alt ${bb['lower']:.2f} | Üst ${bb['upper']:.2f}

🎯 SİNYAL: {signal} | 📍 Pozisyon: {self.position or 'Yok'}
📊 İşlem: {self.risk_manager.today_trades}/{self.risk_manager.max_daily_trades}
{'='*50}
"""
        print(status_msg)
        
        can_trade, msg = self.risk_manager.can_trade()
        if not can_trade:
            print(f"⚠️ {msg}")
            return
        
        if self.check_stop_loss_take_profit():
            quantity = self.risk_manager.calculate_position_size(total_balance, self.current_price)
            if btc_balance >= quantity * 0.95:
                self.place_sell_order(quantity)
                self.position = None
                self.risk_manager.today_trades += 1
            return
        
        if signal == 'BUY' and self.position != 'LONG' and usdt_balance > 10:
            quantity = self.risk_manager.calculate_position_size(usdt_balance, self.current_price)
            if self.place_buy_order(quantity):
                self.position = 'LONG'
                self.entry_price = self.current_price
                self.risk_manager.today_trades += 1
                self.send_notification("🚀 ALIM SİNYALİ - Pozisyon açıldı")
                
        elif signal == 'SELL' and self.position == 'LONG' and btc_balance > 0.0001:
            quantity = min(btc_balance, self.risk_manager.calculate_position_size(total_balance, self.current_price))
            if self.place_sell_order(quantity):
                pnl = ((self.current_price - self.entry_price) / self.entry_price) * 100
                self.risk_manager.daily_pnl += pnl
                self.position = None
                self.risk_manager.today_trades += 1
                emoji = "📈" if pnl > 0 else "📉"
                self.send_notification(f"{emoji} SATIM - P&L: %{pnl:.2f}")
    
    def start(self):
        mode = "TESTNET" if TEST_MODE else "GERÇEK"
        self.send_notification(f"🤖 BOT BAŞLATILDI - {mode} MODU")
        
        print("Bot çalışıyor... Durdurmak için Ctrl+C basın")
        
        try:
            while True:
                self.run_analysis()
                print(f"Sonraki kontrol: 5 dakika sonra...")
                print(f"{'='*50}\\n")
                time.sleep(300)  # 5 dakika
        except KeyboardInterrupt:
            self.send_notification("🛑 Bot durduruldu")
            print("\\nBot durduruldu.")

# ==========================================
# BAŞLAT
# ==========================================

if __name__ == "__main__":
    bot = BinanceTradingBot()
    bot.start()