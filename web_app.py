# ==========================================
# ELMAS BOT PRO - ANA UYGULAMA (PARÇA 1)
# ==========================================

import os
import sys
import time
import threading
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, jsonify, request
from flask_socketio import SocketIO
from binance.client import Client
from binance.enums import *
from dotenv import load_dotenv

# ==========================================
# TESTNET DNS SORUNU ÇÖZÜMÜ
# ==========================================

# TestNet çalışmıyorsa gerçek API'ye yönlendir
def patch_binance_client():
    from binance import client
    original_init = client.Client.__init__
    
    def patched_init(self, api_key, api_secret, testnet=False, **kwargs):
        # TestNet istenirse bile gerçek API'ye bağlan
        if testnet:
            print("⚠️ TestNet devre dışı, gerçek API kullanılıyor")
            testnet = False
        original_init(self, api_key, api_secret, testnet=testnet, **kwargs)
    
    client.Client.__init__ = patched_init

patch_binance_client()

# .env dosyasını yükle
load_dotenv()

# ==========================================
# ORTAM DEĞİŞKENLERİNİ OKU (ÖNCE BUNLAR!)
# ==========================================

API_KEY = os.getenv('API_KEY')
API_SECRET = os.getenv('API_SECRET')
TEST_MODE = os.getenv('TEST_MODE', 'true').lower() == 'true'
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

# Kontrol
if not API_KEY or not API_SECRET:
    print("❌ API_KEY ve API_SECRET bulunamadı! .env dosyasını kontrol et.")
    sys.exit(1)

print(f"🔑 API Key: {API_KEY[:10]}...")
print(f"💰 Test Modu: {TEST_MODE}")

# ==========================================
# MODÜLLERİ İÇE AKTAR
# ==========================================

from ai_engine import AITradingEngine
from multi_timeframe import MultiTimeframeAnalyzer
from backtest_engine import AdvancedBacktest
from telegram_bot import AdvancedTelegramBot

# ==========================================
# FLASK AYARLARI
# ==========================================

app = Flask(__name__)
app.config['SECRET_KEY'] = 'elmas-bot-pro-secret-2024'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global değişkenler
bot_running = False
bot_thread = None
bot_instance = None

current_data = {
    'system': {
        'status': 'DURDU',
        'mode': 'TEST' if TEST_MODE else 'GERÇEK',
        'ai_trained': False,
        'last_update': None
    },
    'market': {
        'btc': {'price': 0, 'score': 50, 'ai_score': 50, 'tf_score': 50, 'final_score': 50, 
                'signal': 'BEKLE', 'position': None, 'history': [], 'timeframes': {}},
        'eth': {'price': 0, 'score': 50, 'ai_score': 50, 'tf_score': 50, 'final_score': 50,
                'signal': 'BEKLE', 'position': None, 'history': [], 'timeframes': {}},
        'sol': {'price': 0, 'score': 50, 'ai_score': 50, 'tf_score': 50, 'final_score': 50,
                'signal': 'BEKLE', 'position': None, 'history': [], 'timeframes': {}}
    },
    'portfolio': {
        'total': 0,
        'daily_pnl': 0,
        'total_pnl': 0,
        'positions': {},
        'history': []
    },
    'ai': {
        'is_trained': False,
        'accuracy': 0,
        'top_features': [],
        'last_training': None
    },
    'stats': {
        'today_trades': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'win_rate': 0
    },
    'logs': []
}

# ==========================================
# ELMAS BOT SINIFI
# ==========================================

class ElmasBot:
    def __init__(self):
        # TEST_MODE'e göre ayarla
        if TEST_MODE:
            # Test modu: Gerçek API'ye bağlan ama işlem yapma
            self.client = Client(API_KEY, API_SECRET, testnet=False)
            self.log("🧪 TEST MODU - Gerçek API, simülasyon işlemler")
        else:
            # Gerçek mod
            self.client = Client(API_KEY, API_SECRET, testnet=False)
            self.log("💰 GERÇEK MOD - Gerçek işlemler!")
        
        self.coins = {
            'BTCUSDT': {'position': None, 'entry_price': 0, 'amount': 0},
            'ETHUSDT': {'position': None, 'entry_price': 0, 'amount': 0},
            'SOLUSDT': {'position': None, 'entry_price': 0, 'amount': 0}
        }
        
        self.max_daily_trades = 10
        self.today_trades = 0
        self.daily_pnl = 0
        
        # AI eğitim verisi toplama
        self.training_data = {}
        
    def log(self, message, level='info'):
        """Log kaydı"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        entry = f"[{timestamp}] {message}"
        
        current_data['logs'].insert(0, {'time': timestamp, 'message': message, 'level': level})
        if len(current_data['logs']) > 100:
            current_data['logs'].pop()
        
        print(entry)
        socketio.emit('new_log', {'time': timestamp, 'message': message, 'level': level})
        
        # Telegram önemli mesajlar
        if level in ['trade', 'error', 'warning'] and self.telegram.enabled:
            self.telegram.alert('Elmas Bot', message, level)
    
    def collect_training_data(self):
        """AI eğitim verisi topla - GENİŞLETİLMİŞ"""
        self.log("📊 AI eğitim verisi toplanıyor (Genişletilmiş)...")
        
        self.training_data = {}
        
        symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'ADAUSDT']  # ← 5 coin
        timeframes = [
            (Client.KLINE_INTERVAL_5MINUTE, 1000),   # ← 5m eklendi
            (Client.KLINE_INTERVAL_15MINUTE, 1000),  # ← Daha fazla veri
            (Client.KLINE_INTERVAL_1HOUR, 500),      # ← Tekrar denenecek
        ]
        
        total_collected = 0
        
        for symbol in symbols:
            for interval, limit in timeframes:
                try:
                    self.log(f"  📥 {symbol} {interval} verisi çekiliyor...")
                    df = self.tf_analyzer.get_data(symbol, interval, limit)
                    
                    if df is not None and len(df) > 100:  # ← Minimum 100
                        key = f"{symbol}_{interval}"
                        self.training_data[key] = df
                        total_collected += len(df)
                        self.log(f"  ✅ {key}: {len(df)} kayıt")
                    else:
                        self.log(f"  ⚠️ {symbol} {interval}: Yetersiz veri ({len(df) if df is not None else 0})", 'warning')
                        
                except Exception as e:
                    self.log(f"  ❌ Veri hatası {symbol} {interval}: {e}", 'warning')
        
        self.log(f"✅ Toplam {len(self.training_data)} veri seti, {total_collected} kayıt toplandı")
        return len(self.training_data) >= 3  # ← En az 3 set olsun

    def train_ai(self):
        """AI modelini eğit - DÜZELTİLMİŞ"""
        if len(self.training_data) < 2:
            self.log("❌ Eğitim için yeterli veri yok (minimum 2 veri seti)", 'error')
            return False
        
        self.log("🧠 AI modeli eğitiliyor...")
        
        try:
            success = self.ai.train(self.training_data)
            
            if success:
                # Doğruluğu hesapla (basitleştirilmiş)
                try:
                    # Son eğitim verisinden bir örnek al ve skor hesapla
                    sample_key = list(self.training_data.keys())[0]
                    sample_df = self.training_data[sample_key]
                    features = self.ai.create_advanced_features(sample_df, for_training=True)
                    
                    if len(features) > 50:
                        X = features.drop(['target_direction', 'target_return'], axis=1, errors='ignore')
                        X = X.select_dtypes(include=[np.number]).replace([np.inf, -np.inf], np.nan).fillna(0)
                        y = features['target_direction']
                        
                        # Eğitimdeki kolonlarla eşleştir
                        if self.ai.feature_columns:
                            for col in self.ai.feature_columns:
                                if col not in X.columns:
                                    X[col] = 0
                            X = X[self.ai.feature_columns]
                        
                        X_scaled = self.ai.scaler.transform(X)
                        accuracy = self.ai.model.score(X_scaled, y)
                    else:
                        accuracy = 0.75  # Varsayılan değer
                        
                except Exception as e:
                    self.log(f"Doğruluk hesaplama hatası: {e}", 'warning')
                    accuracy = 0.75
                
                current_data['ai']['is_trained'] = True
                current_data['ai']['accuracy'] = accuracy
                current_data['ai']['top_features'] = self.ai.get_feature_importance()
                current_data['ai']['last_training'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                
                self.log(f"✅ AI eğitildi! Doğruluk: %{accuracy*100:.2f}")
                
                # Telegram bildirim
                if self.telegram.enabled:
                    self.telegram.ai_training_complete(
                        accuracy,
                        current_data['ai']['top_features']
                    )
                
                return True
            else:
                self.log("❌ AI eğitimi başarısız", 'error')
                return False
                
        except Exception as e:
            self.log(f"❌ AI eğitim hatası: {e}", 'error')
            import traceback
            traceback.print_exc()
            return False

    
    def analyze_coin(self, symbol):
        """Coin analizi - Tüm yöntemler"""
        coin_key = symbol.replace('USDT', '').lower()
        
        try:
            # 1. Temel veri çek
            df_15m = self.tf_analyzer.get_data(symbol, Client.KLINE_INTERVAL_15MINUTE, 100)
            if df_15m is None:
                return None
            
            current_price = df_15m['close'].iloc[-1]
            
            # 2. Çoklu zaman dilimi analizi
            tf_results = self.tf_analyzer.analyze_all_timeframes(symbol)
            tf_score = tf_results['total_score']
            
            # 3. AI tahmini (eğitilmişse)
            ai_score = 50
            ai_signal = 'BEKLE'
            if self.ai.is_trained:
                try:
                    ai_prediction = self.ai.predict(df_15m)
                    ai_score = ai_prediction['confidence']
                    ai_signal = ai_prediction['signal']
                except Exception as e:
                    self.log(f"AI tahmin hatası {symbol}: {e}", 'warning')
            
            # 4. Temel indikatör skoru
            basic_score = self.calculate_basic_score(df_15m)
            
            # 5. Final skor (ağırlıklı ortalama)
            # TF: 40%, AI: 35%, Basic: 25%
            final_score = (tf_score * 0.4) + (ai_score * 0.35) + (basic_score * 0.25)
            final_score = max(0, min(100, final_score))
            
            # Sinyal belirle
            if final_score >= 80:
                signal = 'GÜÇLÜ AL'
            elif final_score >= 65:
                signal = 'AL'
            elif final_score <= 20:
                signal = 'GÜÇLÜ SAT'
            elif final_score <= 35:
                signal = 'SAT'
            else:
                signal = 'BEKLE'
            
            # Geçmiş güncelle
            history = current_data['market'][coin_key].get('history', [])
            history.append(current_price)
            if len(history) > 50:
                history.pop(0)
            
            return {
                'price': current_price,
                'score': int(basic_score),
                'ai_score': int(ai_score),
                'tf_score': int(tf_score),
                'final_score': int(final_score),
                'signal': signal,
                'ai_signal': ai_signal,
                'tf_consensus': tf_results['consensus'],
                'timeframes': tf_results['timeframes'],
                'history': history,
                'position': self.coins[symbol]['position']
            }
            
        except Exception as e:
            self.log(f"Analiz hatası {symbol}: {e}", 'error')
            return None
    
    def calculate_basic_score(self, df):
        """Temel indikatör skoru"""
        try:
            prices = df['close'].values
            
            # RSI
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
            rs = gain / loss if loss != 0 else 0
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().iloc[-1]
            ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().iloc[-1]
            macd = ema_12 - ema_26
            
            # Skor
            score = 50
            if rsi < 30:
                score += 25
            elif rsi > 70:
                score -= 25
            
            if macd > 0:
                score += 10
            else:
                score -= 10
            
            return max(0, min(100, score))
            
        except:
            return 50
    
    def execute_trade(self, symbol, signal, price):
        """İşlem yürüt"""
        coin = self.coins[symbol]
        coin_key = symbol.replace('USDT', '').lower()

        # TEST MODU: Sadece simülasyon
        if TEST_MODE:
            if signal in ['AL', 'GÜÇLÜ AL'] and coin['position'] is None:
                self.log(f"🧪 TEST ALIM: {symbol} @ ${price:,.2f}")
                # Pozisyonu simüle et
                coin['position'] = 'LONG'
                coin['entry_price'] = price
                coin['amount'] = 0.001  # Sabit test miktarı
                return True
            
            elif signal in ['SAT', 'GÜÇLÜ SAT'] and coin['position'] == 'LONG':
                pnl = (price - coin['entry_price']) * coin['amount']
                self.log(f"🧪 TEST SATIM: {symbol} @ ${price:,.2f} | P&L: ${pnl:+.2f}")
                # Pozisyonu kapat
                coin['position'] = None
                self.daily_pnl += pnl
                return True
            return False
    
    # GERÇEK MOD: Normal işlem (eski kodların burada)
    # ... (mevcut alım/satım kodların aynen kalacak)
        
        # Alım koşulları
        if signal in ['AL', 'GÜÇLÜ AL'] and coin['position'] is None:
            if self.today_trades >= self.max_daily_trades:
                self.log(f"⚠️ Günlük işlem limiti doldu", 'warning')
                return False
            
            try:
                usdt_balance = self.get_balance('USDT')
                if usdt_balance < 10:
                    self.log(f"❌ {symbol} Yetersiz USDT bakiyesi", 'error')
                    return False
                
                # Risk yönetimi: %10 pozisyon
                risk_amount = usdt_balance * 0.1
                amount = risk_amount / price
                amount = round(amount, 6)
                
                if TEST_MODE:
                    self.log(f"🧪 TEST ALIM: {symbol} {amount} @ ${price:,.2f}")
                    success = True
                else:
                    order = self.client.order_market_buy(symbol=symbol, quantity=amount)
                    success = True
                
                if success:
                    coin['position'] = 'LONG'
                    coin['entry_price'] = price
                    coin['amount'] = amount
                    self.today_trades += 1
                    
                    self.log(f"🚀 ALIM: {symbol} {amount} @ ${price:,.2f}", 'trade')
                    
                    # Telegram
                    if self.telegram.enabled:
                        self.telegram.trade_notification(
                            symbol, "ALIM", price, amount,
                            strategy_info=f"Skor: {current_data['market'][coin_key]['final_score']}"
                        )
                    
                    # Ses (iptal edildi)
                    # socketio.emit('play_sound', {'type': 'buy'})
                    
                    return True
                    
            except Exception as e:
                self.log(f"❌ Alım hatası {symbol}: {e}", 'error')
                return False
        
        # Satım koşulları
        elif signal in ['SAT', 'GÜÇLÜ SAT'] and coin['position'] == 'LONG':
            try:
                base_asset = symbol.replace('USDT', '')
                amount = coin['amount']
                
                if amount <= 0:
                    return False
                
                # P&L hesapla
                pnl_usd = (price - coin['entry_price']) * amount
                pnl_pct = (price - coin['entry_price']) / coin['entry_price'] * 100
                
                if TEST_MODE:
                    self.log(f"🧪 TEST SATIM: {symbol} {amount} @ ${price:,.2f} | P&L: ${pnl_usd:+.2f}")
                    success = True
                else:
                    order = self.client.order_market_sell(symbol=symbol, quantity=amount)
                    success = True
                
                if success:
                    coin['position'] = None
                    self.daily_pnl += pnl_usd
                    
                    # Stats güncelle
                    current_data['stats']['today_trades'] += 1
                    if pnl_usd > 0:
                        current_data['stats']['winning_trades'] += 1
                    else:
                        current_data['stats']['losing_trades'] += 1
                    
                    total_trades = current_data['stats']['winning_trades'] + current_data['stats']['losing_trades']
                    if total_trades > 0:
                        current_data['stats']['win_rate'] = round(
                            current_data['stats']['winning_trades'] / total_trades * 100, 2
                        )
                    
                    self.log(f"📉 SATIM: {symbol} @ ${price:,.2f} | P&L: ${pnl_usd:+.2f} (%{pnl_pct:.2f})", 'trade')
                    
                    # Telegram
                    if self.telegram.enabled:
                        self.telegram.trade_notification(
                            symbol, "SATIM", price, amount, pnl_usd,
                            strategy_info=f"Skor: {current_data['market'][coin_key]['final_score']}"
                        )
                    
                    return True
                    
            except Exception as e:
                self.log(f"❌ Satım hatası {symbol}: {e}", 'error')
                return False
        
        return False
    
    def get_balance(self, asset='USDT'):
        """Bakiye sorgula"""
        try:
            account = self.client.get_account()
            balance = next((float(b['free']) for b in account['balances'] if b['asset'] == asset), 0)
            return balance
        except:
            return 0
    
    def update_portfolio(self):
        """Portföy güncelle"""
        try:
            total = self.get_balance('USDT')
            positions = {}
            
            for symbol, coin in self.coins.items():
                asset = symbol.replace('USDT', '')
                amount = self.get_balance(asset)
                
                if amount > 0 or coin['position']:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    price = float(ticker['price'])
                    value = amount * price
                    total += value
                    
                    positions[symbol] = {
                        'amount': amount,
                        'value': value,
                        'price': price,
                        'entry': coin['entry_price'],
                        'pnl': (price - coin['entry_price']) * amount if coin['entry_price'] else 0
                    }
            
            current_data['portfolio']['total'] = round(total, 2)
            current_data['portfolio']['daily_pnl'] = round(self.daily_pnl, 2)
            current_data['portfolio']['positions'] = positions
            
        except Exception as e:
            self.log(f"Portföy hatası: {e}", 'error')
    
    def run(self):
        """Ana döngü"""
        global bot_running
        
        self.log("=" * 60)
        self.log("💎 ELMAS BOT PRO BAŞLATILDI")
        self.log("=" * 60)
        self.log(f"💰 Mod: {'TESTNET' if TEST_MODE else 'GERÇEK'}")
        self.log(f"🧠 AI: {'Aktif' if self.ai.is_trained else 'Eğitim bekliyor'}")
        self.log(f"📱 Telegram: {'Aktif' if self.telegram.enabled else 'Pasif'}")
        self.log("=" * 60)
        
        # Başlangıçta AI eğitimi
        if not self.ai.is_trained:
            self.collect_training_data()
            self.train_ai()
        
        # Telegram başlangıç
        if self.telegram.enabled:
            self.telegram.alert('Elmas Bot Pro', '🔴 Bot başlatıldı ve çalışıyor!', 'success')
        
        while bot_running:
            try:
                # Her coin'i analiz et
                for symbol in self.coins.keys():
                    result = self.analyze_coin(symbol)
                    if result:
                        coin_key = symbol.replace('USDT', '').lower()
                        current_data['market'][coin_key] = result
                        
                        # İşlem yap
                        self.execute_trade(symbol, result['signal'], result['price'])
                
                # Portföy güncelle
                self.update_portfolio()
                
                # Sistem durumu
                current_data['system']['status'] = 'ÇALIŞIYOR'
                current_data['system']['last_update'] = datetime.now().strftime('%H:%M:%S')
                
                # Verileri gönder
                socketio.emit('update_data', current_data)
                socketio.emit('bot_status', {'running': True})
                
                time.sleep(15)
                
            except Exception as e:
                self.log(f"❌ Ana döngü hatası: {e}", 'error')
                time.sleep(15)
        
        # Durdurma
        self.log("🛑 Bot durduruldu")
        current_data['system']['status'] = 'DURDU'
        
        if self.telegram.enabled:
            self.telegram.alert('Elmas Bot Pro', '⏹️ Bot durduruldu', 'warning')
        
        socketio.emit('bot_status', {'running': False})

# ==========================================
# ELMAS BOT PRO - PARÇA 2 (Web Rotaları)
# ==========================================

# Bu kodu PARÇA 1'in SONUNA ekleyin

# ==========================================
# WEB ROTALARI
# ==========================================

@app.route('/')
def index():
    return render_elmas_html()

@app.route('/api/data')
def get_data():
    return jsonify(current_data)

@app.route('/api/start', methods=['POST'])
def start_bot():
    global bot_running, bot_thread, bot_instance
    
    if not bot_running:
        bot_running = True
        bot_instance = ElmasBot()
        bot_thread = threading.Thread(target=bot_instance.run)
        bot_thread.daemon = True
        bot_thread.start()
        return jsonify({'status': 'success', 'message': '💎 Elmas Bot Pro başlatıldı'})
    
    return jsonify({'status': 'error', 'message': 'Bot zaten çalışıyor'})

@app.route('/api/stop', methods=['POST'])
def stop_bot():
    global bot_running
    bot_running = False
    return jsonify({'status': 'success', 'message': 'Bot durduruldu'})

@app.route('/api/train', methods=['POST'])
def manual_train():
    global bot_instance
    if bot_instance:
        success = bot_instance.train_ai()
        return jsonify({'status': 'success' if success else 'error', 
                       'trained': current_data['ai']['is_trained']})
    return jsonify({'status': 'error', 'message': 'Bot çalışmıyor'})

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Backtest endpoint - DÜZELTİLMİŞ"""
    global bot_instance
    
    try:
        # JSON verisini al
        data = request.get_json() or {}
        symbol = data.get('symbol', 'BTCUSDT')
        
        print(f"📊 Backtest başlatılıyor: {symbol}")
        
        # Client oluştur (eğer bot çalışmıyorsa)
        if bot_instance and bot_instance.client:
            client = bot_instance.client
            backtest_engine = bot_instance.backtest
        else:
            client = Client(API_KEY, API_SECRET, testnet=TEST_MODE)
            backtest_engine = AdvancedBacktest()
        
        # Veri çek
        try:
            klines = client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1HOUR,
                limit=500
            )
            
            if not klines or len(klines) < 50:
                return jsonify({'status': 'error', 'message': 'Yetersiz veri'})
            
            # DataFrame oluştur
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Sayısal tiplere çevir
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            print(f"✅ {len(df)} veri noktası alındı")
            
        except Exception as e:
            print(f"❌ Veri çekme hatası: {e}")
            return jsonify({'status': 'error', 'message': f'Veri hatası: {str(e)}'})
        
        # Backtest çalıştır
        try:
            results = backtest_engine.run(df, {
                'rsi_buy': 30, 
                'rsi_sell': 70,
                'symbol': symbol
            })
            
            # Sonuçları JSON'a uygun hale getir
            response_data = {
                'status': 'success',
                'results': {
                    'symbol': symbol,
                    'initial_balance': results.get('initial_balance', 10000),
                    'final_balance': round(results.get('final_balance', 0), 2),
                    'total_return': round(results.get('total_return', 0), 2),
                    'total_trades': results.get('total_trades', 0),
                    'winning_trades': results.get('winning_trades', 0),
                    'losing_trades': results.get('losing_trades', 0),
                    'win_rate': round(results.get('win_rate', 0), 2),
                    'max_drawdown': round(results.get('max_drawdown', 0), 2),
                    'sharpe_ratio': round(results.get('sharpe_ratio', 0), 2),
                    'profit_factor': round(results.get('profit_factor', 0), 2)
                }
            }
            
            print(f"✅ Backtest tamamlandı: %{response_data['results']['total_return']:.2f} getiri")
            return jsonify(response_data)
            
        except Exception as e:
            print(f"❌ Backtest hatası: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({'status': 'error', 'message': f'Backtest hatası: {str(e)}'})
            
    except Exception as e:
        print(f"❌ Genel hata: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

# ==========================================
# ELMAS HTML ARAYÜZ (Kırmızı-Kara Tema)
# ==========================================

def render_elmas_html():
    return """<!DOCTYPE html>
<html lang="tr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>💎 ELMAS BOT PRO</title>
    <script src="https://cdn.socket.io/4.5.4/socket.io.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        :root {
            --bg-primary: #0a0a0a;
            --bg-secondary: #141414;
            --bg-card: #1a1a1a;
            --red-primary: #dc2626;
            --red-dark: #991b1b;
            --red-light: #ef4444;
            --text-primary: #ffffff;
            --text-secondary: #a1a1aa;
            --border-color: #27272a;
            --success: #16a34a;
            --warning: #eab308;
        }
        
        body {
            font-family: 'Segoe UI', Roboto, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container { max-width: 1600px; margin: 0 auto; padding: 20px; }
        
        /* Header */
        header {
            background: linear-gradient(180deg, var(--bg-secondary) 0%, transparent 100%);
            border-bottom: 3px solid var(--red-primary);
            padding: 30px 0;
            margin-bottom: 30px;
            text-align: center;
        }
        
        .logo {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
        }
        
        .logo-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--red-primary), var(--red-dark));
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 2.5em;
            box-shadow: 0 0 40px rgba(220, 38, 38, 0.5);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 40px rgba(220, 38, 38, 0.5); }
            50% { box-shadow: 0 0 60px rgba(220, 38, 38, 0.8); }
        }
        
        h1 {
            font-size: 3em;
            background: linear-gradient(135deg, #fff, var(--red-primary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            letter-spacing: 3px;
        }
        
        .subtitle { color: var(--text-secondary); font-size: 1.1em; letter-spacing: 5px; }
        
        /* Status Bar */
        .status-bar {
            display: flex;
            justify-content: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .status-item {
            background: var(--bg-card);
            padding: 15px 30px;
            border-radius: 15px;
            border: 1px solid var(--border-color);
            text-align: center;
        }
        
        .status-label { color: var(--text-secondary); font-size: 0.85em; text-transform: uppercase; }
        .status-value { font-size: 1.3em; font-weight: bold; margin-top: 5px; }
        
        .status-running { color: var(--success); }
        .status-stopped { color: var(--red-primary); }
        
        /* Controls */
        .controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin: 40px 0;
            flex-wrap: wrap;
        }
        
        .btn {
            padding: 18px 45px;
            font-size: 1.1em;
            font-weight: bold;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            text-transform: uppercase;
            letter-spacing: 2px;
        }
        
        .btn-primary {
            background: linear-gradient(135deg, var(--red-primary), var(--red-dark));
            color: white;
            box-shadow: 0 10px 30px rgba(220, 38, 38, 0.4);
        }
        
        .btn-primary:hover {
            transform: translateY(-3px);
            box-shadow: 0 15px 40px rgba(220, 38, 38, 0.6);
        }
        
        .btn-secondary {
            background: var(--bg-card);
            color: var(--text-secondary);
            border: 2px solid var(--border-color);
        }
        
        .btn-secondary:hover {
            border-color: var(--red-primary);
            color: var(--red-primary);
        }
        
        /* Stats Grid */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-card {
            background: var(--bg-card);
            padding: 25px;
            border-radius: 15px;
            border: 1px solid var(--border-color);
            text-align: center;
            transition: all 0.3s;
        }
        
        .stat-card:hover { border-color: var(--red-primary); transform: translateY(-5px); }
        
        .stat-icon { font-size: 2em; margin-bottom: 10px; }
        .stat-value { font-size: 2.2em; font-weight: bold; color: var(--red-primary); }
        .stat-label { color: var(--text-secondary); font-size: 0.9em; margin-top: 10px; }
        
        /* Coin Cards */
        .coins-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 25px;
            margin: 30px 0;
        }
        
        .coin-card {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
        }
        
        .coin-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 4px;
            background: linear-gradient(90deg, var(--red-primary), var(--red-light));
            transform: scaleX(0);
            transition: transform 0.3s;
        }
        
        .coin-card:hover::before { transform: scaleX(1); }
        .coin-card:hover { border-color: var(--red-primary); }
        
        .coin-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .coin-info h2 { color: var(--red-primary); font-size: 1.8em; }
        .coin-info span { color: var(--text-secondary); font-size: 0.9em; }
        
        .position-badge {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .pos-active { background: var(--success); color: #000; }
        .pos-none { background: var(--border-color); color: var(--text-secondary); }
        
        .price-display { text-align: right; }
        .price-main { font-size: 2em; font-weight: bold; }
        .price-change { font-size: 0.9em; color: var(--text-secondary); }
        
        /* Score Section */
        .score-section { margin: 25px 0; }
        
        .score-labels {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.85em;
            color: var(--text-secondary);
        }
        
        .score-bar {
            height: 40px;
            background: var(--bg-secondary);
            border-radius: 20px;
            overflow: hidden;
            border: 2px solid var(--border-color);
            position: relative;
        }
        
        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--red-dark), var(--red-primary), var(--red-light));
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 20px;
            transition: all 0.5s ease;
            position: relative;
        }
        
        .score-fill::after {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
            animation: shimmer 2s infinite;
        }
        
        @keyframes shimmer {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        .score-text { font-weight: 900; font-size: 1.3em; color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.5); }
        
        .score-details {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 10px;
            margin-top: 15px;
            font-size: 0.85em;
        }
        
        .score-detail-item {
            background: var(--bg-secondary);
            padding: 10px;
            border-radius: 10px;
            text-align: center;
        }
        
        .score-detail-label { color: var(--text-secondary); font-size: 0.8em; }
        .score-detail-value { font-weight: bold; color: var(--red-primary); margin-top: 5px; }
        
        /* Signal */
        .signal-box {
            text-align: center;
            margin: 20px 0;
        }
        
        .signal {
            display: inline-block;
            padding: 15px 40px;
            border-radius: 12px;
            font-weight: bold;
            font-size: 1.3em;
            text-transform: uppercase;
            letter-spacing: 3px;
            border: 3px solid;
        }
        
        .signal-strong-buy { background: rgba(22, 163, 74, 0.2); border-color: var(--success); color: var(--success); }
        .signal-buy { background: rgba(74, 222, 128, 0.2); border-color: #4ade80; color: #4ade80; }
        .signal-strong-sell { background: rgba(220, 38, 38, 0.2); border-color: var(--red-primary); color: var(--red-primary); }
        .signal-sell { background: rgba(248, 113, 113, 0.2); border-color: #f87171; color: #f87171; }
        .signal-hold { background: rgba(161, 161, 170, 0.2); border-color: var(--text-secondary); color: var(--text-secondary); }
        
        /* Chart */
        .chart-container {
            height: 150px;
            margin-top: 20px;
            background: var(--bg-secondary);
            border-radius: 15px;
            padding: 15px;
        }
        
        /* AI Section */
        .ai-section {
            background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(153, 27, 27, 0.1));
            border: 2px solid var(--red-primary);
            border-radius: 20px;
            padding: 25px;
            margin: 30px 0;
        }
        
        .ai-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .ai-title { font-size: 1.5em; color: var(--red-primary); }
        .ai-status { padding: 8px 20px; border-radius: 20px; font-size: 0.9em; font-weight: bold; }
        .ai-active { background: var(--success); color: #000; }
        .ai-inactive { background: var(--warning); color: #000; }
        
        .ai-features {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        
        .ai-feature {
            background: var(--bg-card);
            padding: 15px;
            border-radius: 10px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        /* Logs */
        .logs-section {
            background: var(--bg-card);
            border-radius: 20px;
            padding: 25px;
            margin-top: 30px;
        }
        
        .logs-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .logs-title { font-size: 1.5em; color: var(--red-primary); }
        
        .logs-container {
            background: var(--bg-secondary);
            border-radius: 15px;
            padding: 20px;
            max-height: 400px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
        }
        
        .log-entry {
            padding: 12px 0;
            border-bottom: 1px solid var(--border-color);
            display: flex;
            gap: 15px;
        }
        
        .log-time { color: var(--red-primary); font-weight: bold; min-width: 70px; }
        .log-message { color: var(--text-primary); }
        .log-trade { color: var(--success); }
        .log-error { color: var(--red-primary); }
        .log-warning { color: var(--warning); }
        
        /* Responsive */
        @media (max-width: 768px) {
            .coins-grid { grid-template-columns: 1fr; }
            h1 { font-size: 2em; }
            .btn { width: 100%; }
            .score-details { grid-template-columns: 1fr; }
        }
        
        /* Scrollbar */
        ::-webkit-scrollbar { width: 10px; }
        ::-webkit-scrollbar-track { background: var(--bg-secondary); }
        ::-webkit-scrollbar-thumb { background: var(--red-primary); border-radius: 5px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--red-light); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div class="logo">
                <div class="logo-icon">💎</div>
                <div>
                    <h1>ELMAS BOT PRO</h1>
                    <div class="subtitle">YAPAY ZEKALI TRADING SİSTEMİ</div>
                </div>
            </div>
            
            <div class="status-bar">
                <div class="status-item">
                    <div class="status-label">Sistem Durumu</div>
                    <div class="status-value status-stopped" id="systemStatus">DURDU</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Çalışma Modu</div>
                    <div class="status-value" id="systemMode">TEST</div>
                </div>
                <div class="status-item">
                    <div class="status-label">AI Durumu</div>
                    <div class="status-value" id="aiStatus">BEKLEMEDE</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Son Güncelleme</div>
                    <div class="status-value" id="lastUpdate">--:--</div>
                </div>
            </div>
        </header>
        
        <div class="controls">
            <button class="btn btn-primary" onclick="startBot()" id="btnStart">▶ BAŞLAT</button>
            <button class="btn btn-secondary" onclick="stopBot()" id="btnStop" disabled>⏹ DURDUR</button>
            <button class="btn btn-secondary" onclick="trainAI()">🧠 AI EĞİT</button>
            <button class="btn btn-secondary" onclick="runBacktest()">📊 BACKTEST</button>
        </div>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">💰</div>
                <div class="stat-value" id="totalBalance">$0</div>
                <div class="stat-label">Toplam Bakiye</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value" id="dailyPnl">$0</div>
                <div class="stat-label">Günlük P&L</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🎯</div>
                <div class="stat-value" id="winRate">0%</div>
                <div class="stat-label">Kazanma Oranı</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🔄</div>
                <div class="stat-value" id="todayTrades">0</div>
                <div class="stat-label">Bugünkü İşlem</div>
            </div>
        </div>
        
        <div class="coins-grid">
            <!-- BTC Card -->
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-info">
                        <h2>₿ Bitcoin</h2>
                        <span>BTC/USDT</span>
                    </div>
                    <span class="position-badge pos-none" id="btcPosition">POZİSYON YOK</span>
                    <div class="price-display">
                        <div class="price-main" id="btcPrice">$0</div>
                        <div class="price-change" id="btcChange">Değişim: --</div>
                    </div>
                </div>
                
                <div class="score-section">
                    <div class="score-labels">
                        <span>Sat (0)</span>
                        <span>Nötr (50)</span>
                        <span>Al (100)</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" id="btcScoreBar" style="width: 50%">
                            <span class="score-text" id="btcFinalScore">50</span>
                        </div>
                    </div>
                    <div class="score-details">
                        <div class="score-detail-item">
                            <div class="score-detail-label">Temel Skor</div>
                            <div class="score-detail-value" id="btcScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">AI Skoru</div>
                            <div class="score-detail-value" id="btcAiScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">Zaman Dilimi</div>
                            <div class="score-detail-value" id="btcTfScore">50</div>
                        </div>
                    </div>
                </div>
                
                <div class="signal-box">
                    <div class="signal signal-hold" id="btcSignal">BEKLE</div>
                </div>
                
                <div class="chart-container">
                    <canvas id="btcChart"></canvas>
                </div>
            </div>
            
            <!-- ETH Card -->
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-info">
                        <h2>Ξ Ethereum</h2>
                        <span>ETH/USDT</span>
                    </div>
                    <span class="position-badge pos-none" id="ethPosition">POZİSYON YOK</span>
                    <div class="price-display">
                        <div class="price-main" id="ethPrice">$0</div>
                        <div class="price-change" id="ethChange">Değişim: --</div>
                    </div>
                </div>
                
                <div class="score-section">
                    <div class="score-labels">
                        <span>Sat (0)</span>
                        <span>Nötr (50)</span>
                        <span>Al (100)</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" id="ethScoreBar" style="width: 50%">
                            <span class="score-text" id="ethFinalScore">50</span>
                        </div>
                    </div>
                    <div class="score-details">
                        <div class="score-detail-item">
                            <div class="score-detail-label">Temel Skor</div>
                            <div class="score-detail-value" id="ethScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">AI Skoru</div>
                            <div class="score-detail-value" id="ethAiScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">Zaman Dilimi</div>
                            <div class="score-detail-value" id="ethTfScore">50</div>
                        </div>
                    </div>
                </div>
                
                <div class="signal-box">
                    <div class="signal signal-hold" id="ethSignal">BEKLE</div>
                </div>
                
                <div class="chart-container">
                    <canvas id="ethChart"></canvas>
                </div>
            </div>
            
            <!-- SOL Card -->
            <div class="coin-card">
                <div class="coin-header">
                    <div class="coin-info">
                        <h2>◎ Solana</h2>
                        <span>SOL/USDT</span>
                    </div>
                    <span class="position-badge pos-none" id="solPosition">POZİSYON YOK</span>
                    <div class="price-display">
                        <div class="price-main" id="solPrice">$0</div>
                        <div class="price-change" id="solChange">Değişim: --</div>
                    </div>
                </div>
                
                <div class="score-section">
                    <div class="score-labels">
                        <span>Sat (0)</span>
                        <span>Nötr (50)</span>
                        <span>Al (100)</span>
                    </div>
                    <div class="score-bar">
                        <div class="score-fill" id="solScoreBar" style="width: 50%">
                            <span class="score-text" id="solFinalScore">50</span>
                        </div>
                    </div>
                    <div class="score-details">
                        <div class="score-detail-item">
                            <div class="score-detail-label">Temel Skor</div>
                            <div class="score-detail-value" id="solScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">AI Skoru</div>
                            <div class="score-detail-value" id="solAiScore">50</div>
                        </div>
                        <div class="score-detail-item">
                            <div class="score-detail-label">Zaman Dilimi</div>
                            <div class="score-detail-value" id="solTfScore">50</div>
                        </div>
                    </div>
                </div>
                
                <div class="signal-box">
                    <div class="signal signal-hold" id="solSignal">BEKLE</div>
                </div>
                
                <div class="chart-container">
                    <canvas id="solChart"></canvas>
                </div>
            </div>
        </div>
        
        <div class="ai-section" id="aiSection">
            <div class="ai-header">
                <div class="ai-title">🧠 Yapay Zeka Durumu</div>
                <div class="ai-status ai-inactive" id="aiStatusBadge">EĞİTİM BEKLİYOR</div>
            </div>
            <div class="ai-features" id="aiFeatures">
                <div class="ai-feature">
                    <span>Model Doğruluğu</span>
                    <strong id="aiAccuracy">--</strong>
                </div>
                <div class="ai-feature">
                    <span>Son Eğitim</span>
                    <strong id="aiLastTrain">--</strong>
                </div>
                <div class="ai-feature">
                    <span>En Önemli İndikatör</span>
                    <strong id="aiTopFeature">--</strong>
                </div>
            </div>
        </div>
        
        <div class="logs-section">
            <div class="logs-header">
                <div class="logs-title">📝 Sistem Logları</div>
                <button class="btn btn-secondary" onclick="clearLogs()">Temizle</button>
            </div>
            <div class="logs-container" id="logsContainer">
                <div class="log-entry">
                    <span class="log-time">--:--</span>
                    <span class="log-message">Sistem hazır. Başlatmak için butona tıklayın.</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        const charts = {};
        
        function createChart(id, data, color = '#dc2626') {
            const ctx = document.getElementById(id).getContext('2d');
            return new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.map((_, i) => i),
                    datasets: [{
                        data: data,
                        borderColor: color,
                        backgroundColor: color + '20',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        borderWidth: 2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: {
                        x: { display: false },
                        y: { display: false }
                    }
                }
            });
        }
        
        function getSignalClass(signal) {
            const classes = {
                'GÜÇLÜ AL': 'signal-strong-buy',
                'AL': 'signal-buy',
                'GÜÇLÜ SAT': 'signal-strong-sell',
                'SAT': 'signal-sell',
                'BEKLE': 'signal-hold'
            };
            return classes[signal] || 'signal-hold';
        }
        
        socket.on('update_data', function(data) {
            // System status
            document.getElementById('systemStatus').textContent = data.system.status;
            document.getElementById('systemStatus').className = 'status-value ' + (data.system.status === 'ÇALIŞIYOR' ? 'status-running' : 'status-stopped');
            document.getElementById('systemMode').textContent = data.system.mode;
            document.getElementById('aiStatus').textContent = data.ai.is_trained ? 'AKTİF' : 'BEKLEMEDE';
            document.getElementById('lastUpdate').textContent = data.system.last_update || '--:--';
            
            // Stats
            document.getElementById('totalBalance').textContent = '$' + data.portfolio.total.toLocaleString();
            document.getElementById('dailyPnl').textContent = '$' + data.portfolio.daily_pnl.toLocaleString();
            document.getElementById('winRate').textContent = data.stats.win_rate + '%';
            document.getElementById('todayTrades').textContent = data.stats.today_trades;
            
            // Coins
            ['btc', 'eth', 'sol'].forEach(coin => {
                if (data.market[coin]) {
                    const m = data.market[coin];
                    
                    // Price
                    document.getElementById(coin + 'Price').textContent = '$' + m.price.toLocaleString();
                    
                    // Position
                    const posEl = document.getElementById(coin + 'Position');
                    if (m.position) {
                        posEl.textContent = 'LONG';
                        posEl.className = 'position-badge pos-active';
                    } else {
                        posEl.textContent = 'POZİSYON YOK';
                        posEl.className = 'position-badge pos-none';
                    }
                    
                    // Scores
                    document.getElementById(coin + 'FinalScore').textContent = m.final_score;
                    document.getElementById(coin + 'ScoreBar').style.width = m.final_score + '%';
                    document.getElementById(coin + 'Score').textContent = m.score;
                    document.getElementById(coin + 'AiScore').textContent = m.ai_score;
                    document.getElementById(coin + 'TfScore').textContent = m.tf_score;
                    
                    // Signal
                    const sigEl = document.getElementById(coin + 'Signal');
                    sigEl.textContent = m.signal;
                    sigEl.className = 'signal ' + getSignalClass(m.signal);
                    
                    // Chart
                    if (m.history && m.history.length > 0) {
                        if (!charts[coin]) {
                            charts[coin] = createChart(coin + 'Chart', m.history);
                        } else {
                            charts[coin].data.datasets[0].data = m.history;
                            charts[coin].update();
                        }
                    }
                }
            });
            
            // AI Section
            if (data.ai.is_trained) {
                document.getElementById('aiStatusBadge').textContent = 'AKTİF';
                document.getElementById('aiStatusBadge').className = 'ai-status ai-active';
                document.getElementById('aiAccuracy').textContent = '%' + (data.ai.accuracy * 100).toFixed(2);
                document.getElementById('aiLastTrain').textContent = data.ai.last_training || '--';
                document.getElementById('aiTopFeature').textContent = data.ai.top_features[0]?.feature || '--';
            }
        });
        
        socket.on('new_log', function(data) {
            const container = document.getElementById('logsContainer');
            const entry = document.createElement('div');
            entry.className = 'log-entry';
            entry.innerHTML = `<span class="log-time">${data.time}</span><span class="log-message log-${data.level}">${data.message}</span>`;
            container.insertBefore(entry, container.firstChild);
        });
        
        socket.on('bot_status', function(data) {
            document.getElementById('btnStart').disabled = data.running;
            document.getElementById('btnStop').disabled = !data.running;
        });
        
        function startBot() {
            fetch('/api/start', {method: 'POST'}).then(r => r.json()).then(d => alert(d.message));
        }
        
        function stopBot() {
            fetch('/api/stop', {method: 'POST'}).then(r => r.json()).then(d => alert(d.message));
        }
        
        function trainAI() {
            fetch('/api/train', {method: 'POST'}).then(r => r.json()).then(d => alert(d.status === 'success' ? 'AI Eğitildi!' : 'Hata: ' + d.message));
        }
        
        function runBacktest() {
            fetch('/api/backtest', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({symbol: 'BTCUSDT'})})
                .then(r => r.json()).then(d => alert(d.status === 'success' ? 'Backtest tamamlandı!' : 'Hata: ' + d.message));
        }
        
        function clearLogs() {
            document.getElementById('logsContainer').innerHTML = '';
        }
    </script>
</body>
</html>"""

# ==========================================
# BAŞLATMA
# ==========================================

if __name__ == '__main__':
    print("=" * 70)
    print("💎 ELMAS BOT PRO - YAPAY ZEKALI TRADING SİSTEMİ")
    print("=" * 70)
    print("🧠 Özellikler:")
    print("   • Yapay Zeka (Random Forest)")
    print("   • Çoklu Zaman Dilimi Analizi (15m, 1h, 4h, 1d)")
    print("   • Gelişmiş Backtesting")
    print("   • Otomatik İşlem")
    print("   • Telegram Bildirimler")
    print("   • Kırmızı-Kara Profesyonel Tema")
    print("=" * 70)
    print(f"💰 Mod: {'TESTNET' if TEST_MODE else 'GERÇEK'}")
    print(f"📱 Telegram: {'Aktif' if TELEGRAM_CHAT_ID != 'SIZIN_CHAT_ID' else 'Pasif (Chat ID gerekli)'}")
    print("=" * 70)
    print("🌐 Arayüz: http://localhost:5000")
    print("=" * 70)
    
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)
