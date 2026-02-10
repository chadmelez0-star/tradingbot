# ==========================================
# ÇOKLU ZAMAN DİLİMİ ANALİZİ
# ==========================================

import pandas as pd
import numpy as np
import time
from binance.enums import *

class MultiTimeframeAnalyzer:
    def __init__(self, client):
        self.client = client
        self.timeframes = {
            '15m': KLINE_INTERVAL_15MINUTE,
            '1h': KLINE_INTERVAL_1HOUR,
            '4h': KLINE_INTERVAL_4HOUR,
            '1d': KLINE_INTERVAL_1DAY
        }
    
    def get_data(self, symbol, interval, limit=100):
        """Veri çek - Geliştirilmiş"""
        max_retries = 3  # ← Tekrar deneme mekanizması
        
        for attempt in range(max_retries):
            try:
                # Limiti biraz artır (Binance bazen eksik veriyor)
                actual_limit = min(limit + 50, 1000)
                
                klines = self.client.get_klines(
                    symbol=symbol, 
                    interval=interval, 
                    limit=actual_limit
                )
                
                if not klines:
                    if attempt < max_retries - 1:
                        time.sleep(1)  # ← Bekle ve tekrar dene
                        continue
                    return None
                
                df = pd.DataFrame(klines, columns=[
                    'timestamp', 'open', 'high', 'low', 'close', 'volume',
                    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                    'taker_buy_quote', 'ignore'
                ])
                
                # Sayısal dönüşümler
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # İstenen limite göre son kayıtları al
                if len(df) > limit:
                    df = df.tail(limit).reset_index(drop=True)
                
                # En az %80'i geçerli mi?
                valid_ratio = df['close'].notna().sum() / len(df)
                if valid_ratio < 0.8:
                    self.log(f"  ⚠️ {symbol} {interval}: Çok fazla eksik veri (%{valid_ratio*100:.0f})")
                    return None
                
                return df
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    print(f"❌ Veri çekme hatası {symbol} {interval}: {e}")
                    return None
        
        return None
    
    def analyze_timeframe(self, df):
        """Tek zaman dilimi analizi"""
        if df is None or len(df) < 50:
            return {'trend': 'N/A', 'strength': 0, 'score': 50}
        
        prices = df['close'].values
        
        # EMA'lar
        ema_20 = pd.Series(prices).ewm(span=20, adjust=False).mean().iloc[-1]
        ema_50 = pd.Series(prices).ewm(span=50, adjust=False).mean().iloc[-1]
        current = prices[-1]
        
        # Trend gücü
        if current > ema_20 > ema_50:
            trend = 'YUKARI'
            strength = min(100, 50 + (current - ema_20) / ema_20 * 1000)
        elif current < ema_20 < ema_50:
            trend = 'AŞAĞI'
            strength = min(100, 50 + (ema_20 - current) / ema_20 * 1000)
        else:
            trend = 'YAN'
            strength = 30
        
        # RSI
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean().iloc[-1]
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean().iloc[-1]
        rs = gain / loss if loss != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        
        # Skor
        if trend == 'YUKARI':
            score = 50 + strength * 0.5
            if rsi < 70:
                score += 20
        elif trend == 'AŞAĞI':
            score = 50 - strength * 0.5
            if rsi > 30:
                score -= 20
        else:
            score = 50
        
        return {
            'trend': trend,
            'strength': round(strength, 2),
            'score': max(0, min(100, score)),
            'rsi': round(rsi, 2),
            'ema_20': ema_20,
            'ema_50': ema_50
        }
    
    def analyze_all_timeframes(self, symbol):
        """Tüm zaman dilimlerini analiz et"""
        results = {}
        
        for name, interval in self.timeframes.items():
            df = self.get_data(symbol, interval)
            results[name] = self.analyze_timeframe(df)
        
        # Ağırlıklı ortalama
        weights = {'15m': 0.1, '1h': 0.3, '4h': 0.4, '1d': 0.2}
        total_score = sum(results[tf]['score'] * weights[tf] for tf in results if results[tf]['trend'] != 'N/A')
        
        # Trend uyumu
        trends = [results[tf]['trend'] for tf in results if results[tf]['trend'] != 'N/A']
        if all(t == 'YUKARI' for t in trends):
            consensus = 'GÜÇLÜ YUKARI'
        elif all(t == 'AŞAĞI' for t in trends):
            consensus = 'GÜÇLÜ AŞAĞI'
        elif trends.count('YUKARI') > trends.count('AŞAĞI'):
            consensus = 'YUKARI'
        elif trends.count('AŞAĞI') > trends.count('YUKARI'):
            consensus = 'AŞAĞI'
        else:
            consensus = 'KARARSIZ'
        
        return {
            'timeframes': results,
            'total_score': round(total_score, 2),
            'consensus': consensus
        }
