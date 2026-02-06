# ==========================================
# ÇOKLU COİN YÖNETİCİSİ
# ==========================================

import pandas as pd
from concurrent.futures import ThreadPoolExecutor

class MultiCoinManager:
    def __init__(self, client, analyzer):
        self.client = client
        self.analyzer = analyzer
        self.coins = {
            'BTCUSDT': {'weight': 0.3, 'active': True},  # %30
            'ETHUSDT': {'weight': 0.25, 'active': True}, # %25
            'SOLUSDT': {'weight': 0.2, 'active': True},  # %20
            'ADAUSDT': {'weight': 0.15, 'active': True}, # %15
            'BNBUSDT': {'weight': 0.1, 'active': True}   # %10
        }
        self.positions = {}
        self.daily_pnl = {}
        
    def analyze_all_coins(self):
        """Tüm coinleri analiz et"""
        results = {}
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(self.analyze_single, symbol): symbol 
                for symbol in self.coins.keys() 
                if self.coins[symbol]['active']
            }
            
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"❌ {symbol} analiz hatası: {e}")
        
        return results
    
    def analyze_single(self, symbol):
        """Tek coin analizi"""
        score, details = self.analyzer.calculate_signal_score(self.client, symbol)
        
        return {
            'symbol': symbol,
            'score': score,
            'price': details.get('price', 0),
            'trend_15m': details.get('trend_15m', '-'),
            'volume': details.get('volume', '-'),
            'weight': self.coins[symbol]['weight']
        }
    
    def get_best_opportunity(self):
        """En iyi fırsatı bul"""
        results = self.analyze_all_coins()
        
        # Skora göre sırala
        sorted_coins = sorted(
            results.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        # En iyi 3 fırsat
        top_3 = sorted_coins[:3]
        
        return top_3
    
    def allocate_funds(self, total_balance):
        """Fon dağılımı"""
        allocations = {}
        
        for symbol, config in self.coins.items():
            if config['active']:
                amount = total_balance * config['weight']
                allocations[symbol] = amount
        
        return allocations
    
    def rebalance_portfolio(self, current_balances):
        """Portföy dengeleme"""
        # Hedef ağırlıklara göre ayarlama
        total = sum(current_balances.values())
        recommendations = []
        
        for symbol, target_weight in [(s, c['weight']) for s, c in self.coins.items()]:
            current_amount = current_balances.get(symbol, 0)
            current_weight = current_amount / total if total > 0 else 0
            
            if abs(current_weight - target_weight) > 0.05:  # %5 tolerans
                recommendations.append({
                    'symbol': symbol,
                    'action': 'INCREASE' if current_weight < target_weight else 'DECREASE',
                    'current': current_weight,
                    'target': target_weight
                })
        
        return recommendations