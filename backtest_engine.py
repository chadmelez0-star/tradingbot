# ==========================================
# GELİŞMİŞ BACKTEST MOTORU (GÜNCELLENMİŞ)
# ==========================================

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

class AdvancedBacktest:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        self.results = {}
        
    def run(self, df, strategy_params):
        """Strateji backtest'i - GÜNCELLENMİŞ"""
        try:
            balance = self.initial_balance
            position = None
            entry_price = 0
            trades = []
            equity_curve = [self.initial_balance]
            
            # Minimum veri kontrolü
            if len(df) < 50:
                return self._empty_results("Yetersiz veri")
            
            symbol = strategy_params.get('symbol', 'UNKNOWN')
            
            for i in range(50, len(df)):
                try:
                    current = df.iloc[i]
                    past = df.iloc[:i]
                    
                    # Sinyal üret
                    signal = self.generate_signal(past, strategy_params)
                    
                    # İşlem mantığı
                    if signal == 'BUY' and position is None:
                        position = 'LONG'
                        entry_price = float(current['close'])
                        size = (balance * 0.2) / entry_price  # %20 risk
                        
                        trades.append({
                            'type': 'BUY',
                            'price': entry_price,
                            'time': i,
                            'size': size,
                            'balance': balance
                        })
                        
                    elif signal == 'SELL' and position == 'LONG':
                        exit_price = float(current['close'])
                        pnl = (exit_price - entry_price) * size
                        balance += pnl
                        position = None
                        
                        trades.append({
                            'type': 'SELL',
                            'price': exit_price,
                            'time': i,
                            'pnl': pnl,
                            'pnl_pct': (exit_price - entry_price) / entry_price * 100 if entry_price else 0,
                            'balance': balance
                        })
                        
                        equity_curve.append(balance)
                        
                except Exception as e:
                    print(f"  ⚠️ İşlem hatası (index {i}): {e}")
                    continue
            
            # Sonuçları hesapla
            return self._calculate_results(balance, trades, equity_curve, symbol)
            
        except Exception as e:
            print(f"❌ Backtest çalışma hatası: {e}")
            return self._empty_results(str(e))
    
    def generate_signal(self, df, params):
        """Sinyal üret - Güvenli"""
        try:
            # RSI hesapla
            closes = df['close'].astype(float)
            delta = closes.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=5).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
            
            if loss.iloc[-1] == 0 or pd.isna(loss.iloc[-1]):
                rsi = 50
            else:
                rs = gain.iloc[-1] / loss.iloc[-1]
                rsi = 100 - (100 / (1 + rs))
            
            # MACD hesapla
            ema_12 = closes.ewm(span=12, adjust=False, min_periods=5).mean()
            ema_26 = closes.ewm(span=26, adjust=False, min_periods=5).mean()
            macd = ema_12.iloc[-1] - ema_26.iloc[-1]
            
            # Sinyal
            if rsi < params.get('rsi_buy', 30) and macd > 0:
                return 'BUY'
            elif rsi > params.get('rsi_sell', 70) and macd < 0:
                return 'SELL'
            return 'HOLD'
            
        except Exception as e:
            print(f"  ⚠️ Sinyal hatası: {e}")
            return 'HOLD'
    
    def _calculate_results(self, final_balance, trades, equity_curve, symbol):
        """Sonuçları hesapla"""
        try:
            total_return = (final_balance - self.initial_balance) / self.initial_balance * 100
            
            sell_trades = [t for t in trades if t['type'] == 'SELL']
            
            if sell_trades:
                winning_trades = len([t for t in sell_trades if t.get('pnl', 0) > 0])
                losing_trades = len([t for t in sell_trades if t.get('pnl', 0) < 0])
                win_rate = winning_trades / len(sell_trades) * 100
                
                profits = [t['pnl'] for t in sell_trades if t.get('pnl', 0) > 0]
                losses = [abs(t['pnl']) for t in sell_trades if t.get('pnl', 0) < 0]
                
                avg_profit = np.mean(profits) if profits else 0
                avg_loss = np.mean(losses) if losses else 0
                
                # Drawdown
                peak = self.initial_balance
                max_drawdown = 0
                for eq in equity_curve:
                    if eq > peak:
                        peak = eq
                    drawdown = (peak - eq) / peak * 100 if peak > 0 else 0
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown
                
                # Sharpe ratio
                returns = pd.Series(equity_curve).pct_change().dropna()
                sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if len(returns) > 1 and returns.std() != 0 else 0
                
                profit_factor = avg_profit / avg_loss if avg_loss > 0 else (999 if avg_profit > 0 else 0)
                
            else:
                winning_trades = losing_trades = win_rate = 0
                avg_profit = avg_loss = max_drawdown = sharpe = profit_factor = 0
            
            results = {
                'symbol': symbol,
                'initial_balance': self.initial_balance,
                'final_balance': round(final_balance, 2),
                'total_return': round(total_return, 2),
                'total_trades': len(sell_trades),
                'winning_trades': winning_trades,
                'losing_trades': losing_trades,
                'win_rate': round(win_rate, 2),
                'avg_profit': round(avg_profit, 2),
                'avg_loss': round(avg_loss, 2),
                'profit_factor': round(profit_factor, 2),
                'max_drawdown': round(max_drawdown, 2),
                'sharpe_ratio': round(sharpe, 2),
                'equity_curve': equity_curve,
                'trades': trades
            }
            
            self.save_results(results)
            return results
            
        except Exception as e:
            print(f"❌ Sonuç hesaplama hatası: {e}")
            return self._empty_results(str(e))
    
    def _empty_results(self, error_msg):
        """Boş sonuç döndür"""
        return {
            'symbol': 'ERROR',
            'initial_balance': self.initial_balance,
            'final_balance': self.initial_balance,
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'equity_curve': [self.initial_balance],
            'trades': [],
            'error': error_msg
        }
    
    def save_results(self, results):
        """Sonuçları kaydet"""
        try:
            os.makedirs('backtest_results', exist_ok=True)
            filename = f"backtest_results/backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # JSON'a uygun hale getir
            results_json = {k: v for k, v in results.items() if k not in ['equity_curve', 'trades']}
            results_json['trades_count'] = len(results.get('trades', []))
            
            with open(filename, 'w') as f:
                json.dump(results_json, f, indent=2)
            
            print(f"✅ Backtest kaydedildi: {filename}")
        except Exception as e:
            print(f"⚠️ Kaydetme hatası: {e}")
    
    def print_report(self, results):
        """Rapor yazdır"""
        print("\n" + "="*60)
        print("📊 BACKTEST RAPORU")
        print("="*60)
        print(f"Sembol: {results.get('symbol', 'N/A')}")
        print(f"Başlangıç Bakiyesi: ${results['initial_balance']:,.2f}")
        print(f"Bitiş Bakiyesi: ${results['final_balance']:,.2f}")
        print(f"Toplam Getiri: %{results['total_return']:.2f}")
        print(f"-"*60)
        print(f"Toplam İşlem: {results['total_trades']}")
        print(f"Kazanan: {results['winning_trades']} | Kaybeden: {results['losing_trades']}")
        print(f"Kazanma Oranı: %{results['win_rate']:.2f}")
        print(f"-"*60)
        print(f"Ortalama Kâr: ${results['avg_profit']:.2f}")
        print(f"Ortalama Zarar: ${results['avg_loss']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        print(f"-"*60)
        print(f"Maksimum Düşüş: %{results['max_drawdown']:.2f}")
        print(f"Sharpe Oranı: {results['sharpe_ratio']:.2f}")
        print("="*60)