# ==========================================
# GELİŞMİŞ TELEGRAM BOTU
# ==========================================

import requests
from datetime import datetime

class AdvancedTelegramBot:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.enabled = token != 'SIZIN_BOT_TOKEN' and chat_id != 'SIZIN_CHAT_ID'
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send(self, message, parse_mode='HTML'):
        """Mesaj gönder"""
        if not self.enabled:
            return False
        
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': parse_mode
            }
            response = requests.post(url, data=data, timeout=10)
            return response.json().get('ok', False)
        except Exception as e:
            print(f"Telegram hatası: {e}")
            return False
    
    def trade_notification(self, symbol, action, price, amount, pnl=None, strategy_info=None):
        """İşlem bildirimi"""
        emoji = "🟢" if action == "ALIM" else "🔴"
        
        strategy_text = ""
        if strategy_info:
            strategy_text = f"\n📊 <b>Strateji:</b> {strategy_info}"
        
        pnl_text = ""
        if pnl is not None:
            emoji_pnl = "📈" if pnl > 0 else "📉"
            pnl_text = f"\n{emoji_pnl} <b>P&L:</b> ${pnl:+.2f}"
        
        message = f"""
{emoji} <b>İŞLEM GERÇEKLEŞTİ</b>

💎 <b>{symbol}</b>
🎯 <b>{action}</b>
💵 Fiyat: ${price:,.2f}
📈 Miktar: {amount:.6f}{pnl_text}{strategy_text}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send(message)
    
    def daily_report(self, portfolio_data):
        """Günlük rapor"""
        message = f"""
📊 <b>GÜNLÜK PERFORMANS RAPORU</b>

💰 Toplam Bakiye: ${portfolio_data['total']:,.2f}
📈 Günlük P&L: ${portfolio_data['daily_pnl']:+.2f}
🔄 Toplam İşlem: {portfolio_data['trades']}
📊 Aktif Pozisyon: {portfolio_data['active_positions']}

<b>Coin Dağılımı:</b>
{self._format_positions(portfolio_data.get('positions', {}))}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M')}
"""
        return self.send(message)
    
    def ai_training_complete(self, accuracy, features):
        """AI eğitim bildirimi"""
        features_text = "\n".join([f"  {i+1}. {f['feature']}: %{f['importance']*100:.1f}" 
                                   for i, f in enumerate(features[:5])])
        
        message = f"""
🧠 <b>AI MODEL EĞİTİMİ TAMAMLANDI</b>

✅ Doğruluk: %{accuracy*100:.2f}

<b>En Önemli İndikatörler:</b>
{features_text}

🤖 Model artık aktif!
"""
        return self.send(message)
    
    def alert(self, title, message, level='info'):
        """Genel uyarı"""
        emojis = {
            'info': 'ℹ️',
            'warning': '⚠️',
            'error': '🚨',
            'success': '✅'
        }
        
        emoji = emojis.get(level, 'ℹ️')
        
        msg = f"""
{emoji} <b>{title}</b>

{message}

⏰ {datetime.now().strftime('%H:%M:%S')}
"""
        return self.send(msg)
    
    def _format_positions(self, positions):
        """Pozisyonları formatla"""
        if not positions:
            return "  Yok"
        
        text = ""
        for symbol, data in positions.items():
            text += f"  • {symbol}: {data['amount']:.4f} @ ${data['entry']:,.2f}\n"
        return text.strip()