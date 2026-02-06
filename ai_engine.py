# ==========================================
# YAPAY ZEKA MOTORU - ELMAS BOT (TAM DÜZELTME)
# ==========================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime

class AITradingEngine:
    def __init__(self):
        self.model_path = 'ai_models/trained_model.pkl'
        self.scaler_path = 'ai_models/scaler.pkl'
        self.model = None
        self.scaler = None
        self.feature_importance = {}
        self.is_trained = False
        self.feature_columns = None
        
    def create_advanced_features(self, df, for_training=True):
        """Gelişmiş özellik mühendisliği - DÜZELTİLMİŞ"""
        if df is None or len(df) < 50:
            print(f"  ⚠️ Yetersiz ham veri: {len(df) if df is not None else 0}")
            return pd.DataFrame()
        
        print(f"  📊 Ham veri: {len(df)} satır")
        
        # Kopya al ve tip dönüşümleri yap
        data = df.copy()
        
        # Temel kolonları kontrol et
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                print(f"  ❌ Eksik kolon: {col}")
                return pd.DataFrame()
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        features = pd.DataFrame(index=data.index)
        
        # 1. TEMEL FİYAT ÖZELLİKLERİ (en az hesaplama gerektiren)
        features['close'] = data['close']
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # 2. VOLATİLİTE (kısa window ile başla)
        features['volatility_5'] = features['returns'].rolling(window=5, min_periods=1).std()
        features['volatility_10'] = features['returns'].rolling(window=10, min_periods=1).std()
        
        # 3. HAREKETLİ ORTALAMALAR (min_periods=1 ile)
        for period in [5, 10, 20]:
            sma = data['close'].rolling(window=period, min_periods=1).mean()
            features[f'sma_{period}'] = sma
            features[f'ema_{period}'] = data['close'].ewm(span=period, adjust=False, min_periods=1).mean()
            features[f'distance_sma_{period}'] = (data['close'] - sma) / sma
        
        # 4. BASİT FİYAT ÖZELLİKLERİ
        features['high_low_pct'] = (data['high'] - data['low']) / data['close']
        features['open_close_pct'] = (data['close'] - data['open']) / data['open']
        
        # 5. HACİM ÖZELLİKLERİ
        features['volume'] = data['volume']
        features['volume_sma_5'] = data['volume'].rolling(window=5, min_periods=1).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma_5']
        
        # 6. BASİT RSI (14 period ama min_periods=5)
        delta = data['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14, min_periods=5).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=5).mean()
        rs = gain / loss.replace(0, np.nan)
        features['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 7. BASİT MACD
        ema_12 = data['close'].ewm(span=12, adjust=False, min_periods=1).mean()
        ema_26 = data['close'].ewm(span=26, adjust=False, min_periods=1).mean()
        features['macd'] = ema_12 - ema_26
        features['macd_signal'] = features['macd'].ewm(span=9, adjust=False, min_periods=1).mean()
        features['macd_histogram'] = features['macd'] - features['macd_signal']
        
        # 8. BOLLINGER BANDS (20 period, min_periods=5)
        bb_middle = data['close'].rolling(window=20, min_periods=5).mean()
        bb_std = data['close'].rolling(window=20, min_periods=5).std()
        features['bb_middle'] = bb_middle
        features['bb_upper'] = bb_middle + (bb_std * 2)
        features['bb_lower'] = bb_middle - (bb_std * 2)
        features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle
        features['bb_position'] = (data['close'] - features['bb_lower']) / (features['bb_upper'] - features['bb_lower'])
        
        # 9. MOMENTUM
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        
        # Hedef değişkenler (eğitim modu için)
        if for_training:
            # Gelecek 3 mumda %1 kazanç? (daha esnek)
            future_return = data['close'].shift(-3) / data['close'] - 1
            features['target_direction'] = (future_return > 0.01).astype(int)
            features['target_return'] = future_return
        
        # NaN ve Inf temizliği - Dikkatli yap
        print(f"  🔧 Özellikler oluşturuldu: {len(features)} satır")
        
        # Önce sadece tamamen NaN olan satırları at
        features = features.dropna(how='all')
        
        # Sonra kalan NaN'ları 0 ile doldur (çok az olmalı)
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)
        
        print(f"  ✅ Temizlik sonrası: {len(features)} satır")
        
        return features
    
    def train(self, historical_data_dict):
        """Model eğitimi - BÜYÜK VERİ İÇİN OPTİMİZE"""
        print("🧠 AI Modeli eğitiliyor (Büyük veri seti)...")
        
        all_features = []
        all_targets = []
        total_samples = 0
        
        for symbol, df in historical_data_dict.items():
            try:
                print(f"\n  📊 {symbol} işleniyor...")
                features = self.create_advanced_features(df, for_training=True)
                
                if len(features) < 50:  # ← Minimum 50 örnek
                    print(f"  ⚠️ {symbol}: Yetersiz özellik verisi ({len(features)})")
                    continue
                
                if 'target_direction' not in features.columns:
                    print(f"  ⚠️ {symbol}: Hedef değişken yok")
                    continue
                
                y = features['target_direction']
                X = features.drop(['target_direction', 'target_return'], axis=1, errors='ignore')
                X = X.select_dtypes(include=[np.number])
                
                if X.sum().sum() == 0:
                    print(f"  ⚠️ {symbol}: Tüm değerler 0")
                    continue
                
                all_features.append(X)
                all_targets.append(y)
                total_samples += len(X)
                print(f"  ✅ {symbol}: {len(X)} örnek eklendi")
                    
            except Exception as e:
                print(f"  ❌ {symbol} atlandı: {e}")
        
        if not all_features or total_samples < 500:  # ← Minimum 500 örnek
            print(f"❌ Eğitim için yeterli veri yok (Toplam: {total_samples})")
            return False
        
        # Birleştir
        X = pd.concat(all_features, ignore_index=True)
        y = pd.concat(all_targets, ignore_index=True)
        
        print(f"\n📈 Toplam eğitim verisi: {len(X):,} örnek, {len(X.columns)} özellik")
        
        # Sınıf dağılımı
        class_counts = y.value_counts()
        print(f"📊 Sınıf dağılımı: {dict(class_counts)}")
        
        # Dengesizlik varsa uyarı
        min_class = class_counts.min()
        max_class = class_counts.max()
        imbalance_ratio = min_class / max_class
        print(f"⚖️ Dengesizlik oranı: %{imbalance_ratio*100:.1f}")
        
        if imbalance_ratio < 0.3:
            print("⚠️ Veri seti çok dengesiz, sonuçlar yanıltıcı olabilir")
        
        # Eğitim/test ayrımı (stratify ile dengeli)
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Ölçeklendirme
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Daha büyük veri için optimize model
        n_estimators = min(200, max(50, len(X) // 100))  # ← Veri boyutuna göre ayarla
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=12,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'  # ← Dengesiz veri için daha iyi
        )
        
        print(f"\n🤖 Model eğitiliyor ({n_estimators} ağaç)...")
        self.model.fit(X_train_scaled, y_train)
        
        # Performans
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        # Overfitting kontrolü
        overfit_gap = train_score - test_score
        print(f"\n📊 Performans:")
        print(f"   Eğitim doğruluğu: %{train_score*100:.2f}")
        print(f"   Test doğruluğu: %{test_score*100:.2f}")
        print(f"   Fark: %{overfit_gap*100:.2f} ({'⚠️ Overfitting!' if overfit_gap > 0.15 else '✅ Normal'})")
        
        # Özellik önemleri
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        self.feature_importance = importance.head(10).to_dict('records')
        self.feature_columns = list(X.columns)
        
        # Kaydet
        os.makedirs('ai_models', exist_ok=True)
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        joblib.dump(self.feature_columns, 'ai_models/feature_columns.pkl')
        
        self.is_trained = True
        
        print(f"\n📊 En önemli 5 özellik:")
        for i, row in importance.head(5).iterrows():
            print(f"   {row['feature']}: %{row['importance']*100:.2f}")
        
        return True    
    def predict(self, current_df):
        """Tahmin yap - DÜZELTİLMİŞ"""
        if not self.is_trained:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                try:
                    self.model = joblib.load(self.model_path)
                    self.scaler = joblib.load(self.scaler_path)
                    self.feature_columns = joblib.load('ai_models/feature_columns.pkl')
                    self.is_trained = True
                except Exception as e:
                    print(f"❌ Model yükleme hatası: {e}")
                    return self._default_prediction()
            else:
                return self._default_prediction()
        
        try:
            features = self.create_advanced_features(current_df, for_training=False)
            if len(features) == 0:
                return self._default_prediction()
            
            # Son satırı al
            X = features.select_dtypes(include=[np.number])
            
            # Kolon eşleştirme
            if self.feature_columns:
                for col in self.feature_columns:
                    if col not in X.columns:
                        X[col] = 0
                X = X[self.feature_columns]
            
            X_last = X.iloc[-1:].values
            X_scaled = self.scaler.transform(X_last)
            
            probability = self.model.predict_proba(X_scaled)[0][1]
            prediction = self.model.predict(X_scaled)[0]
            
            confidence = probability if prediction == 1 else (1 - probability)
            
            if confidence > 0.75:
                signal = 'GÜÇLÜ AL'
            elif confidence > 0.6:
                signal = 'AL'
            elif confidence < 0.25:
                signal = 'GÜÇLÜ SAT'
            elif confidence < 0.4:
                signal = 'SAT'
            else:
                signal = 'BEKLE'
            
            return {
                'confidence': round(confidence * 100, 2),
                'signal': signal,
                'probability': round(probability, 4),
                'prediction': int(prediction)
            }
            
        except Exception as e:
            print(f"❌ AI tahmin hatası: {e}")
            return self._default_prediction()
    
    def _default_prediction(self):
        """Varsayılan tahmin"""
        return {'confidence': 50.0, 'signal': 'BEKLE', 'probability': 0.5, 'prediction': 0}
    
    def get_feature_importance(self):
        return self.feature_importance