import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import warnings
import scipy.stats
warnings.filterwarnings('ignore')
import matplotlib.patches as mpatches
import json
from datetime import datetime, timedelta
from collections import namedtuple
from binance.client import Client
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    try:
        matplotlib.use('Qt5Agg')
    except:
        matplotlib.use('Agg')
        print("⚠️ İnteraktif grafik penceresi kullanılamıyor. Grafik dosyaya kaydedilecek.")


TIMEFRAME_LEVEL_CONFIG = {
    '1h': {
        'major': 2, 'minor': 3, 'ema': 1, 'fib': 1, 'psycho': 1, 'volume': 0, 'total': 7,
        'major_tf': '4h', 'minor_tf': '1h', 'ema_periods': [50], 'fib_lookback': 20,
        'min_tests': 2, 'tolerance': 0.005, 'lookback_factor': 0.15
    },
    '4h': {
        'major': 2, 'minor': 2, 'ema': 1, 'fib': 1, 'psycho': 1, 'volume': 0, 'total': 6,
        'major_tf': '1d', 'minor_tf': '4h', 'ema_periods': [200], 'fib_lookback': 40,
        'min_tests': 2, 'tolerance': 0.008, 'lookback_factor': 0.25
    },
    '1d': {
        'major': 2, 'minor': 1, 'ema': 1, 'fib': 1, 'psycho': 1, 'volume': 0, 'total': 5,
        'major_tf': '1w', 'minor_tf': '1d', 'ema_periods': [200], 'fib_lookback': 60,
        'min_tests': 3, 'tolerance': 0.01, 'lookback_factor': 0.4
    },
    '1w': {
        'major': 1, 'minor': 0, 'ema': 0, 'fib': 1, 'psycho': 0, 'volume': 0, 'total': 2,
        'major_tf': '1M', 'minor_tf': '1w', 'ema_periods': [], 'fib_lookback': 100,
        'min_tests': 2, 'tolerance': 0.015, 'lookback_factor': 0.6
    },
}

def calculate_adx(df, period=14):
    try:
        # True Range hesapla
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Directional Movement hesapla
        df['dm_plus'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                np.maximum(df['high'] - df['high'].shift(1), 0), 0)
        df['dm_minus'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                 np.maximum(df['low'].shift(1) - df['low'], 0), 0)
        
        # Smoothed values
        df['tr_smoothed'] = df['tr'].rolling(window=period).mean()
        df['dm_plus_smoothed'] = df['dm_plus'].rolling(window=period).mean()
        df['dm_minus_smoothed'] = df['dm_minus'].rolling(window=period).mean()
        
        # DI hesapla
        df['di_plus'] = 100 * (df['dm_plus_smoothed'] / df['tr_smoothed'])
        df['di_minus'] = 100 * (df['dm_minus_smoothed'] / df['tr_smoothed'])
        
        # DX ve ADX hesapla
        df['dx'] = 100 * abs(df['di_plus'] - df['di_minus']) / (df['di_plus'] + df['di_minus'])
        df['adx'] = df['dx'].rolling(window=period).mean()
        
        return df
        
    except Exception as e:
        print(f"❌ ADX hesaplama hatası: {str(e)}")
        return df

class CryptoAnalyzer:
    def __init__(self):
        try:
            print("🔑 Binance API bağlantısı kuruluyor...")
            
            # API kimlik bilgileri
            api_key = "****************************************************************"
            api_secret = "****************************************************************"
            
            self.client = Client(api_key, api_secret)
            self.client.ping()
            print("✅ Binance API'ye başarıyla bağlanıldı")
            
        except Exception as e:
            print(f"❌ Binance API bağlantı hatası: {str(e)}")
            raise

    def fetch_klines(self, symbol, interval, limit):
        try:
            print(f"📊 {symbol} için {interval} verisi çekiliyor...")
            
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            
            if not klines:
                print(f"❌ {symbol} için veri alınamadı")
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            # Hareketli ortalamalar
            df['EMA_50'] = df['close'].ewm(span=50).mean()
            df['EMA_200'] = df['close'].ewm(span=200).mean()
            
            # ADX
            df = calculate_adx(df, period=14)
            
            print(f"✅ {len(df)} mum verisi başarıyla alındı")
            return df
            
        except Exception as e:
            print(f"❌ Veri çekme hatası: {str(e)}")
            return None

    def get_timeframe_strategy(self, interval):
        strategies = {
            Client.KLINE_INTERVAL_1HOUR: {
                'name': '1 Saatlik (Scalping)',
                'max_levels': 5,  # 3-5 seviye - hızlı fiyat hareketleri için
                'lookback_factor': 0.15,  # Son %15'ini değerlendir
                'min_tests': 2,
                'tolerance': 0.005,  # %0.5 tolerans
                'volume_weight': 0.3,
                'psychology_weight': 0.4,
                'description': 'Hızlı Fiyat Hareketleri '
            },
            Client.KLINE_INTERVAL_4HOUR: {
                'name': '4 Saatlik (Swing Trading)', 
                'max_levels': 3,  # 3-4 seviye - orta vadeli hareketler
                'lookback_factor': 0.25,  # Son %25'ini değerlendir
                'min_tests': 2,
                'tolerance': 0.01,  # %1 tolerans
                'volume_weight': 0.4,
                'psychology_weight': 0.3,
                'description': 'Yarı-Uzun Vadeli '
            },
            Client.KLINE_INTERVAL_1DAY: {
                'name': '1 Günlük (Orta-Uzun Vade)',
                'max_levels': 3,  # 2-3 seviye - major seviyeler
                'lookback_factor': 0.4,  # Son %40'ını değerlendir
                'min_tests': 3,
                'tolerance': 0.015,  # %1.5 tolerans
                'volume_weight': 0.5,
                'psychology_weight': 0.2,
                'description': 'Günlük Trend'
            },
            Client.KLINE_INTERVAL_1WEEK: {
                'name': '1 Haftalık (Uzun Vade)',
                'max_levels': 2,  # 2 seviye - tarihi major + Fibonacci 100%
                'lookback_factor': 0.6,  # Son %60'ını değerlendir
                'min_tests': 2,
                'tolerance': 0.02,  # %2 tolerans
                'volume_weight': 0.6,
                'psychology_weight': 0.1,
                'description': 'Uzun Vadeli Analiz'
            }
        }
        return strategies.get(interval, strategies[Client.KLINE_INTERVAL_1DAY])

    def find_psychological_levels(self, price_range):
        psychological_levels = []
        min_price, max_price = min(price_range), max(price_range)
        
        if max_price > 100000:
            step = 10000  # 10K aralıkları
        elif max_price > 10000:
            step = 1000   # 1K aralıkları
        elif max_price > 1000:
            step = 100    # 100 aralıkları
        elif max_price > 100:
            step = 10     # 10 aralıkları
        else:
            step = 1      # 1 aralıkları
        
        start = int(min_price // step) * step
        end = int(max_price // step + 1) * step
        
        for level in range(start, end + step, step):
            if min_price <= level <= max_price and level > 0:
                psychological_levels.append(float(level))
        
        return psychological_levels

    def analyze_trend(self, df, interval):
        """3 Kritik Gösterge ile Entegre Trend Analizi"""
        try:
            # Son 20 mum verisi için trend analizi
            recent_data = df.tail(20)
            current_price = recent_data['close'].iloc[-1]
            
            # 1. HAREKETLİ ORTALAMA (MA) ANALİZİ
            ma_trend = self._analyze_ma_trend(recent_data, current_price)
            
            # 2. SWING HIGH/LOW ANALİZİ
            swing_trend = self._analyze_swing_trend(recent_data)
            
            # 3. ADX ANALİZİ
            adx_trend = self._analyze_adx_trend(recent_data)
            
            # ENTEGRE TREND ANALİZİ - Oylama Sistemi
            final_trend = self._integrate_trend_signals(ma_trend, swing_trend, adx_trend, df)
            
            return final_trend
            
        except Exception as e:
            print(f"❌ Trend analizi hatası: {str(e)}")
            return {
                'direction': "BELİRSİZ",
                'strength': "BELİRSİZ",
                'ma_trend': "BELİRSİZ",
                'swing_trend': "BELİRSİZ", 
                'adx_trend': "BELİRSİZ",
                'vote_count': 0,
                'price_change': 0,
                'adx_value': 0,
                'di_plus': 0,
                'di_minus': 0
            }

    def _analyze_ma_trend(self, recent_data, current_price):
        try:
            ema_50 = recent_data['EMA_50'].iloc[-1]
            ema_200 = recent_data['EMA_200'].iloc[-1]
            
            if current_price > ema_50 > ema_200:
                return "GÜÇLÜ YÜKSELİŞ"
            elif ema_50 > current_price > ema_200:
                return "ZAYIF YÜKSELİŞ"
            elif current_price < ema_50 < ema_200:
                return "GÜÇLÜ DÜŞÜŞ"
            elif ema_50 < current_price < ema_200:
                return "ZAYIF DÜŞÜŞ"
            else:
                return "YATAY"
                
        except:
            return "BELİRSİZ"

    def _analyze_swing_trend(self, recent_data):
        """Swing High/Low Analizi"""
        try:
            # Son 10 mum için swing analizi (daha az volatilite)
            swing_data = recent_data.tail(10)
            highs = swing_data['high'].values
            lows = swing_data['low'].values
            
            # Higher Highs ve Lower Lows sayısı
            higher_highs = sum(1 for i in range(1, len(highs)) if highs[i] > highs[i-1])
            lower_lows = sum(1 for i in range(1, len(lows)) if lows[i] < lows[i-1])
            
            # Trend Kuralları
            if higher_highs >= 7 and lower_lows < 3:
                return "YÜKSELİŞ"
            elif lower_lows >= 7 and higher_highs < 3:
                return "DÜŞÜŞ"
            else:
                return "YATAY"
                
        except:
            return "BELİRSİZ"

    def _analyze_adx_trend(self, recent_data):
        """ADX Analizi"""
        try:
            adx_value = recent_data['adx'].iloc[-1]
            di_plus = recent_data['di_plus'].iloc[-1]
            di_minus = recent_data['di_minus'].iloc[-1]
            
            # NaN kontrolü
            if pd.isna(adx_value) or pd.isna(di_plus) or pd.isna(di_minus):
                return "BELİRSİZ"
            
            # Trend Kuralları (eşik 20)
            if adx_value > 20:
                if di_plus > di_minus:
                    return "GÜÇLÜ YÜKSELİŞ"
                else:
                    return "GÜÇLÜ DÜŞÜŞ"
            else:
                return "YATAY"
                
        except:
            return "BELİRSİZ"

    def _simplify_trend_direction(self, trend):
        if "YÜKSELİŞ" in trend:
            return "YÜKSELİŞ"
        elif "DÜŞÜŞ" in trend:
            return "DÜŞÜŞ"
        elif "YATAY" in trend:
            return "YATAY"
        elif "ZAYIF YÜKSELİŞ" in trend:
            return "YÜKSELİŞ"
        elif "ZAYIF DÜŞÜŞ" in trend:
            return "DÜŞÜŞ"
        else:
            return "BELİRSİZ"

    def _integrate_trend_signals(self, ma_trend, swing_trend, adx_trend, df):
        try:
            price_change = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20] * 100
            adx_value = df['adx'].iloc[-1] if not pd.isna(df['adx'].iloc[-1]) else 0
            di_plus = df['di_plus'].iloc[-1] if not pd.isna(df['di_plus'].iloc[-1]) else 0
            di_minus = df['di_minus'].iloc[-1] if not pd.isna(df['di_minus'].iloc[-1]) else 0

            ma_direction = self._simplify_trend_direction(ma_trend)
            swing_direction = self._simplify_trend_direction(swing_trend)
            adx_direction = self._simplify_trend_direction(adx_trend)

            votes = {'YÜKSELİŞ': 0, 'DÜŞÜŞ': 0, 'YATAY': 0}
            for d in [ma_direction, swing_direction, adx_direction]:
                if d != "BELİRSİZ":
                    votes[d] += 1

            # Geliştirilmiş oylama sistemi - Fiyat değişimini de dikkate al
            # Eğer fiyat değişimi çok küçükse (yatay hareket), trend yatay olarak değerlendir
            if abs(price_change) < 1.0:  # %1'den az değişim = yatay
                final_direction = "YATAY"
                strength = "ZAYIF"
            # Eğer fiyat değişimi belirginse, gösterge oylarını kullan
            elif votes['YÜKSELİŞ'] >= 2 and votes['DÜŞÜŞ'] == 0:
                final_direction = "YÜKSELİŞ"
                strength = "GÜÇLÜ" if votes['YÜKSELİŞ'] == 3 else "ORTA"
            elif votes['DÜŞÜŞ'] >= 2 and votes['YÜKSELİŞ'] == 0:
                final_direction = "DÜŞÜŞ"
                strength = "GÜÇLÜ" if votes['DÜŞÜŞ'] == 3 else "ORTA"
            elif votes['YATAY'] >= 2:
                final_direction = "YATAY"
                strength = "ZAYIF"
            elif votes['YÜKSELİŞ'] == 1 and votes['DÜŞÜŞ'] == 0 and price_change > 0.5:
                final_direction = "YÜKSELİŞ"
                strength = "ZAYIF"
            elif votes['DÜŞÜŞ'] == 1 and votes['YÜKSELİŞ'] == 0 and price_change < -0.5:
                final_direction = "DÜŞÜŞ"
                strength = "ZAYIF"
            else:
                final_direction = "YATAY"
                strength = "ZAYIF"

            vote_count = sum(1 for v in votes.values() if v > 0)

            return {
                'direction': final_direction,
                'strength': strength,
                'ma_trend': ma_trend,
                'swing_trend': swing_trend,
                'adx_trend': adx_trend,
                'vote_count': vote_count,
                'price_change': price_change,
                'adx_value': adx_value,
                'di_plus': di_plus,
                'di_minus': di_minus
            }
        except Exception as e:
            print(f"❌ Trend entegrasyon hatası: {str(e)}")
            return {
                'direction': "BELİRSİZ",
                'strength': "BELİRSİZ",
                'ma_trend': ma_trend,
                'swing_trend': swing_trend,
                'adx_trend': adx_trend,
                'vote_count': 0,
                'price_change': 0,
                'adx_value': 0,
                'di_plus': 0,
                'di_minus': 0
            }

    def calculate_fibonacci_levels(self, df, lookback_ratio=0.3):
        """Fibonacci retracement seviyeleri"""
        lookback_period = int(len(df) * lookback_ratio)
        recent_data = df.tail(lookback_period)
        
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Fibonacci seviyeleri
        fib_levels = {
            'fib_23.6': swing_high - (swing_high - swing_low) * 0.236,
            'fib_38.2': swing_high - (swing_high - swing_low) * 0.382,
            'fib_50.0': swing_high - (swing_high - swing_low) * 0.5,
            'fib_61.8': swing_high - (swing_high - swing_low) * 0.618,
            'fib_78.6': swing_high - (swing_high - swing_low) * 0.786,
            'fib_100.0': swing_low
        }
        
        return fib_levels, swing_high, swing_low

    def prioritize_levels(self, levels, interval, current_price, trend_analysis):
        """Seviyeleri önem sırasına göre filtrele"""
        try:
            if not levels:
                return []
            
            # Seviye kategorilerini belirle
            major_levels = []
            dynamic_levels = []
            secondary_levels = []
            
            for level in levels:
                # Major seviyeler (psikolojik + güçlü test sayısı)
                if (level['type'] == 'Psikolojik Destek' or level['type'] == 'Psikolojik Direnç' or
                    level['test_count'] >= 4):
                    level['priority'] = 'MAJOR'
                    level['priority_score'] = level['test_count'] * 3 + 10
                    major_levels.append(level)
                
                # Dinamik seviyeler (EMA + Fibonacci)
                elif ('EMA' in level['type'] or 'Fibonacci' in level['type']):
                    level['priority'] = 'DYNAMIC'
                    level['priority_score'] = level['test_count'] * 2 + 5
                    dynamic_levels.append(level)
                
                # İkincil seviyeler (teknik + zayıf test)
                else:
                    level['priority'] = 'SECONDARY'
                    level['priority_score'] = level['test_count'] * 1
                    secondary_levels.append(level)
            
            # Zaman dilimine göre seviye seçimi
            strategy = self.get_timeframe_strategy(interval)
            max_levels = strategy['max_levels']
            
            selected_levels = []
            
            # 1. Major seviyeler
            selected_levels.extend(major_levels[:min(2, max_levels)])
            
            # 2. Dinamik seviyeler
            remaining_slots = max_levels - len(selected_levels)
            if remaining_slots > 0:
                selected_levels.extend(dynamic_levels[:remaining_slots])
            
            # 3. İkincil seviyeler (kalan slot varsa)
            remaining_slots = max_levels - len(selected_levels)
            if remaining_slots > 0:
                selected_levels.extend(secondary_levels[:remaining_slots])
            
            # Güce göre sıralama ve en iyilerini seçme
            selected_levels.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Çakışan seviyeleri birleştir
            final_levels = self._merge_overlapping_levels(selected_levels, current_price)
            
            # --- Destek ve dirençlerde distance anahtarı eksikse ekle ---
            for level in final_levels:
                if 'distance' not in level:
                    level['distance'] = abs(level['price'] - current_price) / current_price
            
            return final_levels[:max_levels]
            
        except Exception as e:
            print(f"❌ Seviye önceliklendirme hatası: {str(e)}")
            return levels[:strategy['max_levels']] if levels else []

    def _merge_overlapping_levels(self, levels, current_price):
        """Çakışan seviyeleri birleştir"""
        if not levels:
            return []
        
        merged_levels = []
        
        for level in levels:
            is_overlapping = False
            
            for existing in merged_levels:
                # %1'den yakın seviyeleri birleştir
                price_diff = abs(level['price'] - existing['price']) / existing['price']
                if price_diff < 0.01:
                    # Daha güçlü olanı tut
                    if level['priority_score'] > existing['priority_score']:
                        merged_levels[merged_levels.index(existing)] = level
                    is_overlapping = True
                    break
            
            if not is_overlapping:
                merged_levels.append(level)
        
        return merged_levels

    def _merge_close_levels(self, levels, min_diff=0.01):
        """Birbirine çok yakın seviyeleri birleştirme"""
        if not levels:
            return []
        merged = []
        for level in sorted(levels, key=lambda x: (x['distance'], -x.get('test_count', 1), -x.get('strength', 0))):
            too_close = False
            for m in merged:
                if abs(level['price'] - m['price']) / m['price'] < min_diff:
                    too_close = True
                    # Güçlü olanı tut
                    if (level.get('test_count', 1), level.get('strength', 0)) > (m.get('test_count', 1), m.get('strength', 0)):
                        merged[merged.index(m)] = level
                    break
            if not too_close:
                merged.append(level)
        return merged

    def _find_major_levels(self, df, current_price, n=2, lookback=100):
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        
        # Swing high/low'ları bul
        swing_highs = []
        swing_lows = []
        
        for i in range(1, len(highs)-1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                swing_highs.append(highs[i])
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                swing_lows.append(lows[i])
        
        # En önemli n seviyeyi seçme (fiyata yakınlık ve test sayısına göre)
        def get_top_levels(levels, current_price, n):
            levels = sorted(set(levels), key=lambda x: (abs(x - current_price), len([p for p in levels if abs(p-x)/x < 0.01])))
            return sorted(levels[:n])
        
        major_highs = get_top_levels(swing_highs, current_price, n)
        major_lows = get_top_levels(swing_lows, current_price, n)
        
        return major_lows, major_highs

    def _find_minor_levels(self, df, current_price, n=3, lookback=50, tolerance=0.01):
        highs = df['high'].values[-lookback:]
        lows = df['low'].values[-lookback:]
        
        # Yerel minimum/maksimumları bul
        minor_lows = []
        minor_highs = []
        
        for i in range(1, len(highs)-1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                test_count = sum(1 for p in lows if abs(p - lows[i])/lows[i] < tolerance)
                if test_count >= 2:  # En az 2 test
                    minor_lows.append(lows[i])
            
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                test_count = sum(1 for p in highs if abs(p - highs[i])/highs[i] < tolerance)
                if test_count >= 2:  # En az 2 test
                    minor_highs.append(highs[i])
        
        # Fiyata yakınlığa göre sırala ve n tane seç
        minor_lows = sorted(set(minor_lows), key=lambda x: abs(x - current_price))[:n]
        minor_highs = sorted(set(minor_highs), key=lambda x: abs(x - current_price))[:n]
        
        return minor_lows, minor_highs

    def _add_level(self, container, price, level_type, priority, current_price):
        distance = abs(price - current_price)/current_price
        test_count = 3 if priority == 'MAJOR' else 2 if priority in ['MINOR', 'FIB'] else 1
        strength = 10 if priority == 'MAJOR' else 6 if priority == 'MINOR' else 4
        
        container.append({
            'price': price,
            'type': level_type,
            'priority': priority,
            'distance': distance,
            'test_count': test_count,
            'strength': strength
        })

    def _find_weekly_levels(self, df, current_price):
        supports, resistances = [], []
        
        try:
            # 1. Tarihi Major Destek/Direnç seviyeleri
            # Son 100 haftalık veriden en önemli swing high/low'ları bul
            lookback = min(100, len(df))
            recent_data = df.tail(lookback)
            
            # Swing high/low'ları bul
            highs = recent_data['high'].values
            lows = recent_data['low'].values
            
            swing_highs = []
            swing_lows = []
            
            # Swing high/low tespiti (en az 3 bar aralıklı)
            for i in range(2, len(highs)-2):
                # Swing high
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    swing_highs.append(highs[i])
                # Swing low
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    swing_lows.append(lows[i])
            
            # En önemli seviyeleri seç (fiyata yakınlık ve test sayısına göre)
            if swing_lows:
                # En yakın 2 destek seviyesi
                swing_lows = sorted(swing_lows, key=lambda x: abs(x - current_price))[:2]
                for price in swing_lows:
                    if price < current_price:  # Sadece destek seviyeleri
                        self._add_level(supports, price, 'Tarihi Major Destek', 'MAJOR', current_price)
            
            if swing_highs:
                # En yakın 2 direnç seviyesi
                swing_highs = sorted(swing_highs, key=lambda x: abs(x - current_price))[:2]
                for price in swing_highs:
                    if price > current_price:  # Sadece direnç seviyeleri
                        self._add_level(resistances, price, 'Tarihi Major Direnç', 'MAJOR', current_price)
            
            # 2. Fibonacci 100% Retracement seviyesi
            fib_levels, swing_high, swing_low = self.calculate_fibonacci_levels(df, 0.6)
            if 'fib_100.0' in fib_levels:
                fib_100_price = fib_levels['fib_100.0']
                if fib_100_price < current_price:
                    self._add_level(supports, fib_100_price, 'Fibonacci 100% Retracement', 'FIB', current_price)
                else:
                    self._add_level(resistances, fib_100_price, 'Fibonacci 100% Retracement', 'FIB', current_price)
            
            # Eğer hala seviye bulunamadıysa, basit destek/direnç seviyeleri ekle
            if not supports and not resistances:
                # Son 20 haftalık veriden basit seviyeler
                simple_data = df.tail(20)
                min_price = simple_data['low'].min()
                max_price = simple_data['high'].max()
                
                if min_price < current_price:
                    self._add_level(supports, min_price, 'Basit Destek', 'MINOR', current_price)
                if max_price > current_price:
                    self._add_level(resistances, max_price, 'Basit Direnç', 'MINOR', current_price)
            
            print(f"✅ 1 Haftalık seviye tespiti: {len(supports)} destek, {len(resistances)} direnç")
            
        except Exception as e:
            print(f"❌ 1 Haftalık seviye tespiti hatası: {str(e)}")
            # Hata durumunda basit seviyeler ekle
            if not supports:
                self._add_level(supports, current_price * 0.9, 'Acil Destek', 'MINOR', current_price)
            if not resistances:
                self._add_level(resistances, current_price * 1.1, 'Acil Direnç', 'MINOR', current_price)
        
        return supports, resistances



    def _filter_and_sort_levels(self, levels, current_price, config):
        if not levels:
            return []
        
        # 1. Çakışan seviyeleri birleştir (%1 tolerans)
        levels = self._merge_close_levels(levels, min_diff=config['tolerance'])
        
        # 2. Test sayısına göre filtrele (1 haftalık grafiklerde daha esnek)
        tf_key = None
        if hasattr(self, 'current_interval'):
            tf_map = {
                Client.KLINE_INTERVAL_1HOUR: '1h',
                Client.KLINE_INTERVAL_4HOUR: '4h',
                Client.KLINE_INTERVAL_1DAY: '1d',
                Client.KLINE_INTERVAL_1WEEK: '1w',
            }
            tf_key = tf_map.get(self.current_interval, '1d')
        
        # 1 haftalık grafiklerde daha esnek filtreleme
        if tf_key == '1w':
            levels = [lvl for lvl in levels if lvl['test_count'] >= 1 or lvl['priority'] in ['MAJOR', 'FIB', 'MINOR']]
        else:
            levels = [lvl for lvl in levels if lvl['test_count'] >= config['min_tests'] or lvl['priority'] in ['MAJOR', 'FIB', 'PSYCHO']]
        
        # 3. Öncelik sırasına göre sıralama (MAJOR > MINOR > DYNAMIC > FIB > PSYCHO)
        priority_order = {'MAJOR': 0, 'MINOR': 1, 'DYNAMIC': 2, 'FIB': 3, 'PSYCHO': 4}
        levels.sort(key=lambda x: (priority_order.get(x['priority'], 5), x['distance']))
        
        # 4. Maksimum seviye sayısını uygulama
        return levels[:config['total']]

    def find_timeframe_optimized_levels(self, df, interval, symbol=None):
        tf_map = {
            Client.KLINE_INTERVAL_1HOUR: '1h',
            Client.KLINE_INTERVAL_4HOUR: '4h',
            Client.KLINE_INTERVAL_1DAY: '1d',
            Client.KLINE_INTERVAL_1WEEK: '1w',
        }
        tf_key = tf_map.get(interval, '1d')
        config = TIMEFRAME_LEVEL_CONFIG[tf_key]
        current_price = df['close'].iloc[-1]
        
        supports, resistances = [], []
        
        # 1 haftalık grafikler için özel seviye tespiti
        if tf_key == '1w':
            supports, resistances = self._find_weekly_levels(df, current_price)
        else:
            # 1. Major seviyeler (üst zaman diliminden)
            upper_tf_map = {
                '1h': Client.KLINE_INTERVAL_4HOUR,
                '4h': Client.KLINE_INTERVAL_1DAY,
                '1d': Client.KLINE_INTERVAL_1WEEK,
                '1w': Client.KLINE_INTERVAL_1WEEK,
            }
            
            if tf_key != '1w':
                fetch_symbol = symbol if symbol is not None else 'BTCUSDT'
                upper_df = self.fetch_klines(fetch_symbol, upper_tf_map[tf_key], 100)
                if upper_df is not None:
                    major_lows, major_highs = self._find_major_levels(upper_df, current_price, 
                                                                    n=config['major'],
                                                                    lookback=100)
                    for price in major_lows:
                        self._add_level(supports, price, 'Major Destek', 'MAJOR', current_price)
                    for price in major_highs:
                        self._add_level(resistances, price, 'Major Direnç', 'MAJOR', current_price)
            
            # 2. Minor seviyeler (mevcut zaman diliminden)
            minor_lows, minor_highs = self._find_minor_levels(df, current_price,
                                                             n=config['minor'],
                                                             lookback=50,
                                                             tolerance=config['tolerance'])
            for price in minor_lows:
                self._add_level(supports, price, 'Minor Destek', 'MINOR', current_price)
            for price in minor_highs:
                self._add_level(resistances, price, 'Minor Direnç', 'MINOR', current_price)
            
            # 3. EMA seviyeleri (1 haftalık grafiklerde kullanılmaz)
            if tf_key != '1w':
                for period in config['ema_periods']:
                    ema_col = f'EMA_{period}'
                    if ema_col in df.columns:
                        ema_value = df[ema_col].iloc[-1]
                        level_type = f'EMA {period}'
                        if ema_value < current_price:
                            self._add_level(supports, ema_value, level_type, 'DYNAMIC', current_price)
                        else:
                            self._add_level(resistances, ema_value, level_type, 'DYNAMIC', current_price)
            
            # 5. Psikolojik seviyeler
            if config['psycho'] > 0 and tf_key != '1w':
                psycho_levels = self.find_psychological_levels([df['low'].min(), df['high'].max()])
                for level in psycho_levels:
                    if level < current_price:
                        self._add_level(supports, level, 'Psikolojik Destek', 'PSYCHO', current_price)
                    else:
                        self._add_level(resistances, level, 'Psikolojik Direnç', 'PSYCHO', current_price)
        
        # Seviyeleri önem sırasına göre filtreleme
        supports = self._filter_and_sort_levels(supports, current_price, config)
        resistances = self._filter_and_sort_levels(resistances, current_price, config)
        
        return supports, resistances

    def _count_level_tests(self, prices, target_price, tolerance=0.015):
        """Bir seviyenin kaç kez test edildiğini say"""
        count = 0
        for price in prices:
            if abs(price - target_price) / target_price <= tolerance:
                count += 1
        return count

    def plot_advanced_analysis(self, symbol, df, supports, resistances, strategy, interval):
        try:
            # Mevcut grafikleri temizle
            plt.close('all')
            
            # 1 haftalık grafikler için basit grafik
            if interval == Client.KLINE_INTERVAL_1WEEK:
                print("📈 1 Haftalık grafik oluşturuluyor...")
                return self._plot_weekly_analysis(symbol, df, supports, resistances, strategy)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12), 
                                         gridspec_kw={'height_ratios': [3, 1]})
            
            current_price = df['close'].iloc[-1]
            
            # Trend analizi yap
            trend_analysis = self.analyze_trend(df, interval)
            
            # Fibonacci seviyelerini hesapla
            fib_levels, swing_high, swing_low = self.calculate_fibonacci_levels(df, strategy['lookback_factor'])

            # --- Yükselen ve Düşen Trend Çizgileri ---
            trend_lookback = min(50, len(df))
            trend_result = trend_analysis['direction'] if trend_analysis else 'YATAY'

            # Trend çizgisi sadece belirgin trendler için çizilsin
            if trend_result == 'YÜKSELİŞ' and trend_analysis['strength'] in ['GÜÇLÜ', 'ORTA']:
                x_up, trend_up, slope_up = self.find_trend_lines(df, trend_type='up', lookback=trend_lookback)
                if x_up is not None and trend_up is not None and len(trend_up) == trend_lookback and slope_up > 0:
                    linewidth = 2.5 if abs(slope_up) > 0.01 else 1.5
                    ax1.plot(df['timestamp'].iloc[-trend_lookback:], trend_up, color='#32CD32', linestyle='-', linewidth=linewidth, label='Yükselen Trend')
                    # Ok ekle
                    ax1.annotate('', xy=(df['timestamp'].iloc[-1], trend_up[-1]),
                                 xytext=(df['timestamp'].iloc[-2], trend_up[-2]),
                                 arrowprops=dict(facecolor='green', arrowstyle='->'))
            elif trend_result == 'DÜŞÜŞ' and trend_analysis['strength'] in ['GÜÇLÜ', 'ORTA']:
                x_dn, trend_dn, slope_dn = self.find_trend_lines(df, trend_type='down', lookback=trend_lookback)
                if x_dn is not None and trend_dn is not None and len(trend_dn) == trend_lookback and slope_dn < 0:
                    linewidth = 2.5 if abs(slope_dn) > 0.01 else 1.5
                    ax1.plot(df['timestamp'].iloc[-trend_lookback:], trend_dn, color='#FF6347', linestyle='-', linewidth=linewidth, label='Düşen Trend')
                    # Ok ekle
                    ax1.annotate('', xy=(df['timestamp'].iloc[-1], trend_dn[-1]),
                                 xytext=(df['timestamp'].iloc[-2], trend_dn[-2]),
                                 arrowprops=dict(facecolor='red', arrowstyle='->'))
            # YATAY trend için trend çizgisi çizilmez

            legend_items = [
                ("#008000", "Major Destek"),
                ("#B22222", "Major Direnç"),
                ("#32CD32", "Minor Destek"),
                ("#FF6347", "Minor Direnç"),
                ("#FFD700", "Psikolojik"),
            ]
            
            if interval == Client.KLINE_INTERVAL_1HOUR:
                legend_items.append(("#8A2BE2", "EMA 50"))
            elif interval == Client.KLINE_INTERVAL_4HOUR:
                legend_items.append(("#8A2BE2", "EMA 200"))
            elif interval == Client.KLINE_INTERVAL_1DAY:
                legend_items.append(("#8A2BE2", "EMA 200"))
            # 1 haftalık grafiklerde EMA yok
            
            legend_items.append(("#FFFFFF", "Volume"))
            if interval == Client.KLINE_INTERVAL_1HOUR:
                legend_items.append(("#4ECDC4", "Fibonacci 38.2%"))
            if interval == Client.KLINE_INTERVAL_4HOUR:
                legend_items.append(("#FFA07A", "Fibonacci 61.8%"))
            if interval == Client.KLINE_INTERVAL_1DAY:
                legend_items.append(("#20B2AA", "Fibonacci 78.6%"))
            if interval == Client.KLINE_INTERVAL_1WEEK:
                legend_items.append(("#FF1493", "Fibonacci 100%"))
            x0 = 0.08
            y0 = 0.32
            dx = 0.07
            box_w = 0.014
            box_h = 0.014
            max_items_per_row = 99
            for i, (color, label) in enumerate(legend_items):
                row = i // max_items_per_row
                col = i % max_items_per_row
                legend_x = x0 + col * dx
                legend_y = y0 - row * (box_h + 0.015)
                fig.patches.append(mpatches.Rectangle((legend_x, legend_y), box_w, box_h, transform=fig.transFigure, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.7, zorder=10))
                fig.text(legend_x + box_w + 0.003, legend_y + box_h/2, label, fontsize=7, va='center', ha='left', color='black', alpha=0.7)

            fig.text(
                x0 + len(legend_items)*dx + 0.01,  
                y0 + box_h/2,                      
                f"Trend: {trend_result.capitalize()}",
                fontsize=7,
                va='center',
                ha='left',
                color="green" if trend_result.upper() == "YÜKSELİŞ" else "red",  
                alpha=0.9,
                transform=fig.transFigure          
            )

            # Ana fiyat grafiği
            ax1.plot(df['timestamp'], df['close'], 
                    label='Kapanış Fiyatı', linewidth=2, color='#2E86AB', alpha=0.8)

            # --- Otomatik yakınlaştırma: fiyat ve en yakın destek/direnç etrafında ---
            # all_prices = [current_price]
            # if supports:
            #     all_prices.append(min(s['price'] for s in supports))
            # if resistances:
            #     all_prices.append(max(r['price'] for r in resistances))
            # min_f = min(all_prices)
            # max_f = max(all_prices)
            # margin = (max_f - min_f) * 0.05
            # ax1.set_ylim(min_f - margin, max_f + margin)

            # --- YENİ: Son 100 bar içindeki min/max fiyat + %5 marj ile sınırla ---
            min_f = df['low'].iloc[-100:].min() if len(df) >= 100 else df['low'].min()
            max_f = df['high'].iloc[-100:].max() if len(df) >= 100 else df['high'].max()
            margin = (max_f - min_f) * 0.05
            ax1.set_ylim(min_f - margin, max_f + margin)
            
            # EMA çizgileri artık her zaman dilimi için ayrı ayrı tanımlanıyor
            
            # Fibonacci seviyelerini çizme
            fib_colors = {
                'fib_23.6': '#FF6B6B',
                'fib_38.2': '#4ECDC4',
                'fib_50.0': '#45B7D1',
                'fib_61.8': '#FFA07A',
                'fib_78.6': '#20B2AA',
                'fib_100.0': '#FF1493',
            }
            # Fibonacci seviyelerini çiz (1 saatlik grafikte fib_38.2, 4 saatlik grafikte fib_61.8, 1 günlük grafikte fib_78.6 düz çizgi)
            for fib_name, fib_price in fib_levels.items():
                color = fib_colors.get(fib_name, '#CCCCCC')
                # 1 saatlik grafikte fib_38.2, 4 saatlik grafikte fib_61.8, 1 günlük grafikte fib_78.6 düz çizgi, diğerleri kesik kesik
                if (interval == Client.KLINE_INTERVAL_1HOUR and fib_name == 'fib_38.2') or \
                   (interval == Client.KLINE_INTERVAL_4HOUR and fib_name == 'fib_61.8') or \
                   (interval == Client.KLINE_INTERVAL_1DAY and fib_name == 'fib_78.6'):
                    linestyle = '-'
                    linewidth = 2.0
                    alpha = 0.8
                else:
                    linestyle = (0, (4, 4))
                    linewidth = 1.0
                    alpha = 0.6
                ax1.axhline(y=fib_price, color=color, linestyle=linestyle, 
                           linewidth=linewidth, alpha=alpha)
                # Fibonacci etiketleri 
                if fib_price < current_price:
                    ax1.text(df['timestamp'].iloc[5], fib_price, 
                            f'{fib_name}: ${fib_price:.2f}', 
                            va='top', ha='left', fontsize=7, color=color,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
                else:
                    ax1.text(df['timestamp'].iloc[5], fib_price, 
                            f'{fib_name}: ${fib_price:.2f}', 
                            va='bottom', ha='left', fontsize=7, color=color,
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.7))
            # EMA çizgileri
            if interval == Client.KLINE_INTERVAL_1HOUR:
                if 'EMA_50' in df.columns:
                    ema_50_value = df['EMA_50'].iloc[-1]
                    ax1.axhline(y=ema_50_value, color='#8A2BE2', linestyle='-', 
                               linewidth=1.5, alpha=0.9, label='EMA 50')
                    # EMA 50 etiketi - çizgiye temas eden
                    ax1.text(df['timestamp'].iloc[-3], ema_50_value + (max_f - min_f) * 0.002, 'EMA 50',
                             va='bottom', ha='center', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='#8A2BE2', alpha=0.18))
            if interval == Client.KLINE_INTERVAL_4HOUR:
                if 'EMA_200' in df.columns:
                    ema_200_value = df['EMA_200'].iloc[-1]
                    ax1.axhline(y=ema_200_value, color='#8A2BE2', linestyle='-', 
                               linewidth=1.5, alpha=0.9, label='EMA 200')
                    # EMA 200 etiketi - çizgiye temas eden
                    ax1.text(df['timestamp'].iloc[-3], ema_200_value + (max_f - min_f) * 0.002, 'EMA 200',
                             va='bottom', ha='center', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='#8A2BE2', alpha=0.18))
            if interval == Client.KLINE_INTERVAL_1DAY:
                if 'EMA_200' in df.columns:
                    ema_200_value = df['EMA_200'].iloc[-1]
                    ax1.axhline(y=ema_200_value, color='#8A2BE2', linestyle='-', 
                               linewidth=1.5, alpha=0.9, label='EMA 200')
                    # EMA 200 etiketi - çizgiye temas eden
                    ax1.text(df['timestamp'].iloc[-3], ema_200_value + (max_f - min_f) * 0.002, 'EMA 200',
                             va='bottom', ha='center', fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor='#8A2BE2', alpha=0.18))
            # 1 Haftalık grafiklerde EMA çizgisi kullanılmıyor


            
            # --- Destek çizgileri ---
            for i, support in enumerate(supports):
                label = None
                if support.get('priority') == 'MAJOR':
                    color = '#008000'  # Koyu yeşil
                    linewidth = 4.0
                    linestyle = '-'
                    label = 'Major Destek'
                elif 'Minor' in support['type']:
                    color = '#32CD32'  # Açık yeşil
                    linewidth = 2.0
                    linestyle = '-'
                    label = 'Minor Destek'
                elif 'Psikolojik' in support['type']:
                    color = '#FFD700'  # Sarı
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'Psikolojik'
                elif 'EMA 50' in support['type']:
                    color = '#1E90FF'  # Mavi
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'EMA 50'
                elif 'EMA 200' in support['type']:
                    color = '#808080'  # Gri
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'EMA 200'
                elif 'Hacimsel' in support['type']:
                    color = '#FFFFFF'  # Beyaz
                    linewidth = 3.0
                    linestyle = '-'
                    label = 'Volume Profile'
                else:
                    color = '#90EE90'  # Yedek açık yeşil
                    linewidth = 2.0
                    linestyle = '-'
                    label = support['type']
                ax1.axhline(y=support['price'], color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
                ax1.text(df['timestamp'].iloc[-3], support['price'], f'{label}',
                         va='top', ha='center', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.18))
            
            # --- Direnç çizgileri ---
            for i, resistance in enumerate(resistances):
                label = None
                if resistance.get('priority') == 'MAJOR':
                    color = '#B22222'  # Koyu kırmızı
                    linewidth = 4.0
                    linestyle = '-'
                    label = 'Major Direnç'
                elif 'Minor' in resistance['type']:
                    color = '#FF6347'  # Açık kırmızı
                    linewidth = 2.0
                    linestyle = '-'
                    label = 'Minor Direnç'
                elif 'Psikolojik' in resistance['type']:
                    color = '#FFD700'  # Sarı
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'Psikolojik'
                elif 'EMA 50' in resistance['type']:
                    color = '#1E90FF'  # Mavi
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'EMA 50'
                elif 'EMA 200' in resistance['type']:
                    color = '#808080'  # Gri
                    linewidth = 2.5
                    linestyle = '-'
                    label = 'EMA 200'
                elif 'Hacimsel' in resistance['type']:
                    color = '#FFFFFF'  # Beyaz
                    linewidth = 3.0
                    linestyle = '-'
                    label = 'Volume Profile'
                else:
                    color = '#FFB6C1'  # Yedek açık kırmızı
                    linewidth = 2.0
                    linestyle = '-'
                    label = resistance['type']
                ax1.axhline(y=resistance['price'], color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.9)
                ax1.text(df['timestamp'].iloc[-3], resistance['price'], f'{label}',
                         va='bottom', ha='center', fontsize=8, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.18))
            
            # Güncel fiyat çizgisi
            ax1.axhline(y=current_price, color='blue', linestyle=':', 
                       linewidth=2, alpha=0.8, label=f'Güncel: ${current_price:.2f}')
            
            # Grafik düzenlemesi
            ax1.set_title(f'{symbol} - {strategy["name"]} Analizi\n'
                         f'Güncel Fiyat: ${current_price:.2f} | {strategy["description"]}', 
                         fontsize=13, fontweight='bold', pad=20)
            ax1.set_ylabel('Fiyat (USDT)', fontsize=11)
            ax1.grid(True, alpha=0.3)
            ax1.legend(loc='upper left', fontsize=9, framealpha=0.8)
            
            # Hacim grafiği
            ax2.bar(df['timestamp'], df['volume'], alpha=0.6, color='gray', width=0.8)
            ax2.set_ylabel('Hacim', fontsize=9)
            ax2.set_xlabel('Tarih', fontsize=10)
            ax2.grid(True, alpha=0.3)
            
            # X ekseni etiketleri
            plt.setp(ax2.get_xticklabels(), rotation=45, fontsize=8)
            
            # Analiz özet tablosu
            summary_lines = []
            summary_lines.append(f"📊 {strategy['name'].upper()} ANALİZ ÖZETİ")
            summary_lines.append("-" * 35)
            summary_lines.append(f"🎯 TREND: {trend_analysis['direction']} ({trend_analysis['strength']})")
            summary_lines.append(f"📊 Oy: {trend_analysis['vote_count']}/3 gösterge")
            summary_lines.append(f"💰 Fiyat Değişimi: %{trend_analysis['price_change']:.1f}")
            summary_lines.append(f"📊 ADX: {trend_analysis['adx_value']:.0f}")
            summary_lines.append(f"Kullanılan Destek: {len(supports)}, Direnç: {len(resistances)}")
      
            if interval == Client.KLINE_INTERVAL_1HOUR and 'fib_38.2' in fib_levels:
                summary_lines.append(f"🔹 Fibonacci 38.2%: ${fib_levels['fib_38.2']:.2f}")
            
            # Seviye sayılarını kategorilere göre gösterme
            major_supports = len([s for s in supports if s.get('priority') == 'MAJOR' or 'Psikolojik' in s['type'] or s['test_count'] >= 4])
            major_resistances = len([r for r in resistances if r.get('priority') == 'MAJOR' or 'Psikolojik' in r['type'] or r['test_count'] >= 4])
            dynamic_supports = len([s for s in supports if 'EMA' in s['type']])
            dynamic_resistances = len([r for r in resistances if 'EMA' in r['type']])
            fib_supports = len([s for s in supports if 'Fibonacci' in s['type']])
            fib_resistances = len([r for r in resistances if 'Fibonacci' in r['type']])
            
            summary_lines.append(f"🥇 Major Seviyeler: {major_supports} destek, {major_resistances} direnç")
            summary_lines.append(f"📈 Dinamik Seviyeler: {dynamic_supports} destek, {dynamic_resistances} direnç")
            summary_lines.append(f"📐 Fibonacci Seviyeleri: {fib_supports} destek, {fib_resistances} direnç")
            summary_lines.append(f"📐 Toplam Fibonacci: 5 (referans)")
            
            if supports:
                closest_support = min(supports, key=lambda x: x['distance'])
                distance_pct = (current_price - closest_support['price']) / current_price * 100
                support_type = "🥇 MAJOR" if closest_support.get('priority') == 'MAJOR' else "📈 DYNAMIC" if 'EMA' in closest_support['type'] else "📐 FIB" if 'Fibonacci' in closest_support['type'] else "🥉 MINOR"
                summary_lines.append(f"En Yakın Destek: %{distance_pct:.1f} aşağıda ({support_type})")
            
            if resistances:
                closest_resistance = min(resistances, key=lambda x: x['distance'])
                distance_pct = (closest_resistance['price'] - current_price) / current_price * 100
                resistance_type = "🥇 MAJOR" if closest_resistance.get('priority') == 'MAJOR' else "📈 DYNAMIC" if 'EMA' in closest_resistance['type'] else "📐 FIB" if 'Fibonacci' in closest_resistance['type'] else "🥉 MINOR"
                summary_lines.append(f"En Yakın Direnç: %{distance_pct:.1f} yukarıda ({resistance_type})")
            
            # 1 Saatlik zaman diliminde, özet kutusunda tüm destek ve direnç seviyelerini işaretleriyle listeleme
            if interval == Client.KLINE_INTERVAL_1HOUR:
                summary_lines.append("\n🟢 DESTEK SEVİYELERİ:")
                for s in supports:
                    if 'Fibonacci' in s.get('type','') and '38.2' in s.get('type',''):
                        summary_lines.append(f"  🟦 Fibonacci 38.2%: ${s['price']:.2f}")
                    elif s.get('priority') == 'MAJOR' or 'Psikolojik' in s.get('type','') or s.get('test_count',0) >= 4:
                        summary_lines.append(f"  🥇 Major Destek: ${s['price']:.2f}")
                    elif 'EMA' in s.get('type',''):
                        summary_lines.append(f"  📈 Dinamik Destek: ${s['price']:.2f}")
                    else:
                        summary_lines.append(f"  🥉 Minör Destek: ${s['price']:.2f}")
                summary_lines.append("\n🔴 DİRENÇ SEVİYELERİ:")
                for r in resistances:
                    if 'Fibonacci' in r.get('type','') and '38.2' in r.get('type',''):
                        summary_lines.append(f"  🟦 Fibonacci 38.2%: ${r['price']:.2f}")
                    elif r.get('priority') == 'MAJOR' or 'Psikolojik' in r.get('type','') or r.get('test_count',0) >= 4:
                        summary_lines.append(f"  🥇 Major Direnç: ${r['price']:.2f}")
                    elif 'EMA' in r.get('type',''):
                        summary_lines.append(f"  📈 Dinamik Direnç: ${r['price']:.2f}")
                    else:
                        summary_lines.append(f"  🥉 Minör Direnç: ${r['price']:.2f}")
            elif interval == Client.KLINE_INTERVAL_4HOUR:
                summary_lines.append("\n🟧 Fibonacci 61.8% ana seviye (turuncu):")
            elif interval == Client.KLINE_INTERVAL_1DAY:
                summary_lines.append("\n🟩 Fibonacci 78.6% ana seviye (yeşil):")
            elif interval == Client.KLINE_INTERVAL_1WEEK:
                summary_lines.append("\n🟪 Fibonacci 100% Retracement (Büyük Dönüş Seviyesi):")
                if 'fib_100.0' in fib_levels:
                    summary_lines.append(f"  📊 Fibonacci 100%: ${fib_levels['fib_100.0']:.2f}")
            
            summary_text = '\n'.join(summary_lines)
            ax1.text(0.02, 0.02, summary_text, transform=ax1.transAxes, 
                    fontsize=8, verticalalignment='bottom',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            
            for txt in ax1.texts:
                txt.set_fontsize(7)
                txt.set_alpha(0.7)
            
            plt.tight_layout()
            plt.subplots_adjust(hspace=0.3)
            
            print("📈 Grafik penceresi açılıyor...")
            
            # Backend kontrolü
            if matplotlib.get_backend() == 'Agg':
                # Agg backend kullanılıyorsa dosyaya kaydet
                filename = f"{symbol}_{strategy['name'].replace(' ', '_')}_analiz.png"
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"📁 Grafik '{filename}' dosyasına kaydedildi.")
                plt.close()
            else:
                # Diğer zaman dilimleri için pencereyi göster
                plt.show(block=True)
            
            return True
            
        except Exception as e:
            print(f"❌ Grafik çizme hatası: {str(e)}")
            return False

    def print_detailed_analysis(self, supports, resistances, strategy, current_price, trend_analysis=None):
        # Seviye sözlüklerinde test_count ve strength eksikse ekleme
        for level in supports + resistances:
            if 'test_count' not in level:
                level['test_count'] = 1
            if 'strength' not in level:
                level['strength'] = 0
        print(f"\n📊 {strategy['name'].upper()} DETAYLI ANALİZ")
        print("=" * 60)
        print(f"📋 Strateji: {strategy['description']}")
        print(f"🎯 Maksimum Seviye: {strategy['max_levels']}")
        print(f"💰 Güncel Fiyat: ${current_price:.2f}")
        
        # Trend analizi
        if trend_analysis:
            print(f"\n📈 3 KRİTİK GÖSTERGE İLE TREND ANALİZİ:")
            print("-" * 50)
            print(f"🎯 FİNAL SONUÇ: {trend_analysis['direction']} ({trend_analysis['strength']})")
            print(f"📊 Oy Sayısı: {trend_analysis['vote_count']}/3 gösterge uyumlu")
            print(f"💰 Fiyat Değişimi: %{trend_analysis['price_change']:.2f}")
            print(f"\n📋 GÖSTERGE DETAYLARI:")
            print(f"  📈 MA Analizi: {trend_analysis['ma_trend']}")
            print(f"  📊 Swing Analizi: {trend_analysis['swing_trend']}")
            print(f"  📉 ADX Analizi: {trend_analysis['adx_trend']}")
            print(f"  📊 ADX Değeri: {trend_analysis['adx_value']:.1f}")
            print(f"  📈 +DI: {trend_analysis['di_plus']:.1f}")
            print(f"  📉 -DI: {trend_analysis['di_minus']:.1f}")
            
            # Trend gücü açıklaması
            print(f"\n💡 TREND GÜCÜ AÇIKLAMASI:")
            if trend_analysis['strength'] == "GÜÇLÜ":
                print("  ✅ 3 gösterge de aynı yönü gösteriyor - Güçlü trend")
            elif trend_analysis['strength'] == "ORTA":
                print("  ⚠️ 2 gösterge aynı yönü gösteriyor - Orta güçte trend")
            else:
                print("  🔄 Göstergeler karışık - Zayıf veya yatay trend")
        
        if supports:
            print(f"\n🟢 DESTEK SEVİYELERİ ({len(supports)} adet):")
            print("-" * 50)
            for i, support in enumerate(supports, 1):
                distance_pct = (current_price - support['price']) / current_price * 100
                
                # Seviye tipini belirle
                if support.get('priority') == 'MAJOR' or 'Psikolojik' in support['type'] or support['test_count'] >= 4:
                    level_icon = "🥇 MAJOR"
                elif 'EMA' in support['type']:
                    level_icon = "📈 DYNAMIC"
                elif 'Fibonacci' in support['type']:
                    level_icon = "📐 FIB"
                else:
                    level_icon = "🥉 MINOR"
                
                print(f"  {i}. {level_icon} - {support['type']}: ${support['price']:.2f}")
                print(f"     Test Sayısı: {support['test_count']}x | "
                      f"Güç: {support['strength']:.1f} | "
                      f"Mesafe: %{distance_pct:.1f} aşağıda")
        
        if resistances:
            print(f"\n🔴 DİRENÇ SEVİYELERİ ({len(resistances)} adet):")
            print("-" * 50)
            for i, resistance in enumerate(resistances, 1):
                distance_pct = (resistance['price'] - current_price) / current_price * 100
                
                # Seviye tipini belirle
                if resistance.get('priority') == 'MAJOR' or 'Psikolojik' in resistance['type'] or resistance['test_count'] >= 4:
                    level_icon = "🥇 MAJOR"
                elif 'EMA' in resistance['type']:
                    level_icon = "📈 DYNAMIC"
                elif 'Fibonacci' in resistance['type']:
                    level_icon = "📐 FIB"
                else:
                    level_icon = "🥉 MINOR"
                
                print(f"  {i}. {level_icon} - {resistance['type']}: ${resistance['price']:.2f}")
                print(f"     Test Sayısı: {resistance['test_count']}x | "
                      f"Güç: {resistance['strength']:.1f} | "
                      f"Mesafe: %{distance_pct:.1f} yukarıda")
        
        # Strateji önerileri
        print(f"\n💡 {strategy['name'].upper()} STRATEJİ ÖNERİLERİ:")
        print("-" * 50)
        
        if strategy['name'] == '1 Saatlik (Scalping)':
            print("• 🎯 5 seviye çiz, en güçlü 2 tanesini kullan (diğerleri referans)")
            print("• ⚡ RSI/Stokastik ile aşırı alım/satım onayı alın")
            print("• 📊 1H kapanışı ile kırılma onayı bekleyin")
            print("• 💪 Hızlı giriş/çıkış için EMA 50 dinamik desteği kullanın")
            print("• 📈 Fibonacci 23.6% ve 38.2% seviyelerini kısa vadeli hedefler için kullanın")
            
        elif strategy['name'] == '4 Saatlik (Swing Trading)':
            print("• 🎯 3 seviye yeterli: 1 major + 1 Fibonacci + 1 EMA")
            print("• 📊 Pin Bar/Engulfing mum kalıplarını bekleyin")
            print("• ⏰ 4H kapanışı ile trend dönüşümü sinyali arayın")
            print("• 📈 Fibonacci 50% ve 61.8% seviyelerini orta vadeli hedefler için kullanın")
            print("• 🔄 Swing high/low seviyelerini trend dönüşümü için izleyin")
            
        elif strategy['name'] == '1 Günlük (Orta-Uzun Vade)':
            print("• 🎯 Major seviyeler - psikolojik + Fibonacci odaklı")
            print("• 📊 Hacim artışı ile seviye testlerini onaylayın")
            print("• 📈 200 günlük EMA trend desteği takip edin")
            print("• 📅 Haftalık açılış/kapanış seviyelerini izleyin")
            print("• 📈 Fibonacci 78.6% seviyesini uzun vadeli hedefler için kullanın")
            
        elif strategy['name'] == '1 Haftalık (Uzun Vade)':
            print("• 🎯 2 Seviye: Tarihi Major Destek/Direnç + Fibonacci 100%")
            print("• 📊 Haftalık mum formasyonlarını (Hammer, Shooting Star) izleyin")
            print("• 📈 Fibonacci 100% Retracement büyük dönüş seviyesi olarak kullanın")
            print("• ⏳ Uzun vadeli sabırlı bekleyiş stratejisi uygulayın")
            print("• 💪 Tarihi dip/tepe seviyelerini trend dönüşümü için kullanın")
            print("• ❌ EMA'lar ve psikolojik seviyeler kullanılmaz")
        
        # Seviye öncelikleri
        print(f"\n📋 SEVİYE ÖNCELİKLERİ:")
        print("-" * 30)
        print("🥇 MAJOR: Psikolojik seviyeler + 4+ test sayısı")
        print("🥈 DYNAMIC: EMA 50/200 + Fibonacci seviyeleri")
        print("🥉 SECONDARY: Teknik seviyeler + zayıf test")
        
        # Çakışan seviyeler uyarısı
        print(f"\n⚡ ÇAKIŞAN SEVİYELER:")
        print("-" * 25)
        print("• EMA 50 + Fibonacci %50 + Psikolojik seviye aynı yerdeyse")
        print("• Bu çok güçlü bir seviyedir - mutlaka kullanın!")
        print("• %1 toleransla yakın seviyeler otomatik birleştirilir")
        
        # Trend bazlı öneriler
        if trend_analysis:
            print(f"\n🎯 3 GÖSTERGE TREND BAZLI ÖNERİLER:")
            print("-" * 40)
            if trend_analysis['direction'] == "YÜKSELİŞ":
                print("• ✅ Yükseliş trendinde destek seviyelerinden alım yapın")
                print("• 📈 Fibonacci retracement seviyelerini alım fırsatları için kullanın")
                print("• 🎯 Trend devam ederken direnç seviyelerini hedef olarak görün")
                if trend_analysis['strength'] == "GÜÇLÜ":
                    print("• 💪 Güçlü trend - pozisyonları daha uzun tutabilirsiniz")
                elif trend_analysis['strength'] == "ORTA":
                    print("• ⚠️ Orta güçte trend - dikkatli pozisyon alın")
            elif trend_analysis['direction'] == "DÜŞÜŞ":
                print("• 🔴 Düşüş trendinde direnç seviyelerinden satış yapın")
                print("• 📉 Fibonacci retracement seviyelerini satış fırsatları için kullanın")
                print("• 🎯 Trend devam ederken destek seviyelerini hedef olarak görün")
                if trend_analysis['strength'] == "GÜÇLÜ":
                    print("• 💪 Güçlü trend - kısa pozisyonları daha uzun tutabilirsiniz")
                elif trend_analysis['strength'] == "ORTA":
                    print("• ⚠️ Orta güçte trend - dikkatli pozisyon alın")
            else:  # YATAY
                print("• 🔄 Yatay trendde range trading stratejisi uygulayın")
                print("• 📊 Destek ve direnç seviyeleri arasında işlem yapın")
                print("• 🎯 Breakout/breakdown sinyallerini bekleyin")
                print("• ⚠️ Zayıf trend - küçük pozisyonlarla işlem yapın")
            
            # Gösterge uyumluluğu önerisi
            print(f"\n📊 GÖSTERGE UYUMLULUĞU:")
            print(f"  MA: {trend_analysis['ma_trend']}")
            print(f"  Swing: {trend_analysis['swing_trend']}")
            print(f"  ADX: {trend_analysis['adx_trend']}")
            if trend_analysis['vote_count'] == 3:
                print("  ✅ 3/3 gösterge uyumlu - Yüksek güvenilirlik")
            elif trend_analysis['vote_count'] == 2:
                print("  ⚠️ 2/3 gösterge uyumlu - Orta güvenilirlik")
            else:
                print("  🔄 Göstergeler karışık - Düşük güvenilirlik")

    def analyze_symbol(self, symbol, interval=Client.KLINE_INTERVAL_1DAY, limit=200):
        print(f"\n{'='*70}")
        print(f"🚀 {symbol} ANALİZİ")
        print(f"{'='*70}")
        
        self.current_interval = interval
        
        df = self.fetch_klines(symbol, interval, limit)
        if df is None:
            print("❌ Analiz yapılamadı - veri alınamadı")
            return False
        
        print("🔍 Zaman dilimine optimize edilmiş seviyeler tespit ediliyor...")
        supports, resistances = self.find_timeframe_optimized_levels(df, interval)
        
        if not supports and not resistances:
            print("❌ Anlamlı destek/direnç seviyesi bulunamadı")
            return False
        
        current_price = df['close'].iloc[-1]
        
        trend_analysis = self.analyze_trend(df, interval)
        
        self.print_detailed_analysis(supports, resistances, self.get_timeframe_strategy(interval), current_price, trend_analysis)
        
        print("\n📈 Gelişmiş grafik oluşturuluyor...")
        graph_success = self.plot_advanced_analysis(symbol, df, supports, resistances, self.get_timeframe_strategy(interval), interval)
        
        if graph_success:
            print("\n✅ Grafik başarıyla oluşturuldu!")
            print("💡 Grafiği incelemek için pencereyi kapatabilirsiniz.")
        else:
            print("\n⚠️ Grafik oluşturulamadı, ancak analiz tamamlandı.")
        
        return True

    def find_trend_lines(self, df, trend_type='up', lookback=50):
        """
        Belirtilen trend tipine göre (up/down) basit regresyon ile trend çizgisi oluşturma
        """
        try:
            recent_data = df.tail(lookback)
            x = np.arange(len(recent_data))
            if trend_type == 'up':
                y = recent_data['low'].values  # yükselen trend için dipleri bağla
            else:
                y = recent_data['high'].values  # düşen trend için tepeleri bağla
            # Doğrusal regresyon (np.polyfit ile)
            slope, intercept = np.polyfit(x, y, 1)
            trend_line = slope * x + intercept
            return x, trend_line, slope
        except Exception as e:
            print(f"❌ Trend çizgisi hatası: {str(e)}")
            return None, None, 0

    def _plot_weekly_analysis(self, symbol, df, supports, resistances, strategy):
        try:
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            current_price = df['close'].iloc[-1]
            trend_analysis = self.analyze_trend(df, Client.KLINE_INTERVAL_1WEEK)
            fib_levels, swing_high, swing_low = self.calculate_fibonacci_levels(df, 0.6)

            # Fiyat grafiği
            ax.plot(df['timestamp'], df['close'], label='Kapanış Fiyatı', linewidth=2, color='#2E86AB', alpha=0.8)

            # --- Trend Çizgileri ---
            trend_lookback = min(50, len(df))
            trend_result = trend_analysis['direction'] if trend_analysis else 'YATAY'

            if trend_result == 'YÜKSELİŞ' and trend_analysis['strength'] in ['GÜÇLÜ', 'ORTA']:
                x_up, trend_up, slope_up = self.find_trend_lines(df, trend_type='up', lookback=trend_lookback)
                if x_up is not None and trend_up is not None and len(trend_up) == trend_lookback and slope_up > 0:
                    linewidth = 2.5 if abs(slope_up) > 0.01 else 1.5
                    ax.plot(df['timestamp'].iloc[-trend_lookback:], trend_up, color='#32CD32', linestyle='-', linewidth=linewidth, label='Yükselen Trend')
                    # Ok ekle
                    ax.annotate('', xy=(df['timestamp'].iloc[-1], trend_up[-1]),
                                 xytext=(df['timestamp'].iloc[-2], trend_up[-2]),
                                 arrowprops=dict(facecolor='green', arrowstyle='->'))
            elif trend_result == 'DÜŞÜŞ' and trend_analysis['strength'] in ['GÜÇLÜ', 'ORTA']:
                x_dn, trend_dn, slope_dn = self.find_trend_lines(df, trend_type='down', lookback=trend_lookback)
                if x_dn is not None and trend_dn is not None and len(trend_dn) == trend_lookback and slope_dn < 0:
                    linewidth = 2.5 if abs(slope_dn) > 0.01 else 1.5
                    ax.plot(df['timestamp'].iloc[-trend_lookback:], trend_dn, color='#FF6347', linestyle='-', linewidth=linewidth, label='Düşen Trend')
                    # Ok ekle
                    ax.annotate('', xy=(df['timestamp'].iloc[-1], trend_dn[-1]),
                                 xytext=(df['timestamp'].iloc[-2], trend_dn[-2]),
                                 arrowprops=dict(facecolor='red', arrowstyle='->'))
            # YATAY trend için trend çizgisi çizilmez

            # Legend Kutusu
            legend_items = [
                ("#008000", "Major Destek"),
                ("#B22222", "Major Direnç"),
                ("#32CD32", "Minor Destek"),
                ("#FF6347", "Minor Direnç"),
                ("#FFD700", "Psikolojik"),
                ("#1E90FF", "EMA 50"),
                ("#808080", "EMA 200"),
                ("#FFFFFF", "Volume"),
                ("#FF1493", "Fibonacci 100%"),
            ]
            x0 = 0.75  # Sağ taraf
            y0 = 0.25  # Alt taraf
            dx = 0.08  # Geniş aralık
            box_w = 0.012  # Küçük kutular
            box_h = 0.012
            max_items_per_row = 2  # 2 sütun halinde düzenle
            for i, (color, label) in enumerate(legend_items):
                row = i // max_items_per_row
                col = i % max_items_per_row
                legend_x = x0 + col * dx
                legend_y = y0 - row * (box_h + 0.02)
                fig.patches.append(mpatches.Rectangle((legend_x, legend_y), box_w, box_h, transform=fig.transFigure, facecolor=color, edgecolor='black', linewidth=0.5, alpha=0.8, zorder=10))
                fig.text(legend_x + box_w + 0.002, legend_y + box_h/2, label, fontsize=6, va='center', ha='left', color='black', alpha=0.8)

            fig.text(
                x0,  # legend kutusuyla aynı x pozisyonu
                y0 + 0.05,  # legend kutusunun üstünde
                f"Trend: {trend_result.capitalize()}",
                fontsize=7,
                va='center',
                ha='left',
                color="green" if trend_result == "YÜKSELİŞ" else "red" if trend_result == "DÜŞÜŞ" else "gray",
                alpha=0.9,
                fontweight='bold',
                transform=fig.transFigure
            )

            # Fibonacci 100% çizgisi
            if 'fib_100.0' in fib_levels:
                fib_100_price = fib_levels['fib_100.0']
                ax.axhline(y=fib_100_price, color='#FF1493', linestyle=(0, (4, 4)), linewidth=2, alpha=0.8, label='Fibonacci 100%')
                ax.text(df['timestamp'].iloc[-5], fib_100_price, 'Fibonacci 100%', va='center', ha='right', fontsize=10, bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))

            # Destek ve direnç çizgileri
            for support in supports:
                ax.axhline(y=support['price'], color='green', linestyle='-', linewidth=3, alpha=0.8)
                ax.text(df['timestamp'].iloc[-3], support['price'], f"Destek: ${support['price']:.2f}", va='top', ha='right', fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.3))
            for resistance in resistances:
                ax.axhline(y=resistance['price'], color='red', linestyle='-', linewidth=3, alpha=0.8)
                ax.text(df['timestamp'].iloc[-3], resistance['price'], f"Direnç: ${resistance['price']:.2f}", va='bottom', ha='right', fontsize=9, bbox=dict(boxstyle="round,pad=0.2", facecolor='red', alpha=0.3))

            # Güncel fiyat çizgisi
            ax.axhline(y=current_price, color='blue', linestyle=':', linewidth=2, alpha=0.8, label=f'Güncel: ${current_price:.2f}')

            # Başlık ve düzen
            ax.set_title(f'{symbol} - {strategy["name"]} Analizi\nGüncel Fiyat: ${current_price:.2f} | Trend: {trend_analysis["direction"]}', fontsize=12, fontweight='bold', pad=20)
            ax.set_ylabel('Fiyat (USDT)', fontsize=10)
            ax.set_xlabel('Tarih', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            plt.setp(ax.get_xticklabels(), rotation=45, fontsize=8)
            
            # --- Analiz Özet Kutusu ---
            summary_lines = []
            summary_lines.append(f"📊 {strategy['name'].upper()} ANALİZ ÖZETİ")
            summary_lines.append("-" * 35)
            summary_lines.append(f"🎯 TREND: {trend_analysis['direction']} ({trend_analysis['strength']})")
            summary_lines.append(f"📊 Oy: {trend_analysis['vote_count']}/3 gösterge")
            summary_lines.append(f"💰 Fiyat Değişimi: %{trend_analysis['price_change']:.1f}")
            summary_lines.append(f"📊 ADX: {trend_analysis['adx_value']:.0f}")
            summary_lines.append(f"Kullanılan Destek: {len(supports)}, Direnç: {len(resistances)}")
            
            if 'fib_100.0' in fib_levels:
                summary_lines.append(f"🔹 Fibonacci 100%: ${fib_levels['fib_100.0']:.2f}")
            
            # Seviye sayılarını kategorilere göre gösterme
            major_supports = len([s for s in supports if s.get('priority') == 'MAJOR' or 'Tarihi' in s['type']])
            major_resistances = len([r for r in resistances if r.get('priority') == 'MAJOR' or 'Tarihi' in r['type']])
            fib_supports = len([s for s in supports if 'Fibonacci' in s['type']])
            fib_resistances = len([r for r in resistances if 'Fibonacci' in r['type']])
            
            summary_lines.append(f"🥇 Tarihi Seviyeler: {major_supports} destek, {major_resistances} direnç")
            summary_lines.append(f"📐 Fibonacci Seviyeleri: {fib_supports} destek, {fib_resistances} direnç")
            summary_lines.append(f"📐 Toplam Fibonacci: 1 (100% Retracement)")
            
            if supports:
                closest_support = min(supports, key=lambda x: x['distance'])
                distance_pct = (current_price - closest_support['price']) / current_price * 100
                support_type = "🥇 TARİHİ" if 'Tarihi' in closest_support['type'] else "📐 FIB" if 'Fibonacci' in closest_support['type'] else "🥉 BASİT"
                summary_lines.append(f"En Yakın Destek: %{distance_pct:.1f} aşağıda ({support_type})")
            
            if resistances:
                closest_resistance = min(resistances, key=lambda x: x['distance'])
                distance_pct = (closest_resistance['price'] - current_price) / current_price * 100
                resistance_type = "🥇 TARİHİ" if 'Tarihi' in closest_resistance['type'] else "📐 FIB" if 'Fibonacci' in closest_resistance['type'] else "🥉 BASİT"
                summary_lines.append(f"En Yakın Direnç: %{distance_pct:.1f} yukarıda ({resistance_type})")
            
            # 1 Haftalık özel bilgiler
            summary_lines.append("\n🟪 Fibonacci 100% Retracement (Büyük Dönüş Seviyesi):")
            if 'fib_100.0' in fib_levels:
                summary_lines.append(f"  📊 Fibonacci 100%: ${fib_levels['fib_100.0']:.2f}")
            
            summary_text = '\n'.join(summary_lines)
            ax.text(0.02, 0.05, summary_text, transform=ax.transAxes, 
                   fontsize=8, verticalalignment='bottom',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
            
            plt.tight_layout()

            # Grafiği pencerede göster
            print("📈 1 Haftalık grafik penceresi açılıyor...")
            plt.show(block=True)
            print("✅ 1 Haftalık grafik penceresi kapatıldı.")
            return True
        except Exception as e:
            print(f"❌ 1 Haftalık grafik hatası: {str(e)}")
            plt.close('all')
            return False

def validate_symbol(symbol):
    """Kripto sembol formatını doğrulama"""
    symbol = symbol.strip().upper()
    if not symbol or not symbol.isalnum() or len(symbol) < 5:
        print("❌ Geçersiz sembol formatı")
        return None
    return symbol

def get_timeframe_choice():
    """Kullanıcıdan zaman dilimi seçimi alma"""
    print("\n⏰ Zaman Dilimi Seçiniz:")
    print("[1] 1 Saat - Scalping")
    print("[2] 4 Saat - Swing Trading") 
    print("[3] 1 Gün - Orta-Uzun Vade")
    print("[4] 1 Hafta - Uzun Vade")
    
    while True:
        try:
            choice = int(input("Seçiminizi yapınız (1-4): "))
            if choice in range(1, 5):
                break
            else:
                print("❌ Lütfen 1-4 arası bir sayı giriniz.")
        except ValueError:
            print("❌ Lütfen geçerli bir sayı giriniz.")
    
    timeframe_map = {
        1: (Client.KLINE_INTERVAL_1HOUR, 168),
        2: (Client.KLINE_INTERVAL_4HOUR, 168),
        3: (Client.KLINE_INTERVAL_1DAY, 200),
        4: (Client.KLINE_INTERVAL_1WEEK, 100)
    }
    
    return timeframe_map[choice]

def count_valid_tests(price_data, level, tolerance=0.005):
    lows = price_data['low'].values
    highs = price_data['high'].values
    test_lows = np.sum(np.abs(lows - level)/level < tolerance)
    test_highs = np.sum(np.abs(highs - level)/level < tolerance)
    return (test_lows + test_highs) >= 2  # Min 2 test (1h/4h için)

def find_key_levels(price_data):
    levels = []
    for idx in argrelextrema(price_data['low'].values, np.less, order=3)[0]:
        price = price_data.iloc[idx]['low']
        if count_valid_tests(price_data, price):
            levels.append({'price': price, 'type': 'support', 'last_tested': price_data.index[idx]})
    return levels

def filter_levels(levels, current_price, threshold_pct=3):
    valid_range = (current_price * (1 - threshold_pct/100), 
                  current_price * (1 + threshold_pct/100))
    return [lvl for lvl in levels if valid_range[0] <= lvl['price'] <= valid_range[1]]

def filter_by_time(levels, price_data, max_hours=12):
    current_time = price_data.index[-1]
    return [lvl for lvl in levels if 
            (current_time - lvl['last_tested']).total_seconds()/3600 <= max_hours]

def get_precision_levels(symbol, timeframe, price_data):
    # Seviyeleri bul
    supports = find_key_levels(price_data)
    resistances = find_key_levels(price_data.rename(columns={'high':'low'}))
    
    # Filtreleme
    current_price = price_data['close'].iloc[-1]
    valid_levels = {
        'supports': filter_levels(supports, current_price),
        'resistances': filter_levels(resistances, current_price)
    }
    
    # Zaman filtreleme (sadece 1h/4h için)
    if timeframe in ['1h', '4h']:
        valid_levels['supports'] = filter_by_time(valid_levels['supports'], price_data)
        valid_levels['resistances'] = filter_by_time(valid_levels['resistances'], price_data)
    
    return valid_levels

def main():
    """Ana fonksiyon"""
    print("=" * 80)
    print("🪙 GELİŞMİŞ KRİPTO PARA DESTEK & DİRENÇ ANALİZ ARACI v5.0")
    print("=" * 80)
    print("📋 Zaman dilimine özel optimize edilmiş profesyonel analiz")
    print("🎯 Teknik + Psikolojik + Fibonacci seviye tespiti")
    print("⚡ Hareketli ortalama ve hacim onayları")
    

    
    try:
        analyzer = CryptoAnalyzer()
        
        # Sembol al
        while True:
            print("\n" + "="*50)
            symbol_input = input("🪙 Kripto sembolünü giriniz (örn: BTCUSDT, ETHUSDT): ").strip()
            
            if not symbol_input:
                print("❌ Boş sembol giremezsiniz!")
                continue
                
            symbol = validate_symbol(symbol_input)
            if symbol:
                break
        
        # Zaman dilimi seçme
        interval, limit = get_timeframe_choice()
        
        # Analizi başlatma
        print(f"\n🚀 {symbol} analizi başlatılıyor...")
        success = analyzer.analyze_symbol(symbol, interval, limit)
        
        if success:
            print(f"\n✅ {symbol} analizi başarıyla tamamlandı!")
            
            while True:
                choice = input("\n❓ Başka bir sembol analiz etmek ister misiniz? (e/h): ").lower()
                if choice in ['e', 'evet', 'y', 'yes']:
                    main()  # Recursive call
                    break
                elif choice in ['h', 'hayır', 'n', 'no']:
                    print("\n👋 Analiz tamamlandı. İyi günler!")
                    break
                else:
                    print("❌ Lütfen 'e' (evet) veya 'h' (hayır) yazınız.")
        else:
            print(f"\n❌ {symbol} analizi tamamlanamadı!")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n❌ Beklenmeyen hata: {str(e)}")
        print("🔧 Lütfen internet bağlantınızı kontrol edin ve tekrar deneyiniz.")

if __name__ == "__main__":
    main()
