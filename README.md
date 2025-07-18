# 📊 CryptoDataAnalyzer

**CryptoDataAnalyzer** is an advanced crypto analytics tool that uses multi-timeframe price action, EMA, ADX, swing high/low patterns, and psychological/Fibonacci levels to detect optimized support/resistance zones and overall market trend direction. Designed for data-driven traders and analysts.

> ✅ Language: Turkish Interface & Commentary  
> ✅ Ideal for scalping, swing trading, and long-term position analysis

---

## 🧠 Overview

The tool utilizes a three-tiered trend detection system (EMA, ADX, Swing Analysis) and dynamically identifies support/resistance levels using:
- Historical major/minor swing points
- Fibonacci retracements
- EMA levels (50/200)
- Psychological price levels

**Trend Detection System:**
- **📈 Yükseliş (Uptrend)**: All indicators suggest bullish movement
- **🔻 Düşüş (Downtrend)**: Strong bearish consensus across indicators
- **🔄 Yatay (Sideways)**: Inconclusive trend; price consolidation likely
- **❓ Belirsiz (Uncertain)**: Weak or noisy signal, further confirmation needed

---

## ⚙️ Features

- ✅ **Multi-Timeframe Support/Resistance Detection** (1H, 4H, 1D, 1W)
- ✅ **EMA (50 / 200)** Dynamic Levels per timeframe
- ✅ **ADX + Swing High/Low + EMA Combo Trend Detection**
- ✅ **Fibonacci retracement overlays**
- ✅ **Psychological level mapping**
- ✅ **Auto-prioritization of support/resistance levels**
- ✅ **Custom visual chart rendering (matplotlib)**
- ✅ **Adaptive strategy selection based on timeframe**
- ✅ **Turkish analysis output with color-coded signal guidance**

---
## 🔍 Analysis Methodology

### 📊 Multi-Factor Convergence System
This tool employs a weighted scoring algorithm that synthesizes data from:

#### 1. Trend Detection Layer
```mermaid
graph TD
    A[EMA Cross] --> B[50EMA > 200EMA = Bullish]
    A --> C[50EMA < 200EMA = Bearish]
    D[ADX] --> E[>25 = Strong Trend]
    D --> F[<20 = Weak Trend]
    G[Swing Fractals] --> H[Higher Highs/Lows = Uptrend]
    G --> I[Lower Highs/Lows = Downtrend]
```
---

## 🖼️ Visual Output

- 📌 Support & resistance lines (color-coded by priority)
- 📊 EMA levels drawn with explanation
- 🧠 Trend summary box with indicator details
- 📉 Volume bars with dynamic price scaling
- 🔺 Fibonacci levels drawn as dashed or solid lines

---

## 🧰 Technologies Used

| Tech              | Purpose                        |
|------------------|--------------------------------|
| Python           | Core programming language      |
| Pandas / NumPy   | Data wrangling & computation   |
| Matplotlib       | Financial chart visualization  |
| Binance API      | Real-time and historical data  |
| Scipy / Stats    | Swing point & statistical calc |
| TkAgg / Agg      | Flexible matplotlib backend    |

---

## 🚀 Usage

```bash
# 1. Clone the repo
git clone https://github.com/your-username/CryptoDataAnalyzer.git

# 2. Install requirements
pip install -r requirements.txt

# 3. Create a .env file and add your Binance credentials
API_KEY=your_api_key_here
API_SECRET=your_secret_key_here

# 4. Run the tool
python Kripto5.1.py
```

## 🖥️ Preview
### 🔍Crypto and Timeframe Selection Panel
![Ekran görüntüsübtc](https://github.com/user-attachments/assets/fb819abd-d69a-4db5-94ea-1094b0966a79)

### 🤖 Strategy Analysis Results
### ⏳ Hour (Scalping Framework)
![Figure_btc](https://github.com/user-attachments/assets/ddc48c80-5214-4d7d-a99a-d4f1d8b51605)
* Micro-structure visualization
* High-precision level marking

### 🕓 4 Hour (Swing Dashboard)
![Figure_btcusdt4hours](https://github.com/user-attachments/assets/26f74d21-b6c6-4a5e-bc07-653043fc02f4)
* Intermediate-term trend channels
* Volume-profile integrated zones

### 🌞 Daily (Position Trader View)
![Figure-btcusdtdays](https://github.com/user-attachments/assets/db6b867d-759b-46e3-8707-68e934e917ab)
* Macro structure mapping
* Institutional-level pivots

### 📅 Weekly (Strategic Vision)
![Figure_btcusdtweekly](https://github.com/user-attachments/assets/19ef6425-2de8-4626-a882-60aff4a1f5a4)
* Multi-year reference levels
* Cyclical pattern analysis

## ⚠️ Legal Disclaimer

**Educational Use Only**  
- Technical analysis only, **not investment advice**  
- For research/educational purposes  
- Cryptocurrency trading involves high risk  
- No accuracy guarantees  
- Not affiliated with any financial authority  

> *"Always conduct your own research (DYOR) before trading."*

## 👨‍💻 Developer

**Ömer Faruk Karaköy**    
🌐 GitHub: [github.com/OmerFarukKarakoy](https://github.com/OmerFarukKarakoy)  
📧 Mail: omerfarukkarakoy@gmail.com

---
## License

This project is licensed under the MIT License.  
© 2025 Ömer Faruk Karakoy — You are free to use, modify, and distribute this software.  
Provided "as is", without warranty of any kind.
