import os, sys, argparse, time, datetime
import requests, math
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------
def fetch_sp500_tickers():
    print("Fetching S&P 500 tickers...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }

    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        tables = pd.read_html(StringIO(response.text))
        if len(tables) == 0:
            raise ValueError("No tables found on Wikipedia page")

        df = tables[0]

        if "Symbol" not in df.columns:
            raise ValueError("Wikipedia structure changed — 'Symbol' column missing")

        tickers = df["Symbol"].tolist()
        print(f"✅ Wikipedia fetch successful — {len(tickers)} tickers found.")
        print("Sample:", tickers[:10])
        return tickers

    except Exception as e:
        print(f"⚠️ Wikipedia fetch failed: {e}")
        print("Trying fallback source (DataHub)...")
        try:
            fallback_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            df = pd.read_csv(fallback_url)
            col = "Symbol" if "Symbol" in df.columns else "symbol"
            tickers = df[col].tolist()
            print(f"✅ Fallback successful — {len(tickers)} tickers found.")
            print("Sample:", tickers[:10])
            return tickers
        except Exception as e2:
            print(f"❌ Fallback failed too: {e2}")
            raise RuntimeError("Failed to fetch S&P 500 tickers from all sources.")


def fetch_tsx_tickers_online():
    """Fetch TSX tickers from online source instead of local CSV"""
    print("Fetching TSX tickers from online source...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/115.0.0.0 Safari/537.36"
    }
    
    try:
        # Try Wikipedia TSX 60 list
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_60"
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        tables = pd.read_html(StringIO(response.text))
        if len(tables) > 0:
            df = tables[0]
            # Look for ticker column
            ticker_col = None
            for col in df.columns:
                if 'symbol' in str(col).lower() or 'ticker' in str(col).lower():
                    ticker_col = col
                    break
            
            if ticker_col:
                tickers = df[ticker_col].tolist()
                # Add .TO suffix if not present
                tickers = [t if '.TO' in str(t) else f"{t}.TO" for t in tickers]
                print(f"✅ TSX fetch successful — {len(tickers)} tickers found.")
                return tickers
        
        raise ValueError("Could not parse TSX data")
        
    except Exception as e:
        print(f"⚠️ TSX online fetch failed: {e}")
        print("Using default TSX tickers...")
        # Fallback to major TSX stocks
        return ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "TRP.TO", 
                "SHOP.TO", "BCE.TO", "BMO.TO", "CM.TO", "SU.TO", "CNR.TO"]


def fetch_top_crypto_tickers(top_n=10):
    """Fetch top N cryptocurrencies by market cap using free APIs"""
    print(f"Fetching top {top_n} cryptocurrencies by market cap...")
    
    # Try CoinGecko first (no API key needed)
    try:
        url = f"https://api.coingecko.com/api/v3/coins/markets"
        params = {
            "vs_currency": "usd",
            "order": "market_cap_desc",
            "per_page": top_n,
            "page": 1,
            "sparkline": False
        }
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Map CoinGecko symbols to yfinance tickers
        crypto_map = {
            "bitcoin": "BTC-USD",
            "ethereum": "ETH-USD",
            "tether": "USDT-USD",
            "binancecoin": "BNB-USD",
            "solana": "SOL-USD",
            "usd-coin": "USDC-USD",
            "ripple": "XRP-USD",
            "cardano": "ADA-USD",
            "dogecoin": "DOGE-USD",
            "tron": "TRX-USD",
            "avalanche-2": "AVAX-USD",
            "shiba-inu": "SHIB-USD",
            "polkadot": "DOT-USD",
            "chainlink": "LINK-USD",
            "bitcoin-cash": "BCH-USD",
            "litecoin": "LTC-USD",
            "polygon": "MATIC-USD",
            "stellar": "XLM-USD",
            "ethereum-classic": "ETC-USD",
            "monero": "XMR-USD"
        }
        
        tickers = []
        for coin in data[:top_n]:
            coin_id = coin.get("id")
            if coin_id in crypto_map:
                tickers.append(crypto_map[coin_id])
        
        print(f"✅ CoinGecko fetch successful — {len(tickers)} crypto tickers found.")
        print("Cryptos:", tickers)
        return tickers
        
    except Exception as e:
        print(f"⚠️ CoinGecko failed: {e}")
        print("Trying CoinCap fallback...")
        
        try:
            # Try CoinCap as fallback
            url = "https://api.coincap.io/v2/assets"
            params = {"limit": top_n}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            crypto_map = {
                "bitcoin": "BTC-USD",
                "ethereum": "ETH-USD",
                "tether": "USDT-USD",
                "binance-coin": "BNB-USD",
                "solana": "SOL-USD",
                "usd-coin": "USDC-USD",
                "xrp": "XRP-USD",
                "cardano": "ADA-USD",
                "dogecoin": "DOGE-USD",
                "tron": "TRX-USD"
            }
            
            tickers = []
            for coin in data.get("data", [])[:top_n]:
                coin_id = coin.get("id")
                if coin_id in crypto_map:
                    tickers.append(crypto_map[coin_id])
            
            print(f"✅ CoinCap fetch successful — {len(tickers)} crypto tickers found.")
            return tickers
            
        except Exception as e2:
            print(f"⚠️ CoinCap failed: {e2}")
            # Final fallback to static top 10
            print("Using static top 10 crypto list...")
            return ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", 
                    "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"]


def fetch_gdelt_events(query, max_records=20):
    """Fetch events from GDELT Project (free, no API key needed)"""
    try:
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": query,
            "mode": "artlist",
            "maxrecords": max_records,
            "format": "json",
            "timespan": "7d"  # Last 7 days
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()
        return data.get("articles", [])
    except Exception as e:
        print(f"⚠️ GDELT fetch failed for '{query}': {e}")
        return []


def analyze_geopolitical_risk():
    """Analyze geopolitical risks using GDELT and NewsAPI"""
    print("Analyzing geopolitical risks...")
    
    risk_score = 0
    risk_keywords = ["war", "conflict", "military", "attack", "strike", "invasion", "bombing", "missile"]
    
    # Fetch from GDELT
    gdelt_articles = fetch_gdelt_events("war OR conflict OR military OR attack", max_records=30)
    
    # Count mentions
    war_mentions = 0
    for article in gdelt_articles:
        title = article.get("title", "").lower()
        for keyword in risk_keywords:
            if keyword in title:
                war_mentions += 1
                break
    
    # Score: 0-100 (0=peaceful, 100=high risk)
    # More than 20 war-related articles in 7 days = high risk
    risk_score = min((war_mentions / 20.0) * 100, 100)
    
    print(f"  Geopolitical risk score: {risk_score:.1f}/100 ({war_mentions} conflict-related articles)")
    return risk_score


def analyze_trade_conditions():
    """Analyze global trade conditions"""
    print("Analyzing trade conditions...")
    
    trade_risk = 0
    trade_keywords = ["trade war", "tariff", "sanctions", "embargo", "trade dispute", "protectionism"]
    
    # Fetch from GDELT
    gdelt_articles = fetch_gdelt_events("trade war OR tariff OR sanctions", max_records=30)
    
    # Fetch from NewsAPI if available
    if NEWSAPI_KEY:
        news_articles = news_headlines("trade war OR tariffs OR sanctions", max_results=10)
        gdelt_articles.extend(news_articles)
    
    # Count mentions
    trade_mentions = 0
    for article in gdelt_articles:
        title = article.get("title", "").lower()
        for keyword in trade_keywords:
            if keyword in title:
                trade_mentions += 1
                break
    
    # Score: 0-100 (0=free trade, 100=trade war)
    trade_risk = min((trade_mentions / 15.0) * 100, 100)
    
    print(f"  Trade risk score: {trade_risk:.1f}/100 ({trade_mentions} trade-related articles)")
    return trade_risk


def analyze_economic_sentiment():
    """Analyze economic sentiment (inflation, recession, Fed policy)"""
    print("Analyzing economic sentiment...")
    
    if not NEWSAPI_KEY:
        print("  ⚠️ NewsAPI key not found, skipping economic sentiment")
        return 0
    
    # Fetch economic news
    economic_queries = [
        "Federal Reserve OR Fed policy OR interest rates",
        "inflation OR consumer prices",
        "recession OR economic downturn",
        "GDP growth OR economic growth"
    ]
    
    all_articles = []
    for query in economic_queries:
        articles = news_headlines(query, max_results=5)
        all_articles.extend(articles)
        time.sleep(0.5)
    
    # Sentiment analysis
    total_sentiment = 0
    count = 0
    
    for article in all_articles:
        text = article.get("title", "") + " " + (article.get("description") or "")
        sentiment = analyzer.polarity_scores(text)
        total_sentiment += sentiment.get("compound", 0)
        count += 1
    
    avg_sentiment = total_sentiment / count if count > 0 else 0
    
    print(f"  Economic sentiment: {avg_sentiment:.3f} (-1=bearish, +1=bullish)")
    return avg_sentiment


def fetch_macro_sentiment():
    """Fetch and compile global macro factors"""
    print("\n🌍 Fetching Global Macro Sentiment...")
    
    geopolitical_risk = analyze_geopolitical_risk()
    trade_risk = analyze_trade_conditions()
    economic_sentiment = analyze_economic_sentiment()
    
    # Calculate overall macro score (-30 to +30)
    # Negative factors
    risk_penalty = -(geopolitical_risk / 100 * 15)  # -15 max
    trade_penalty = -(trade_risk / 100 * 10)  # -10 max
    
    # Positive factors
    economic_boost = economic_sentiment * 15  # -15 to +15
    
    overall_macro_score = risk_penalty + trade_penalty + economic_boost
    overall_macro_score = max(min(overall_macro_score, 30), -30)
    
    macro_data = {
        "geopolitical_risk": geopolitical_risk,
        "trade_risk": trade_risk,
        "economic_sentiment": economic_sentiment,
        "overall_macro_score": overall_macro_score
    }
    
    print(f"\n📊 Macro Summary:")
    print(f"  Overall Macro Score: {overall_macro_score:.2f}/30")
    print(f"  - Geopolitical Risk: {geopolitical_risk:.1f}/100")
    print(f"  - Trade Risk: {trade_risk:.1f}/100")
    print(f"  - Economic Sentiment: {economic_sentiment:.3f}")
    print()
    
    return macro_data


def apply_macro_adjustment(base_score, asset_type, macro_data):
    """Apply macro factors differently based on asset type"""
    
    geo_risk = macro_data["geopolitical_risk"]
    trade_risk = macro_data["trade_risk"]
    econ_sent = macro_data["economic_sentiment"]
    
    adjustment = 0
    
    if asset_type == "commodity":  # Gold/Silver - safe havens
        # HIGH geopolitical risk → BOOST commodities
        adjustment += (geo_risk / 100) * 15  # 0 to +15
        
        # HIGH trade risk → BOOST commodities (uncertainty)
        adjustment += (trade_risk / 100) * 8  # 0 to +8
        
        # NEGATIVE economic sentiment → BOOST commodities (recession hedge)
        if econ_sent < 0:
            adjustment += abs(econ_sent) * 7  # 0 to +7
        else:
            adjustment -= econ_sent * 5  # Strong economy hurts gold
    
    elif asset_type == "crypto":
        # HIGH geopolitical risk → HURT crypto (risk-off)
        adjustment -= (geo_risk / 100) * 20  # 0 to -20
        
        # HIGH trade risk → HURT crypto (risk-off)
        adjustment -= (trade_risk / 100) * 10  # 0 to -10
        
        # POSITIVE economic sentiment → BOOST crypto (risk-on)
        adjustment += econ_sent * 15  # -15 to +15
        
        # Exception: Bitcoin as "digital gold" - slight safe haven
        # This would need ticker-specific logic
    
    elif asset_type == "stock":
        # HIGH geopolitical risk → HURT stocks (moderate)
        adjustment -= (geo_risk / 100) * 12  # 0 to -12
        
        # HIGH trade risk → HURT stocks (especially exporters)
        adjustment -= (trade_risk / 100) * 10  # 0 to -10
        
        # POSITIVE economic sentiment → BOOST stocks
        adjustment += econ_sent * 18  # -18 to +18
    
    return adjustment


def compute_technical_indicators(series):
    """Compute RSI, MACD, and EMAs using ta library"""
    series = series.dropna()
    if len(series) < 50:
        return None
    df = pd.DataFrame({"close": series})

    # RSI
    rsi = RSIIndicator(df["close"], window=14).rsi()
    df["rsi_14"] = rsi

    # MACD
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()

    # EMAs
    df["ema20"] = EMAIndicator(df["close"], window=20).ema_indicator()
    df["ema50"] = EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema200"] = EMAIndicator(df["close"], window=200).ema_indicator()

    # Simple swing detection
    df["swing_high"] = df["close"][(df["close"].shift(1) < df["close"]) & (df["close"].shift(-1) < df["close"])]
    df["swing_low"] = df["close"][(df["close"].shift(1) > df["close"]) & (df["close"].shift(-1) > df["close"])]

    out = {
        "rsi": float(df["rsi_14"].iloc[-1]) if not pd.isna(df["rsi_14"].iloc[-1]) else 50,
        "macd": float(df["macd"].iloc[-1]) if not pd.isna(df["macd"].iloc[-1]) else 0,
        "macd_signal": float(df["macd_signal"].iloc[-1]) if not pd.isna(df["macd_signal"].iloc[-1]) else 0,
        "ema20_vs_close": float(df["ema20"].iloc[-1] - df["close"].iloc[-1]) if not pd.isna(df["ema20"].iloc[-1]) else 0,
        "ema50_vs_close": float(df["ema50"].iloc[-1] - df["close"].iloc[-1]) if not pd.isna(df["ema50"].iloc[-1]) else 0,
        "ema200_vs_close": float(df["ema200"].iloc[-1] - df["close"].iloc[-1]) if not pd.isna(df["ema200"].iloc[-1]) else 0,
    }
    return out


def compute_momentum_volatility(series):
    """Compute momentum and volatility metrics for crypto/commodities"""
    series = series.dropna()
    if len(series) < 30:
        return None
    
    # Price changes over different periods
    pct_change_7d = ((series.iloc[-1] / series.iloc[-7]) - 1) * 100 if len(series) >= 7 else 0
    pct_change_30d = ((series.iloc[-1] / series.iloc[-30]) - 1) * 100 if len(series) >= 30 else 0
    pct_change_90d = ((series.iloc[-1] / series.iloc[-90]) - 1) * 100 if len(series) >= 90 else 0
    
    # Volatility (standard deviation of returns)
    returns = series.pct_change().dropna()
    volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
    
    return {
        "pct_change_7d": pct_change_7d,
        "pct_change_30d": pct_change_30d,
        "pct_change_90d": pct_change_90d,
        "volatility": volatility
    }


def news_headlines(query, max_results=10):
    if not NEWSAPI_KEY:
        return []
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&pageSize={max_results}&apiKey={NEWSAPI_KEY}"
    try:
        r = requests.get(url, timeout=10).json()
        return r.get("articles", [])
    except Exception:
        return []


def headline_sentiment(headline):
    return analyzer.polarity_scores(headline)


# ---------- scoring ----------
def score_stock(fund, tech, sentiment_score, macro_data):
    score = 0.0
    # fundamentals
    if fund.get("pe") and 0 < fund["pe"] < 40:
        score += 20
    if fund.get("de_ratio") and fund["de_ratio"] < 1.5:
        score += 10
    if fund.get("revenue_3y_growth") and fund["revenue_3y_growth"] > 0.1:
        score += 10
    # technicals
    if tech:
        if tech.get("rsi") and 40 < tech["rsi"] < 70:
            score += 10
        if tech.get("macd") and tech["macd"] > tech.get("macd_signal", 0):
            score += 10
    # growth
    if fund.get("eps_growth_3y", 0) > 0.1:
        score += 15
    # sentiment
    score += max(min((sentiment_score * 10), 10), -10)
    
    # Apply macro adjustment
    macro_adjustment = apply_macro_adjustment(score, "stock", macro_data)
    score += macro_adjustment
    
    return score


def score_crypto_commodity(tech, momentum, sentiment_score, macro_data, asset_type="crypto"):
    """Balanced scoring for crypto/commodities based on technicals, momentum, volatility, and sentiment"""
    score = 0.0
    
    # Technical indicators (40 points possible)
    if tech:
        # RSI - prefer middle range (balanced, not overbought/oversold)
        rsi = tech.get("rsi", 50)
        if 45 < rsi < 55:
            score += 15  # Neutral zone
        elif 40 < rsi < 60:
            score += 10  # Slightly off neutral
        elif 30 < rsi < 70:
            score += 5   # More extreme but not terrible
        
        # MACD - bullish crossover
        if tech.get("macd", 0) > tech.get("macd_signal", 0):
            score += 10
        
        # EMA trends - price above moving averages
        if tech.get("ema20_vs_close", 0) < 0:  # Price above EMA20
            score += 5
        if tech.get("ema50_vs_close", 0) < 0:  # Price above EMA50
            score += 5
        if tech.get("ema200_vs_close", 0) < 0:  # Price above EMA200
            score += 5
    
    # Momentum trends (25 points possible) - balanced approach
    if momentum:
        # 7-day momentum
        pct_7d = momentum.get("pct_change_7d", 0)
        if -2 < pct_7d < 5:  # Slight uptrend, not too volatile
            score += 8
        elif -5 < pct_7d < 10:
            score += 5
        
        # 30-day momentum
        pct_30d = momentum.get("pct_change_30d", 0)
        if 0 < pct_30d < 15:  # Positive but not overheated
            score += 10
        elif -5 < pct_30d < 25:
            score += 5
        
        # 90-day momentum
        pct_90d = momentum.get("pct_change_90d", 0)
        if 0 < pct_90d < 30:  # Steady growth
            score += 7
        elif -10 < pct_90d < 50:
            score += 3
    
    # Volatility (10 points possible) - prefer moderate volatility
    if momentum and momentum.get("volatility"):
        vol = momentum.get("volatility", 100)
        if 20 < vol < 50:  # Moderate volatility
            score += 10
        elif 15 < vol < 70:
            score += 6
        elif 10 < vol < 100:
            score += 3
    
    # Sentiment (20 points possible)
    score += max(min((sentiment_score * 20), 20), -20)
    
    # Base score for being in top assets
    score += 5
    
    # Apply macro adjustment
    macro_adjustment = apply_macro_adjustment(score, asset_type, macro_data)
    score += macro_adjustment
    
    return score


# ---------- main ----------
def main(output="email"):
    # Fetch macro sentiment FIRST (applies to all assets)
    macro_data = fetch_macro_sentiment()
    
    sp500 = fetch_sp500_tickers()
    tsx = fetch_tsx_tickers_online()
    universe = sp500[:200] + tsx[:100]
    print(f"Running for {len(universe)} stock tickers...")

    # Fetch crypto and commodities
    crypto_tickers = fetch_top_crypto_tickers(top_n=10)
    commodity_tickers = ["GC=F", "SI=F"]  # Gold and Silver futures
    
    print(f"Analyzing {len(crypto_tickers)} cryptos and {len(commodity_tickers)} commodities...")

    # Analyze stocks
    stock_results = []
    for ticker in universe:
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="1y", interval="1d")
            if hist.empty:
                continue
            close = hist["Close"]
            tech = compute_technical_indicators(close)
            info = yf_ticker.info
            fund = {
                "pe": info.get("trailingPE"),
                "de_ratio": info.get("debtToEquity"),
                "marketCap": info.get("marketCap"),
                "eps": info.get("trailingEps"),
                "eps_growth_3y": info.get("earningsQuarterlyGrowth") or 0,
                "revenue_3y_growth": None,
            }

            articles = news_headlines(ticker, max_results=5)
            comp = sum(
                analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")).get("compound", 0)
                for a in articles
            )
            avg_sent = comp / len(articles) if articles else 0
            score = score_stock(fund, tech, avg_sent, macro_data)
            stock_results.append({"ticker": ticker, "score": score, "fund": fund, "tech": tech, "sentiment": avg_sent})
            time.sleep(0.25)
        except Exception as e:
            print("err", ticker, e)

    # Analyze crypto/commodities
    crypto_commodity_results = []
    for ticker in crypto_tickers + commodity_tickers:
        try:
            # Determine asset type
            asset_type = "commodity" if "=F" in ticker else "crypto"
            
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="1y", interval="1d")
            if hist.empty:
                continue
            close = hist["Close"]
            tech = compute_technical_indicators(close)
            momentum = compute_momentum_volatility(close)
            
            # Get sentiment
            search_term = ticker.replace("-USD", "").replace("=F", "")
            if ticker == "GC=F":
                search_term = "gold price"
            elif ticker == "SI=F":
                search_term = "silver price"
            
            articles = news_headlines(search_term, max_results=5)
            comp = sum(
                analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")).get("compound", 0)
                for a in articles
            )
            avg_sent = comp / len(articles) if articles else 0
            
            score = score_crypto_commodity(tech, momentum, avg_sent, macro_data, asset_type)
            crypto_commodity_results.append({
                "ticker": ticker,
                "score": score,
                "tech": tech,
                "momentum": momentum,
                "sentiment": avg_sent,
                "type": asset_type
            })
            time.sleep(0.25)
        except Exception as e:
            print("err", ticker, e)

    # Create DataFrames and sort
    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False)
    df_crypto = pd.DataFrame(crypto_commodity_results).sort_values("score", ascending=False)
    
    top30_stocks = df_stocks.head(30)
    bottom30_stocks = df_stocks.tail(30)
    top_crypto = df_crypto.head(len(crypto_commodity_results))  # Show all

    market_news = []
    mnews = news_headlines("stock market OR equities OR S&P 500 OR TSX", max_results=6)
    for a in mnews:
        market_news.append(
            {"title": a.get("title"), "source": a.get("source", {}).get("name"), "url": a.get("url")}
        )

    portfolio = {
        "equities": 0.6,
        "bonds_or_cash": 0.2,
        "gold_silver_crypto_mini": 0.1,
        "high_quality_cyclicals": 0.1,
    }

    # Generate report
    md = []
    md.append(f"# Daily Market Report — {datetime.datetime.utcnow().date()}\n")
    
    md.append("## 🌍 Global Macro Environment\n")
    md.append(f"**Overall Macro Score:** {macro_data['overall_macro_score']:.2f}/30\n")
    md.append(f"- Geopolitical Risk: {macro_data['geopolitical_risk']:.1f}/100\n")
    md.append(f"- Trade Risk: {macro_data['trade_risk']:.1f}/100\n")
    md.append(f"- Economic Sentiment: {macro_data['economic_sentiment']:.3f} (-1 to +1)\n")
    
    md.append("\n## 📈 Top 30 Stocks (by composite score)\n")
    md.append(top30_stocks[["ticker", "score"]].to_markdown(index=False))
    
    md.append("\n## 📉 Bottom 30 Stocks (by composite score)\n")
    md.append(bottom30_stocks[["ticker", "score"]].to_markdown(index=False))
    
    md.append("\n## 🪙 Crypto/Commodities Rankings\n")
    md.append(top_crypto[["ticker", "score", "type"]].to_markdown(index=False))
    
    md.append("\n## 📰 Market headlines\n")
    for n in market_news:
        md.append(f"- **{n['title']}** — {n['source']}")
    
    md.append("\n## 💼 Portfolio (moderate risk) suggestion\n")
    md.append(str(portfolio))
    
    md.append("\n---\n")
    md.append(f"*Report generated at {datetime.datetime.utcnow()} UTC*")
    
    report_md = "\n\n".join(md)

    if output == "email":
        send_email(report_md)
    elif output == "slack":
        send_slack(report_md)
    else:
        print(report_md)

    with open("daily_report.md", "w") as f:
        f.write(report_md)
    print("✅ Done.")


def generate_html_email(top30_stocks, bottom30_stocks, top_crypto, macro_data, market_news):
    """Generate beautiful HTML email template"""
    
    # Determine macro status
    macro_score = macro_data['overall_macro_score']
    if macro_score > 10:
        macro_status = "🟢 Positive"
        macro_color = "#10b981"
    elif macro_score < -10:
        macro_status = "🔴 Negative"
        macro_color = "#ef4444"
    else:
        macro_status = "🟡 Neutral"
        macro_color = "#f59e0b"
    
    # Convert DataFrames to HTML tables
    def df_to_html_table(df, columns, highlight_color="#f3f4f6"):
        rows = ""
        for idx, row in df[columns].iterrows():
            score_color = "#10b981" if row['score'] > 50 else "#ef4444" if row['score'] < 30 else "#6b7280"
            rows += f"""
            <tr style="border-bottom: 1px solid #e5e7eb;">
                <td style="padding: 12px; font-weight: 600; color: #1f2937;">{row['ticker']}</td>
                <td style="padding: 12px; color: {score_color}; font-weight: 700;">{row['score']:.1f}</td>
            </tr>
            """
        return rows
    
    top_stocks_html = df_to_html_table(top30_stocks.head(30), ['ticker', 'score'])
    bottom_stocks_html = df_to_html_table(bottom30_stocks.tail(30), ['ticker', 'score'])
    crypto_html = df_to_html_table(top_crypto, ['ticker', 'score'])
    
    # Market news HTML
    news_html = ""
    for n in market_news[:6]:
        news_html += f"""
        <div style="margin-bottom: 12px; padding: 12px; background: #f9fafb; border-left: 3px solid #3b82f6; border-radius: 4px;">
            <strong style="color: #1f2937;">{n['title']}</strong>
            <div style="color: #6b7280; font-size: 14px; margin-top: 4px;">Source: {n['source']}</div>
        </div>
        """
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Daily Market Report</title>
    </head>
    <body style="margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; background-color: #f3f4f6;">
        <div style="max-width: 800px; margin: 0 auto; background-color: #ffffff;">
            
            <!-- Header -->
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 40px 20px; text-align: center;">
                <h1 style="margin: 0; color: #ffffff; font-size: 32px; font-weight: 800;">📊 Daily Market Report</h1>
                <p style="margin: 10px 0 0 0; color: #e0e7ff; font-size: 16px;">{datetime.datetime.utcnow().strftime('%B %d, %Y')}</p>
            </div>
            
            <!-- Macro Environment -->
            <div style="padding: 30px 20px; background-color: #fafafa; border-bottom: 2px solid #e5e7eb;">
                <h2 style="margin: 0 0 20px 0; color: #1f2937; font-size: 24px; font-weight: 700;">🌍 Global Macro Environment</h2>
                
                <div style="display: flex; flex-wrap: wrap; gap: 15px; margin-bottom: 20px;">
                    <div style="flex: 1; min-width: 200px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #6b7280; font-size: 14px; margin-bottom: 5px;">Overall Score</div>
                        <div style="color: {macro_color}; font-size: 28px; font-weight: 800;">{macro_score:.1f}/30</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 5px;">{macro_status}</div>
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #6b7280; font-size: 14px; margin-bottom: 5px;">Geopolitical Risk</div>
                        <div style="color: #ef4444; font-size: 28px; font-weight: 800;">{macro_data['geopolitical_risk']:.0f}/100</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 5px;">{"High" if macro_data['geopolitical_risk'] > 50 else "Moderate" if macro_data['geopolitical_risk'] > 25 else "Low"}</div>
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #6b7280; font-size: 14px; margin-bottom: 5px;">Trade Risk</div>
                        <div style="color: #f59e0b; font-size: 28px; font-weight: 800;">{macro_data['trade_risk']:.0f}/100</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 5px;">{"High" if macro_data['trade_risk'] > 50 else "Moderate" if macro_data['trade_risk'] > 25 else "Low"}</div>
                    </div>
                    
                    <div style="flex: 1; min-width: 200px; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <div style="color: #6b7280; font-size: 14px; margin-bottom: 5px;">Economic Sentiment</div>
                        <div style="color: {"#10b981" if macro_data['economic_sentiment'] > 0 else "#ef4444"}; font-size: 28px; font-weight: 800;">{macro_data['economic_sentiment']:.2f}</div>
                        <div style="color: #6b7280; font-size: 12px; margin-top: 5px;">{"Bullish" if macro_data['economic_sentiment'] > 0.1 else "Bearish" if macro_data['economic_sentiment'] < -0.1 else "Neutral"}</div>
                    </div>
                </div>
            </div>
            
            <!-- Top 30 Stocks -->
            <div style="padding: 30px 20px;">
                <h2 style="margin: 0 0 20px 0; color: #1f2937; font-size: 24px; font-weight: 700;">📈 Top 30 Stocks</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <thead>
                            <tr style="background: #10b981; color: white;">
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Ticker</th>
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {top_stocks_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Bottom 30 Stocks -->
            <div style="padding: 30px 20px; background-color: #fafafa;">
                <h2 style="margin: 0 0 20px 0; color: #1f2937; font-size: 24px; font-weight: 700;">📉 Bottom 30 Stocks</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <thead>
                            <tr style="background: #ef4444; color: white;">
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Ticker</th>
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {bottom_stocks_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Crypto/Commodities -->
            <div style="padding: 30px 20px;">
                <h2 style="margin: 0 0 20px 0; color: #1f2937; font-size: 24px; font-weight: 700;">🪙 Crypto/Commodities Rankings</h2>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1);">
                        <thead>
                            <tr style="background: #f59e0b; color: white;">
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Asset</th>
                                <th style="padding: 12px; text-align: left; font-weight: 600;">Score</th>
                            </tr>
                        </thead>
                        <tbody>
                            {crypto_html}
                        </tbody>
                    </table>
                </div>
            </div>
            
            <!-- Market News -->
            <div style="padding: 30px 20px; background-color: #fafafa;">
                <h2 style="margin: 0 0 20px 0; color: #1f2937; font-size: 24px; font-weight: 700;">📰 Market Headlines</h2>
                {news_html}
            </div>
            
            <!-- Footer -->
            <div style="padding: 30px 20px; background: #1f2937; color: #9ca3af; text-align: center; font-size: 14px;">
                <p style="margin: 0;">Generated at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p>
                <p style="margin: 10px 0 0 0; font-size: 12px;">This is an automated report. Data sources: Yahoo Finance, NewsAPI, GDELT, CoinGecko</p>
            </div>
            
        </div>
    </body>
    </html>
    """
    
    return html


def send_email(body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    SMTP_USER = os.getenv("SMTP_USER")
    SMTP_PASS = os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        print("SMTP creds missing; printing instead.\n")
        print(body)
        return

    msg = MIMEMultipart('alternative')
    msg["Subject"] = f"📊 Daily Market Report - {datetime.date.today()}"
    msg["From"] = SMTP_USER
    msg["To"] = SMTP_USER
    
    # Attach plain text version (fallback)
    text_part = MIMEText(body, 'plain')
    msg.attach(text_part)
    
    # Attach HTML version (primary)
    html_part = MIMEText(body, 'html')
    msg.attach(html_part)

    s = smtplib.SMTP("smtp.gmail.com", 587)
    s.starttls()
    s.login(SMTP_USER, SMTP_PASS)
    s.send_message(msg)
    s.quit()
    print("✅ Email sent.")


def send_slack(body):
    url = os.getenv("SLACK_WEBHOOK")
    if not url:
        print(body)
        return
    requests.post(url, json={"text": body[:3000]})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="print", choices=["print", "email", "slack"])
    args = parser.parse_args()
    main(output=args.output)
