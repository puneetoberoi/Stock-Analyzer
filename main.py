
import os, sys, argparse, time, datetime, logging
import requests, math
import pandas as pd
import numpy as np
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Centralized headers for all requests to look like a real browser
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36"
}

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY")
ALPHAVANTAGE_KEY = os.getenv("ALPHAVANTAGE_KEY")

analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------

def make_robust_request(url, params=None, headers=REQUEST_HEADERS, timeout=15, retries=3, delay=5):
    """
    A robust requests function with retries and detailed error logging.
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            
            # Specifically check for empty content before trying to decode JSON
            if response.content:
                return response
            else:
                logging.warning(f"Request to {url} was successful but returned empty content.")
                return None

        except requests.exceptions.HTTPError as http_err:
            logging.warning(f"HTTP error on attempt {attempt + 1}/{retries} for {url}: {http_err}")
        except requests.exceptions.ConnectionError as conn_err:
            logging.warning(f"Connection error on attempt {attempt + 1}/{retries} for {url}: {conn_err}")
        except requests.exceptions.Timeout as timeout_err:
            logging.warning(f"Timeout on attempt {attempt + 1}/{retries} for {url}: {timeout_err}")
        except requests.exceptions.RequestException as req_err:
            logging.warning(f"General request error on attempt {attempt + 1}/{retries} for {url}: {req_err}")
        
        if attempt < retries - 1:
            logging.info(f"Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            logging.error(f"All {retries} attempts failed for URL: {url}")
    
    return None

def fetch_sp500_tickers():
    logging.info("Fetching S&P 500 tickers...")
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = make_robust_request(url)
        if not response:
            raise ValueError("Failed to fetch Wikipedia page for S&P 500.")

        tables = pd.read_html(StringIO(response.text))
        if not tables:
            raise ValueError("No tables found on Wikipedia page")

        df = tables[0]
        if "Symbol" not in df.columns:
            raise ValueError("Wikipedia structure changed — 'Symbol' column missing")

        tickers = df["Symbol"].tolist()
        logging.info(f"✅ Wikipedia fetch successful — {len(tickers)} tickers found.")
        logging.info(f"Sample: {tickers[:10]}")
        return tickers

    except Exception as e:
        logging.warning(f"⚠️ Wikipedia fetch failed: {e}")
        logging.info("Trying fallback source (DataHub)...")
        try:
            fallback_url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
            df = pd.read_csv(fallback_url)
            col = "Symbol" if "Symbol" in df.columns else "symbol"
            tickers = df[col].tolist()
            logging.info(f"✅ Fallback successful — {len(tickers)} tickers found.")
            logging.info(f"Sample: {tickers[:10]}")
            return tickers
        except Exception as e2:
            logging.error(f"❌ Fallback failed too: {e2}")
            raise RuntimeError("Failed to fetch S&P 500 tickers from all sources.")


def fetch_tsx_tickers_online():
    """Fetch TSX tickers with smarter parsing and better fallbacks."""
    logging.info("Fetching TSX tickers from online source...")
    try:
        # Try Wikipedia S&P/TSX Composite Index (more comprehensive)
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        response = make_robust_request(url)
        if not response:
            raise ValueError("Failed to fetch Wikipedia page for TSX.")

        tables = pd.read_html(StringIO(response.text))
        
        # Heuristic search for the constituents table
        constituents_table = None
        for table in tables:
            # A good heuristic: the table with 'Symbol' and 'Company' columns is likely the one.
            if 'Symbol' in table.columns and 'Company' in table.columns:
                constituents_table = table
                break
        
        if constituents_table is not None:
            tickers = constituents_table["Symbol"].tolist()
            # Clean up tickers: remove potential junk and ensure .TO suffix
            tickers = [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in tickers]
            logging.info(f"✅ TSX fetch successful — {len(tickers)} tickers found from Composite Index.")
            return tickers
        
        raise ValueError("Could not find the constituents table on the Wikipedia page.")

    except Exception as e:
        logging.warning(f"⚠️ TSX Wikipedia fetch failed: {e}")
        logging.info("Trying fallback source (major TSX stocks)...")
        # A slightly expanded and more reliable static list as a fallback
        return [
            "RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "TRP.TO", 
            "SHOP.TO", "BCE.TO", "BMO.TO", "CM.TO", "SU.TO", "CNR.TO",
            "CP.TO", "ATD.TO", "GIB-A.TO", "L.TO", "TRI.TO", "WCN.TO",
            "NTR.TO", "MFC.TO"
        ]

def fetch_top_crypto_tickers(top_n=10):
    """Fetch top N cryptocurrencies by market cap using free APIs"""
    logging.info(f"Fetching top {top_n} cryptocurrencies by market cap...")
    
    # Try CoinGecko first (no API key needed)
    try:
        url = "https://api.coingecko.com/api/v3/coins/markets"
        params = {"vs_currency": "usd", "order": "market_cap_desc", "per_page": top_n, "page": 1, "sparkline": False}
        response = make_robust_request(url, params=params)
        if not response:
            raise ConnectionError("Failed to connect to CoinGecko API.")
        data = response.json()
        
        crypto_map = {
            "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "tether": "USDT-USD", "binancecoin": "BNB-USD",
            "solana": "SOL-USD", "usd-coin": "USDC-USD", "ripple": "XRP-USD", "cardano": "ADA-USD",
            "dogecoin": "DOGE-USD", "tron": "TRX-USD", "avalanche-2": "AVAX-USD", "shiba-inu": "SHIB-USD",
            "polkadot": "DOT-USD", "chainlink": "LINK-USD", "bitcoin-cash": "BCH-USD", "litecoin": "LTC-USD",
            "polygon": "MATIC-USD", "stellar": "XLM-USD", "ethereum-classic": "ETC-USD", "monero": "XMR-USD"
        }
        
        tickers = [crypto_map[coin.get("id")] for coin in data[:top_n] if coin.get("id") in crypto_map]
        
        logging.info(f"✅ CoinGecko fetch successful — {len(tickers)} crypto tickers found.")
        logging.info(f"Cryptos: {tickers}")
        return tickers
        
    except Exception as e:
        logging.warning(f"⚠️ CoinGecko failed: {e}")
        logging.info("Trying CoinCap fallback...")
        try:
            url = "https://api.coincap.io/v2/assets"
            params = {"limit": top_n}
            response = make_robust_request(url, params=params)
            if not response:
                raise ConnectionError("Failed to connect to CoinCap API.")
            data = response.json()
            
            crypto_map = {
                "bitcoin": "BTC-USD", "ethereum": "ETH-USD", "tether": "USDT-USD", "binance-coin": "BNB-USD",
                "solana": "SOL-USD", "usd-coin": "USDC-USD", "xrp": "XRP-USD", "cardano": "ADA-USD",
                "dogecoin": "DOGE-USD", "tron": "TRX-USD"
            }
            
            tickers = [crypto_map[coin.get("id")] for coin in data.get("data", [])[:top_n] if coin.get("id") in crypto_map]
            logging.info(f"✅ CoinCap fetch successful — {len(tickers)} crypto tickers found.")
            return tickers
            
        except Exception as e2:
            logging.warning(f"⚠️ CoinCap failed: {e2}")
            logging.info("Using static top 10 crypto list as final fallback...")
            return ["BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "SOL-USD", "XRP-USD", "USDC-USD", "ADA-USD", "DOGE-USD", "TRX-USD"]


def fetch_gdelt_events(query, max_records=20):
    """Fetch events from GDELT Project using the robust request handler."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query, "mode": "artlist", "maxrecords": max_records,
        "format": "json", "timespan": "7d"
    }
    response = make_robust_request(url, params=params)
    if response:
        try:
            data = response.json()
            return data.get("articles", [])
        except ValueError: # Catches JSONDecodeError
            logging.error(f"GDELT returned non-JSON data for query '{query}'. Response text: {response.text[:200]}")
            return []
    logging.warning(f"GDELT fetch failed for '{query}' after multiple retries.")
    return []


def analyze_geopolitical_risk():
    """Analyze geopolitical risks using GDELT"""
    logging.info("Analyzing geopolitical risks...")
    risk_keywords = ["war", "conflict", "military", "attack", "strike", "invasion", "bombing", "missile"]
    gdelt_articles = fetch_gdelt_events("war OR conflict OR military OR attack", max_records=30)
    
    war_mentions = 0
    for article in gdelt_articles:
        title = article.get("title", "").lower()
        if any(keyword in title for keyword in risk_keywords):
            war_mentions += 1
    
    risk_score = min((war_mentions / 20.0) * 100, 100)
    logging.info(f"  Geopolitical risk score: {risk_score:.1f}/100 ({war_mentions} conflict-related articles)")
    return risk_score


def analyze_trade_conditions():
    """Analyze global trade conditions"""
    logging.info("Analyzing trade conditions...")
    trade_keywords = ["trade war", "tariff", "sanctions", "embargo", "trade dispute", "protectionism"]
    
    all_articles = fetch_gdelt_events("trade war OR tariff OR sanctions", max_records=30)
    if NEWSAPI_KEY:
        all_articles.extend(news_headlines("trade war OR tariffs OR sanctions", max_results=10))
    
    trade_mentions = 0
    for article in all_articles:
        title = article.get("title", "").lower()
        if any(keyword in title for keyword in trade_keywords):
            trade_mentions += 1
            
    trade_risk = min((trade_mentions / 15.0) * 100, 100)
    logging.info(f"  Trade risk score: {trade_risk:.1f}/100 ({trade_mentions} trade-related articles)")
    return trade_risk


def analyze_economic_sentiment():
    """Analyze economic sentiment (inflation, recession, Fed policy)"""
    logging.info("Analyzing economic sentiment...")
    if not NEWSAPI_KEY:
        logging.warning("  NewsAPI key not found, skipping economic sentiment.")
        return 0
    
    economic_queries = ["Federal Reserve OR Fed policy OR interest rates", "inflation OR consumer prices", "recession OR economic downturn", "GDP growth OR economic growth"]
    all_articles = []
    for query in economic_queries:
        all_articles.extend(news_headlines(query, max_results=5))
        time.sleep(0.5)
    
    if not all_articles:
        logging.warning("  No economic articles found.")
        return 0
        
    sentiments = [analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")).get("compound", 0) for a in all_articles]
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    logging.info(f"  Economic sentiment: {avg_sentiment:.3f} (-1=bearish, +1=bullish)")
    return avg_sentiment


def fetch_macro_sentiment():
    """Fetch and compile global macro factors"""
    logging.info("\n🌍 Fetching Global Macro Sentiment...")
    
    geopolitical_risk = analyze_geopolitical_risk()
    trade_risk = analyze_trade_conditions()
    economic_sentiment = analyze_economic_sentiment()
    
    risk_penalty = -(geopolitical_risk / 100 * 15)
    trade_penalty = -(trade_risk / 100 * 10)
    economic_boost = economic_sentiment * 15
    overall_macro_score = max(min(risk_penalty + trade_penalty + economic_boost, 30), -30)
    
    macro_data = {
        "geopolitical_risk": geopolitical_risk,
        "trade_risk": trade_risk,
        "economic_sentiment": economic_sentiment,
        "overall_macro_score": overall_macro_score
    }
    
    logging.info(f"\n📊 Macro Summary:")
    logging.info(f"  Overall Macro Score: {overall_macro_score:.2f}/30")
    logging.info(f"  - Geopolitical Risk: {geopolitical_risk:.1f}/100")
    logging.info(f"  - Trade Risk: {trade_risk:.1f}/100")
    logging.info(f"  - Economic Sentiment: {economic_sentiment:.3f}\n")
    return macro_data


def apply_macro_adjustment(base_score, asset_type, macro_data):
    """Apply macro factors differently based on asset type"""
    geo_risk, trade_risk, econ_sent = macro_data["geopolitical_risk"], macro_data["trade_risk"], macro_data["economic_sentiment"]
    adjustment = 0
    if asset_type == "commodity":
        adjustment += (geo_risk / 100) * 15
        adjustment += (trade_risk / 100) * 8
        adjustment += abs(econ_sent) * 7 if econ_sent < 0 else -econ_sent * 5
    elif asset_type == "crypto":
        adjustment -= (geo_risk / 100) * 20
        adjustment -= (trade_risk / 100) * 10
        adjustment += econ_sent * 15
    elif asset_type == "stock":
        adjustment -= (geo_risk / 100) * 12
        adjustment -= (trade_risk / 100) * 10
        adjustment += econ_sent * 18
    return adjustment


def compute_technical_indicators(series):
    """Compute RSI, MACD, and EMAs using ta library"""
    series = series.dropna()
    if len(series) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    df["ema20"], df["ema50"], df["ema200"] = EMAIndicator(df["close"], 20).ema_indicator(), EMAIndicator(df["close"], 50).ema_indicator(), EMAIndicator(df["close"], 200).ema_indicator()
    
    latest = df.iloc[-1]
    return {
        "rsi": float(latest.get("rsi_14", 50)),
        "macd": float(latest.get("macd", 0)),
        "macd_signal": float(latest.get("macd_signal", 0)),
        "ema20_vs_close": float(latest.get("ema20", latest["close"]) - latest["close"]),
        "ema50_vs_close": float(latest.get("ema50", latest["close"]) - latest["close"]),
        "ema200_vs_close": float(latest.get("ema200", latest["close"]) - latest["close"]),
    }


def compute_momentum_volatility(series):
    """Compute momentum and volatility metrics for crypto/commodities"""
    series = series.dropna()
    if len(series) < 90: return None
    return {
        "pct_change_7d": ((series.iloc[-1] / series.iloc[-7]) - 1) * 100,
        "pct_change_30d": ((series.iloc[-1] / series.iloc[-30]) - 1) * 100,
        "pct_change_90d": ((series.iloc[-1] / series.iloc[-90]) - 1) * 100,
        "volatility": series.pct_change().std() * np.sqrt(252) * 100
    }


def news_headlines(query, max_results=10):
    if not NEWSAPI_KEY: return []
    url = f"https://newsapi.org/v2/everything?q={requests.utils.quote(query)}&pageSize={max_results}&apiKey={NEWSAPI_KEY}"
    response = make_robust_request(url)
    if response:
        try:
            return response.json().get("articles", [])
        except ValueError:
            logging.error(f"NewsAPI returned non-JSON data for query '{query}'.")
    return []


# ---------- scoring ----------
def score_stock(fund, tech, sentiment_score, macro_data):
    score = 0.0
    if fund.get("pe") and 0 < fund["pe"] < 40: score += 20
    if fund.get("de_ratio") and fund["de_ratio"] < 1.5: score += 10
    if fund.get("revenue_3y_growth") and fund["revenue_3y_growth"] > 0.1: score += 10
    if tech:
        if 40 < tech.get("rsi", 50) < 70: score += 10
        if tech.get("macd", 0) > tech.get("macd_signal", 0): score += 10
    if fund.get("eps_growth_3y", 0) > 0.1: score += 15
    score += max(min((sentiment_score * 10), 10), -10)
    score += apply_macro_adjustment(score, "stock", macro_data)
    return score


def score_crypto_commodity(tech, momentum, sentiment_score, macro_data, asset_type="crypto"):
    score = 0.0
    if tech:
        rsi = tech.get("rsi", 50)
        if 45 < rsi < 55: score += 15
        elif 40 < rsi < 60: score += 10
        elif 30 < rsi < 70: score += 5
        if tech.get("macd", 0) > tech.get("macd_signal", 0): score += 10
        if tech.get("ema20_vs_close", 0) < 0: score += 5
        if tech.get("ema50_vs_close", 0) < 0: score += 5
        if tech.get("ema200_vs_close", 0) < 0: score += 5
    if momentum:
        pct_7d, pct_30d, pct_90d, vol = momentum.get("pct_change_7d", 0), momentum.get("pct_change_30d", 0), momentum.get("pct_change_90d", 0), momentum.get("volatility", 100)
        if -2 < pct_7d < 5: score += 8
        elif -5 < pct_7d < 10: score += 5
        if 0 < pct_30d < 15: score += 10
        elif -5 < pct_30d < 25: score += 5
        if 0 < pct_90d < 30: score += 7
        elif -10 < pct_90d < 50: score += 3
        if 20 < vol < 50: score += 10
        elif 15 < vol < 70: score += 6
    score += max(min((sentiment_score * 20), 20), -20) + 5
    score += apply_macro_adjustment(score, asset_type, macro_data)
    return score


# ---------- main ----------
def main(output="print"):
    macro_data = fetch_macro_sentiment()
    sp500 = fetch_sp500_tickers()
    tsx = fetch_tsx_tickers_online()
    universe = sp500[:200] + tsx[:100]
    logging.info(f"Running for {len(universe)} stock tickers...")

    crypto_tickers = fetch_top_crypto_tickers(top_n=10)
    commodity_tickers = ["GC=F", "SI=F"]
    logging.info(f"Analyzing {len(crypto_tickers)} cryptos and {len(commodity_tickers)} commodities...")

    stock_results, crypto_commodity_results = [], []

    for ticker in universe:
        try:
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="1y", interval="1d")
            if hist.empty: continue
            tech = compute_technical_indicators(hist["Close"])
            info = yf_ticker.info
            fund = {"pe": info.get("trailingPE"), "de_ratio": info.get("debtToEquity"), "eps_growth_3y": info.get("earningsQuarterlyGrowth"), "revenue_3y_growth": None}
            articles = news_headlines(info.get("longName", ticker), max_results=5)
            sentiments = [analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")).get("compound", 0) for a in articles]
            avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
            score = score_stock(fund, tech, avg_sent, macro_data)
            stock_results.append({"ticker": ticker, "score": score, "sentiment": avg_sent})
            time.sleep(0.25)
        except Exception as e:
            logging.error(f"Error processing stock {ticker}: {e}")

    for ticker in crypto_tickers + commodity_tickers:
        try:
            asset_type = "commodity" if "=F" in ticker else "crypto"
            yf_ticker = yf.Ticker(ticker)
            hist = yf_ticker.history(period="1y", interval="1d")
            if hist.empty: continue
            close = hist["Close"]
            tech, momentum = compute_technical_indicators(close), compute_momentum_volatility(close)
            search_term = {"GC=F": "gold price", "SI=F": "silver price"}.get(ticker, ticker.replace("-USD", ""))
            articles = news_headlines(search_term, max_results=5)
            sentiments = [analyzer.polarity_scores(a.get("title", "") + " " + (a.get("description") or "")).get("compound", 0) for a in articles]
            avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
            score = score_crypto_commodity(tech, momentum, avg_sent, macro_data, asset_type)
            crypto_commodity_results.append({"ticker": ticker, "score": score, "type": asset_type, "sentiment": avg_sent})
            time.sleep(0.25)
        except Exception as e:
            logging.error(f"Error processing asset {ticker}: {e}")

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False)
    df_crypto = pd.DataFrame(crypto_commodity_results).sort_values("score", ascending=False)
    
    market_news = []
    mnews = news_headlines("stock market OR equities OR S&P 500 OR TSX", max_results=6)
    for a in mnews:
        market_news.append({"title": a.get("title"), "source": a.get("source", {}).get("name"), "url": a.get("url")})
    
    report_md = generate_markdown_report(df_stocks, df_crypto, macro_data, market_news)

    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, macro_data, market_news)
        send_email(html_email) # Send HTML email
    elif output == "slack":
        send_slack(report_md)
    else:
        print(report_md)

    with open("daily_report.md", "w", encoding='utf-8') as f:
        f.write(report_md)
    logging.info("✅ Done.")

def generate_markdown_report(df_stocks, df_crypto, macro_data, market_news):
    md = [
        f"# Daily Market Report — {datetime.datetime.utcnow().date()}\n",
        "## 🌍 Global Macro Environment",
        f"**Overall Macro Score:** {macro_data['overall_macro_score']:.2f}/30",
        f"- Geopolitical Risk: {macro_data['geopolitical_risk']:.1f}/100",
        f"- Trade Risk: {macro_data['trade_risk']:.1f}/100",
        f"- Economic Sentiment: {macro_data['economic_sentiment']:.3f} (-1 to +1)\n",
        "## 📈 Top 30 Stocks (by composite score)",
        df_stocks.head(30)[["ticker", "score"]].to_markdown(index=False),
        "\n## 📉 Bottom 30 Stocks (by composite score)",
        df_stocks.tail(30)[["ticker", "score"]].to_markdown(index=False),
        "\n## 🪙 Crypto/Commodities Rankings",
        df_crypto[["ticker", "score", "type"]].to_markdown(index=False),
        "\n## 📰 Market headlines"
    ]
    for n in market_news:
        md.append(f"- **{n['title']}** — {n['source']}")
    
    md.extend([
        "\n## 💼 Portfolio (moderate risk) suggestion",
        str({"equities": 0.6, "bonds_or_cash": 0.2, "gold_silver_crypto_mini": 0.1, "high_quality_cyclicals": 0.1}),
        "\n---",
        f"*Report generated at {datetime.datetime.utcnow()} UTC*"
    ])
    return "\n\n".join(md)

def generate_html_email(df_stocks, df_crypto, macro_data, market_news):
    """Generate beautiful HTML email template"""
    macro_score = macro_data['overall_macro_score']
    if macro_score > 10: macro_status, macro_color = "🟢 Positive", "#10b981"
    elif macro_score < -10: macro_status, macro_color = "🔴 Negative", "#ef4444"
    else: macro_status, macro_color = "🟡 Neutral", "#f59e0b"
    
    def df_to_html_table(df, columns):
        rows = ""
        for _, row in df[columns].iterrows():
            score_color = "#10b981" if row['score'] > 50 else "#ef4444" if row['score'] < 30 else "#6b7280"
            rows += f'<tr style="border-bottom: 1px solid #e5e7eb;"><td style="padding: 12px; font-weight: 600;">{row["ticker"]}</td><td style="padding: 12px; color: {score_color}; font-weight: 700;">{row["score"]:.1f}</td></tr>'
        return rows
    
    top_stocks_html = df_to_html_table(df_stocks.head(30), ['ticker', 'score'])
    bottom_stocks_html = df_to_html_table(df_stocks.tail(30).iloc[::-1], ['ticker', 'score'])
    crypto_html = df_to_html_table(df_crypto, ['ticker', 'score'])
    
    news_html = ""
    for n in market_news[:6]:
        news_html += f'<div style="margin-bottom:12px;padding:12px;background:#f9fafb;border-left:3px solid #3b82f6;border-radius:4px;"><strong style="color:#1f2937;">{n["title"]}</strong><div style="color:#6b7280;font-size:14px;margin-top:4px;">Source: {n["source"]}</div></div>'

    # The HTML email content (condensed for brevity)
    return f"""
    <!DOCTYPE html><html><head><title>Daily Market Report</title></head><body style="margin:0;padding:0;font-family:-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;background-color:#f3f4f6;"><div style="max-width:800px;margin:0 auto;background-color:#fff;">
    <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%);padding:40px 20px;text-align:center;"><h1 style="margin:0;color:#fff;font-size:32px;font-weight:800;">📊 Daily Market Report</h1><p style="margin:10px 0 0;color:#e0e7ff;font-size:16px;">{datetime.datetime.utcnow().strftime('%B %d, %Y')}</p></div>
    <div style="padding:30px 20px;background-color:#fafafa;border-bottom:2px solid #e5e7eb;"><h2 style="margin:0 0 20px;color:#1f2937;font-size:24px;">🌍 Global Macro Environment</h2>
    <div style="display:flex;flex-wrap:wrap;gap:15px;margin-bottom:20px;"><div style="flex:1;min-width:180px;background:white;padding:20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div style="color:#6b7280;font-size:14px;margin-bottom:5px;">Overall Score</div><div style="color:{macro_color};font-size:28px;font-weight:800;">{macro_score:.1f}/30</div><div style="color:#6b7280;font-size:12px;margin-top:5px;">{macro_status}</div></div>
    <div style="flex:1;min-width:180px;background:white;padding:20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div style="color:#6b7280;font-size:14px;margin-bottom:5px;">Geopolitical Risk</div><div style="color:#ef4444;font-size:28px;font-weight:800;">{macro_data['geopolitical_risk']:.0f}/100</div></div>
    <div style="flex:1;min-width:180px;background:white;padding:20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div style="color:#6b7280;font-size:14px;margin-bottom:5px;">Trade Risk</div><div style="color:#f59e0b;font-size:28px;font-weight:800;">{macro_data['trade_risk']:.0f}/100</div></div>
    <div style="flex:1;min-width:180px;background:white;padding:20px;border-radius:8px;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><div style="color:#6b7280;font-size:14px;margin-bottom:5px;">Economic Sentiment</div><div style="color:{"#10b981" if macro_data['economic_sentiment']>0 else "#ef4444"};font-size:28px;font-weight:800;">{macro_data['economic_sentiment']:.2f}</div></div></div></div>
    <div style="padding:30px 20px;"><h2 style="margin:0 0 20px;color:#1f2937;font-size:24px;">📈 Top 30 Stocks</h2><div><table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><thead><tr style="background:#10b981;color:white;"><th style="padding:12px;text-align:left;">Ticker</th><th style="padding:12px;text-align:left;">Score</th></tr></thead><tbody>{top_stocks_html}</tbody></table></div></div>
    <div style="padding:30px 20px;background-color:#fafafa;"><h2 style="margin:0 0 20px;color:#1f2937;font-size:24px;">📉 Bottom 30 Stocks</h2><div><table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><thead><tr style="background:#ef4444;color:white;"><th style="padding:12px;text-align:left;">Ticker</th><th style="padding:12px;text-align:left;">Score</th></tr></thead><tbody>{bottom_stocks_html}</tbody></table></div></div>
    <div style="padding:30px 20px;"><h2 style="margin:0 0 20px;color:#1f2937;font-size:24px;">🪙 Crypto/Commodities</h2><div><table style="width:100%;border-collapse:collapse;background:white;border-radius:8px;overflow:hidden;box-shadow:0 1px 3px rgba(0,0,0,0.1);"><thead><tr style="background:#f59e0b;color:white;"><th style="padding:12px;text-align:left;">Asset</th><th style="padding:12px;text-align:left;">Score</th></tr></thead><tbody>{crypto_html}</tbody></table></div></div>
    <div style="padding:30px 20px;background-color:#fafafa;"><h2 style="margin:0 0 20px;color:#1f2937;font-size:24px;">📰 Market Headlines</h2>{news_html}</div>
    <div style="padding:30px 20px;background:#1f2937;color:#9ca3af;text-align:center;font-size:14px;"><p style="margin:0;">Generated at {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC</p><p style="margin:10px 0 0;font-size:12px;">This is an automated report. Not financial advice.</p></div>
    </div></body></html>
    """

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS:
        logging.warning("SMTP creds missing; cannot send email.")
        return

    msg = MIMEMultipart('alternative')
    msg["Subject"] = f"📊 Daily Market Report - {datetime.date.today()}"
    msg["From"], msg["To"] = SMTP_USER, SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")


def send_slack(body):
    url = os.getenv("SLACK_WEBHOOK")
    if not url:
        logging.warning("SLACK_WEBHOOK missing; cannot send to Slack.")
        return
    try:
        response = requests.post(url, json={"text": body[:3000]})
        response.raise_for_status()
        logging.info("✅ Slack message sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send Slack message: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily market analysis.")
    parser.add_argument("--output", default="print", choices=["print", "email", "slack"], help="Output destination for the report.")
    args = parser.parse_args()
    main(output=args.output)
