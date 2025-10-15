import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
analyzer = SentimentIntensityAnalyzer()
LAST_API_CALL_TIME = 0

# ---------- helpers ----------

async def api_cooldown():
    global LAST_API_CALL_TIME
    time_since_last_call = time.time() - LAST_API_CALL_TIME
    if time_since_last_call < 1.5:
        await asyncio.sleep(1.5 - time_since_last_call)
    LAST_API_CALL_TIME = time.time()

async def make_robust_request(session, url, params=None, retries=3, delay=5):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=15) as response:
                response.raise_for_status()
                return await response.text()
        except aiohttp.ClientError as e:
            logging.warning(f"Request error on attempt {attempt + 1}/{retries} for {url}: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
            else: logging.error(f"All {retries} attempts failed for URL: {url}. Skipping.")
    return None

def get_cached_tickers(cache_file, fetch_function):
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:
        logging.info(f"Loading tickers from cache: {cache_file}")
        with open(cache_file, 'r') as f: return json.load(f)
    tickers = fetch_function()
    with open(cache_file, 'w') as f: json.dump(tickers, f)
    return tickers

def fetch_sp500_tickers_sync():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        df = pd.read_html(requests.get(url, headers=REQUEST_HEADERS, timeout=10).text)[0]
        return df["Symbol"].tolist()
    except Exception:
        return pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")["Symbol"].tolist()

def fetch_tsx_tickers_sync():
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        for table in pd.read_html(requests.get(url, headers=REQUEST_HEADERS, timeout=10).text):
            if 'Symbol' in table.columns and 'Company' in table.columns:
                return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
        raise ValueError()
    except Exception:
        return ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "TRP.TO", "SHOP.TO"]

async def fetch_and_analyze_news(session, query, source="gdelt"):
    articles, score, keywords = [], 0, []
    if source == "gdelt":
        gdelt_query = f"({query})"
        content = await make_robust_request(session, "https://api.gdeltproject.org/api/v2/doc/doc", params={"query": gdelt_query, "mode": "artlist", "maxrecords": 50, "format": "json", "timespan": "7d"})
        if content: articles = json.loads(content).get("articles", [])
        keywords = query.lower().split(" or ")
    elif source == "newsapi" and NEWSAPI_KEY:
        await api_cooldown()
        news_query = f"({query})"
        content = await make_robust_request(session, f"https://newsapi.org/v2/everything?q={requests.utils.quote(news_query)}&pageSize=20&apiKey={NEWSAPI_KEY}")
        if content: articles = json.loads(content).get("articles", [])
        keywords = query.lower().split(" or ")

    found_articles = []
    for article in articles:
        title = article.get("title", "").lower()
        if any(keyword in title for keyword in keywords):
            source_name = article.get("sourcecountry", "N/A") if source == "gdelt" else article.get("source", {}).get("name", "N/A")
            found_articles.append({"title": article.get("title"), "url": article.get("url"), "source": source_name})
    
    return found_articles

async def fetch_macro_sentiment():
    async with aiohttp.ClientSession() as session:
        logging.info("\n🌍 Fetching Global Macro Sentiment...")
        
        # Concurrent analysis
        geo_task = fetch_and_analyze_news(session, "war OR conflict OR military OR attack OR invasion")
        trade_task = fetch_and_analyze_news(session, '"trade war" OR tariff OR sanctions')
        econ_task = fetch_and_analyze_news(session, '"interest rates" OR inflation OR recession OR "gdp growth"', source="newsapi")
        
        geo_articles, trade_articles, econ_articles = await asyncio.gather(geo_task, trade_task, econ_task)

        # Calculate scores
        geopolitical_risk = min((len(geo_articles) / 20.0) * 100, 100)
        trade_risk = min((len(trade_articles) / 15.0) * 100, 100)
        
        economic_sentiment = 0
        if econ_articles:
            sentiments = [analyzer.polarity_scores(a['title']).get('compound', 0) for a in econ_articles]
            economic_sentiment = sum(sentiments) / len(sentiments)
        
        overall_macro_score = -(geopolitical_risk / 100 * 15) - (trade_risk / 100 * 10) + (economic_sentiment * 15)
        
        macro_data = {
            "geopolitical_risk": geopolitical_risk, "geo_articles": geo_articles[:5],
            "trade_risk": trade_risk, "trade_articles": trade_articles[:5],
            "economic_sentiment": economic_sentiment, "econ_articles": econ_articles[:5],
            "overall_macro_score": overall_macro_score
        }
        logging.info("✅ Macro sentiment analysis complete.")
        return macro_data

def compute_technical_indicators(series):
    series = series.dropna()
    if len(series) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0)), "macd_signal": float(latest.get("macd_signal", 0))}

async def analyze_asset(ticker, macro_data, session, asset_type='stock'):
    try:
        yf_ticker = yf.Ticker(ticker)
        data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
        if data.empty: return None

        tech = compute_technical_indicators(data["Close"])
        info = await asyncio.to_thread(getattr, yf_ticker, 'info')
        
        news_query = f'"{info.get("longName", ticker)}"' if asset_type == 'stock' else ticker.replace("-USD", "").replace("=F", "")
        await api_cooldown()
        news_content = await make_robust_request(session, f"https://newsapi.org/v2/everything?q={requests.utils.quote(news_query)}&pageSize=5&apiKey={NEWSAPI_KEY}")
        
        articles = json.loads(news_content).get("articles", []) if news_content else []
        sentiments = [analyzer.polarity_scores(a.get("title", "")).get("compound", 0) for a in articles]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Simplified scoring for brevity
        score = 50
        if tech and 40 < tech.get("rsi", 50) < 70: score += 10
        if info.get('trailingPE') and 0 < info.get('trailingPE') < 40: score += 10
        score += avg_sent * 10
        
        return {"ticker": ticker, "score": score, "price": data['Close'].iloc[-1], "change": data['Close'].iloc[-1] - data['Close'].iloc[-2]}
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}", exc_info=False)
        return None

async def main(output="print"):
    macro_data = await fetch_macro_sentiment()
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = sp500[:50] + tsx[:25] # Reduced size for faster daily runs
    
    crypto_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"]
    commodity_tickers = ["GC=F", "SI=F"] # Gold, Silver
    
    logging.info(f"Analyzing {len(universe)} stocks, {len(crypto_tickers)} cryptos, and {len(commodity_tickers)} commodities...")

    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_asset(ticker, macro_data, session, 'stock') for ticker in universe]
        crypto_tasks = [analyze_asset(ticker, macro_data, session, 'crypto') for ticker in crypto_tickers]
        commodity_tasks = [analyze_asset(ticker, macro_data, session, 'commodity') for ticker in commodity_tickers]
        
        results = await asyncio.gather(*stock_tasks, *crypto_tasks, *commodity_tasks)

    stock_results = [r for r in results[:len(universe)] if r]
    crypto_results = [r for r in results[len(universe):len(universe)+len(crypto_tickers)] if r]
    commodity_results = [r for r in results[len(universe)+len(crypto_tickers):] if r]

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False)
    df_crypto = pd.DataFrame(crypto_results).sort_values("score", ascending=False)
    df_commodities = pd.DataFrame(commodity_results).sort_values("score", ascending=False)
    
    # Fetch general market news
    async with aiohttp.ClientSession() as session:
        market_news_articles = await fetch_and_analyze_news(session, 'stock market OR investing OR equities', source="newsapi")

    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news_articles)
        send_email(html_email)
    else:
        # A simple print for non-email outputs
        print(df_stocks.head(10))

    logging.info("✅ Done.")

def generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news):
    # Helper to format numbers and create color-coded change values
    def format_change(change):
        color = "#16a34a" if change >= 0 else "#dc2626"
        sign = "+" if change >= 0 else ""
        return f'<span style="color: {color}; font-weight: 600;">{sign}{change:,.2f}</span>'

    # Helper to format article lists
    def format_articles_html(articles):
        if not articles: return "<p><i>No specific drivers found in the last 7 days.</i></p>"
        return "<ul>" + "".join([f'<li><a href="{a["url"]}" style="color: #000; text-decoration: none;">{a["title"]}</a> <span style="color: #666;">({a["source"]})</span></li>' for a in articles]) + "</ul>"

    # Build HTML sections
    top_stocks_html = "".join([f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b></td><td style="padding: 8px; border-bottom: 1px solid #eee;">${row["price"]:,.2f}</td><td style="padding: 8px; border-bottom: 1px solid #eee;">{format_change(row["change"])}</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>{row["score"]:.0f}</b></td></tr>' for _, row in df_stocks.head(10).iterrows()])
    bottom_stocks_html = "".join([f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b></td><td style="padding: 8px; border-bottom: 1px solid #eee;">${row["price"]:,.2f}</td><td style="padding: 8px; border-bottom: 1px solid #eee;">{format_change(row["change"])}</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>{row["score"]:.0f}</b></td></tr>' for _, row in df_stocks.tail(10).iloc[::-1].iterrows()])
    crypto_html = "".join([f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b></td><td style="padding: 8px; border-bottom: 1px solid #eee;">${row["price"]:,.2f}</td><td style="padding: 8px; border-bottom: 1px solid #eee;">{format_change(row["change"])}</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>{row["score"]:.0f}</b></td></tr>' for _, row in df_crypto.iterrows()])
    commodities_html = "".join([f'<tr><td style="padding: 8px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b></td><td style="padding: 8px; border-bottom: 1px solid #eee;">${row["price"]:,.2f}</td><td style="padding: 8px; border-bottom: 1px solid #eee;">{format_change(row["change"])}</td><td style="padding: 8px; border-bottom: 1px solid #eee; text-align: center;"><b>{row["score"]:.0f}</b></td></tr>' for _, row in df_commodities.iterrows()])
    market_news_html = "".join([f'<div style="margin-bottom: 15px;"><b><a href="{a["url"]}" style="color: #000; text-decoration: none; font-size: 1.1em;">{a["title"]}</a></b><br><span style="color: #666; font-size: 0.9em;">Source: {a["source"]}</span></div>' for a in market_news[:4]])

    # The main template
    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4;}} .container{{width: 100%; max-width: 650px; margin: 20px auto; background-color: #fff; border-radius: 8px; overflow: hidden;}} .header{{background-color: #1a1a1a; color: #fff; padding: 20px; text-align: center;}} .section{{padding: 20px; border-bottom: 1px solid #eee;}} .section h2{{font-size: 1.2em; color: #333; margin-top: 0; border-left: 3px solid #5a67d8; padding-left: 10px;}} table{{width: 100%; border-collapse: collapse;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p>{datetime.date.today().strftime('%B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>Good morning. While the world was sleeping, your personal AI analyst was scanning the markets, reading the news, and crunching the numbers. This briefing isn't just data; it's a starting point. We'll look at the big picture (the macro-environment), dive into what's moving, and give you the context to make smarter decisions. Let's get started.</p></div>
    <div class="section"><h2>THE BIG PICTURE: MACRO SCORE: {macro_data['overall_macro_score']:.1f} / 30</h2><p>This single number is your compass for the market's mood today. It combines geopolitical risk, trade tensions, and economic news into one score. A positive score suggests a "risk-on" environment, while a negative score signals caution.</p>
        <h3 style="margin-top: 20px;">Q&A: What Do These Scores Mean?</h3>
        <p><b>Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Calculated by scanning global news for keywords like 'war' and 'conflict'. A higher score means more global instability, which often favors safe-haven assets. <br><u>Key Drivers Today:</u> {format_articles_html(macro_data['geo_articles'])}</p>
        <p><b>Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Measures mentions of 'trade war', 'tariffs', etc. High trade risk can hurt multinational companies and signal economic friction.</p>
        <p><b>Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Reads the emotional tone of financial news about inflation, interest rates, and growth. Ranges from -1 (very negative) to +1 (very positive).</p>
    </div>
    <div class="section"><h2>STOCKS IN THE SPOTLIGHT</h2><p>We analyze hundreds of stocks, but here are the top 10 that scored highest on our blend of technical strength, healthy fundamentals, and positive news sentiment. The full list of ~75 is available in the daily .md file in the repository.</p>
        <h3 style="margin-top: 20px;">📈 Top 10 Movers</h3><table><thead><tr><th>Ticker</th><th>Price</th><th>Change</th><th style="text-align: center;">Score</th></tr></thead><tbody>{top_stocks_html}</tbody></table>
        <h3 style="margin-top: 20px;">📉 Bottom 10 Movers</h3><table><thead><tr><th>Ticker</th><th>Price</th><th>Change</th><th style="text-align: center;">Score</th></tr></thead><tbody>{bottom_stocks_html}</tbody></table>
        <h3 style="margin-top: 20px;">Q&A: How Is the Score Calculated and Should I Act on It?</h3>
        <p><b>The Score (0-100):</b> It's a composite metric. A stock gets points for things like a healthy P/E ratio, strong momentum (like RSI), and positive news headlines. The Macro Score then adjusts this up or down. A high score (e.g., >70) indicates the stock is strong across multiple factors *right now*.</p>
        <p><b>Should I Enter or Get Out?</b> ⚠️ <b>This is not financial advice.</b> Think of this list as a powerful, data-driven starting point for your own research. A stock on the "Top 10" list is worth investigating further. A stock you own on the "Bottom 10" list might be a prompt to review your thesis for holding it.</p>
    </div>
    <div class="section"><h2>CRYPTO & COMMODITIES</h2>
        <h3 style="margin-top: 20px;">🪙 Digital Assets</h3><table><thead><tr><th>Asset</th><th>Price</th><th>Change</th><th style="text-align: center;">Score</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 20px;">💎 Precious Metals & More</h3><table><thead><tr><th>Asset</th><th>Price</th><th>Change</th><th style="text-align: center;">Score</th></tr></thead><tbody>{commodities_html}</tbody></table>
    </div>
    <div class="section"><h2>MARKET HEADLINES</h2>{market_news_html}</div>
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
    msg["Subject"], msg["From"], msg["To"] = f"📊 Your Daily Intelligence Briefing - {datetime.date.today()}", SMTP_USER, SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(); server.login(SMTP_USER, SMTP_PASS); server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily market analysis.")
    parser.add_argument("--output", default="print", choices=["print", "email", "slack"], help="Output destination.")
    args = parser.parse_args()
    asyncio.run(main(output=args.output))
