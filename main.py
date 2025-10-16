import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from bs4 import BeautifulSoup
from asyncio_throttle import Throttler
from ta.momentum import RSIIndicator
from ta.trend import MACD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('yfinance').setLevel(logging.WARNING)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
MEMORY_FILE = "market_memory.json"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------

async def make_robust_request(session, url, params=None, retries=3, delay=5, timeout=20):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
            logging.warning(f"Request attempt {attempt + 1} failed for {url}: {e}")
            if attempt < retries - 1: await asyncio.sleep(delay)
    return None

def get_cached_tickers(cache_file, fetch_function):
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400:
        with open(cache_file, 'r') as f: return json.load(f)
    tickers = fetch_function()
    if tickers:
        with open(cache_file, 'w') as f: json.dump(tickers, f)
    return tickers

def fetch_sp500_tickers_sync():
    try:
        df = pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=REQUEST_HEADERS, timeout=15).text))[0]
        return [ticker.replace('.', '-') for ticker in df["Symbol"].tolist()]
    except Exception:
        return [ticker.replace('.', '-') for ticker in pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")["Symbol"].tolist()] or []

def fetch_tsx_tickers_sync():
    try:
        for table in pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", headers=REQUEST_HEADERS, timeout=15).text)):
            if 'Symbol' in table.columns: return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception: return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO"]
    return []

async def fetch_finviz_news_throttled(throttler, session, ticker):
    async with throttler:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        content = await make_robust_request(session, url)
        if not content: return []
        soup = BeautifulSoup(content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return []
        return [{"title": row.a.text, "url": row.a['href']} for row in news_table.find_all('tr') if row.a]

async def fetch_market_headlines():
    logging.info("Fetching market headlines with robust engine...")
    async with aiohttp.ClientSession() as session:
        try:
            content = await make_robust_request(session, "https://www.reuters.com/markets/")
            if content:
                soup = BeautifulSoup(content, 'html.parser')
                links = soup.select('[data-testid="Heading"] a', limit=5)
                headlines = [{"title": link.text.strip(), "url": "https://www.reuters.com" + link['href'], "source": "Reuters"} for link in links if link.text.strip()]
                if headlines:
                    logging.info(f"✅ Fetched {len(headlines)} headlines from Reuters.")
                    return headlines
        except Exception as e:
            logging.warning(f"Reuters scrape failed: {e}. Trying fallback.")
    if NEWSAPI_KEY:
        logging.warning("Falling back to NewsAPI for headlines.")
        async with aiohttp.ClientSession() as session:
            content = await make_robust_request(session, f"https://newsapi.org/v2/top-headlines?category=business&language=en&pageSize=5&apiKey={NEWSAPI_KEY}")
            if content and (articles := json.loads(content).get("articles", [])):
                return [{"title": a['title'], "url": a['url'], "source": a['source']['name']} for a in articles]
    logging.error("❌ All news sources failed.")
    return []

async def fetch_macro_sentiment(session):
    if not NEWSAPI_KEY: return {"geopolitical_risk": 0, "trade_risk": 0, "economic_sentiment": 0, "overall_macro_score": 0, "geo_articles": [], "trade_articles": [], "econ_articles": []}
    async def get_news(query):
        url = f"https://newsapi.org/v2/everything?q=({query})&pageSize=20&language=en&apiKey={NEWSAPI_KEY}"
        content = await make_robust_request(session, url)
        return json.loads(content).get("articles", []) if content else []
    geo_articles, trade_articles, econ_articles = await asyncio.gather(get_news("war OR conflict OR geopolitics"), get_news('"trade war" OR tariffs OR sanctions'), get_news('"interest rates" OR inflation OR recession'))
    geopolitical_risk, trade_risk = min(len(geo_articles) / 20 * 100, 100), min(len(trade_articles) / 15 * 100, 100)
    economic_sentiment = sum(analyzer.polarity_scores(a['title']).get('compound', 0) for a in econ_articles) / len(econ_articles) if econ_articles else 0
    overall_macro_score = -(geopolitical_risk / 100 * 15) - (trade_risk / 100 * 10) + (economic_sentiment * 15)
    logging.info("✅ Macro sentiment analysis complete.")
    return {
        "geopolitical_risk": geopolitical_risk, "trade_risk": trade_risk, "economic_sentiment": economic_sentiment, "overall_macro_score": overall_macro_score,
        "geo_articles": [{"title": a['title'], "url": a['url'], "source": a['source']['name']} for a in geo_articles[:3]],
        "trade_articles": [{"title": a['title'], "url": a['url'], "source": a['source']['name']} for a in trade_articles[:3]],
        "econ_articles": [{"title": a['title'], "url": a['url'], "source": a['source']['name']} for a in econ_articles[:3]]
    }

async def fetch_context_data(session):
    ids = ["bitcoin", "ethereum", "solana", "ripple"]
    url = f"https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&ids={','.join(ids)}"
    content = await make_robust_request(session, url)
    context_data = {item['id']: item for item in json.loads(content)} if content else {}
    try:
        gold_info, silver_info = yf.Ticker('GC=F').info, yf.Ticker('SI=F').info
        context_data['gold'] = {'name': 'Gold', 'symbol': 'GC=F', 'current_price': gold_info.get('regularMarketPrice')}
        context_data['silver'] = {'name': 'Silver', 'symbol': 'SI=F', 'current_price': silver_info.get('regularMarketPrice')}
        if (gp := gold_info.get('regularMarketPrice')) and (sp := silver_info.get('regularMarketPrice')):
            context_data['gold_silver_ratio'] = f"{gp/sp:.1f}:1"
    except Exception: pass
    fg_content = await make_robust_request(session, "https://api.alternative.me/fng/?limit=1")
    context_data['crypto_sentiment'] = json.loads(fg_content)['data'][0]['value_classification'] if fg_content else "N/A"
    return context_data

# RESTORED THIS CRITICAL FUNCTION
def compute_technical_indicators(series):
    if len(series.dropna()) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0))}

async def analyze_stock(semaphore, throttler, session, ticker):
    async with semaphore:
        try:
            yf_ticker = yf.Ticker(ticker)
            data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
            if data.empty: return None
            info = await asyncio.to_thread(getattr, yf_ticker, 'info')
            articles = await fetch_finviz_news_throttled(throttler, session, ticker)
            avg_sent = sum(analyzer.polarity_scores(a["title"]).get("compound", 0) for a in articles) / len(articles) if articles else 0
            score = 50 + (avg_sent * 20)
            if (tech := compute_technical_indicators(data["Close"])):
                if 40 < tech.get("rsi", 50) < 65: score += 15
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 35: score += 15
            return { "ticker": ticker, "score": score, "name": info.get('shortName', ticker), "sector": info.get('sector', 'N/A'), "summary": info.get('longBusinessSummary', None) }
        except Exception as e:
            if '$' not in str(e): logging.error(f"Error in analyze_stock for {ticker}: {e}", exc_info=False)
            return None

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)

async def main(output="print"):
    previous_day_memory = load_memory()
    sp500, tsx = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync), get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    
    throttler = Throttler(2)
    semaphore = asyncio.Semaphore(10)
    
    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_stock(semaphore, throttler, session, ticker) for ticker in universe]
        context_task = fetch_context_data(session)
        news_task = fetch_market_headlines()
        macro_task = fetch_macro_sentiment(session)
        results, context_data, market_news, macro_data = await asyncio.gather(asyncio.gather(*stock_tasks), context_task, news_task, macro_task)

    stock_results = [r for r in results if r]
    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False) if stock_results else pd.DataFrame()
    
    if output == "email":
        html_email = generate_html_email(df_stocks, context_data, market_news, macro_data, previous_day_memory)
        send_email(html_email)
    
    if not df_stocks.empty:
        save_memory({"previous_top_stock_name": df_stocks.iloc[0]['name'], "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'], "previous_macro_score": macro_data.get('overall_macro_score', 0)})

    logging.info("✅ Done.")

def generate_html_email(df_stocks, context, market_news, macro_data, memory):
    def format_articles(articles):
        if not articles: return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        return "<ul style='margin:0;padding-left:20px;'>" + "".join([f'<li style="margin-bottom:5px;"><a href="{a["url"]}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a["source"]})</span></li>' for a in articles]) + "</ul>"
    def create_stock_table(df):
        return "".join([f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])
    def create_context_table(ids):
        rows=""
        for asset_id in ids:
            if asset := context.get(asset_id):
                price, change_24h = f"${asset.get('current_price', 0):,.2f}", asset.get('price_change_percentage_24h', 0) or 0
                mcap = f"${asset.get('market_cap', 0) / 1_000_000_000:.1f}B" if asset.get('market_cap') else "N/A"
                color_24h = "#16a34a" if change_24h >= 0 else "#dc2626"
                rows += f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{asset.get("name", "")}</b><br><span style="color:#666;font-size:0.9em;">{asset.get("symbol","").upper()}</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{price}<br><span style="color:{color_24h};font-size:0.9em;">{change_24h:.2f}% (24h)</span></td><td style="padding:10px;border-bottom:1px solid #eee;">{mcap}</td></tr>'
        return rows

    prev_score, current_score = memory.get('previous_macro_score', 0), macro_data.get('overall_macro_score', 0)
    mood_change = "stayed relatively stable"
    if (diff := current_score - prev_score) > 3: mood_change = f"improved since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    elif diff < -3: mood_change = f"turned more cautious since yesterday (from {prev_score:.1f} to {current_score:.1f})"
    editor_note = f"Good morning. The overall market mood has {mood_change}. This briefing is your daily blueprint for navigating the currents."
    if memory.get('previous_top_stock_name'): editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings. Let's see what's changed."

    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector', group_keys=False).apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if not(row['sector'] and row['sector'] != 'N/A'): continue
            summary_text = "Business summary not available."
            if row["summary"] and isinstance(row["summary"], str): summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.'
            sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'

    top10_html, bottom10_html = create_stock_table(df_stocks.head(10)), create_stock_table(df_stocks.tail(10).iloc[::-1])
    crypto_html, commodities_html = create_context_table(["bitcoin", "ethereum", "solana", "ripple"]), create_context_table(["gold", "silver"])
    market_news_html = "".join([f'<div style="margin-bottom:15px;"><b><a href="{a["url"]}" style="color:#000;">{a["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{a.get("source", "N/A")}</span></div>' for a in market_news]) or "<p><i>Headlines not available today.</i></p>"

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} h2{{font-size:1.5em;color:#111;margin-top:0;}} h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>{editor_note}</p></div>
    <div class="section"><h2>THE BIG PICTURE: The Market Weather Report</h2>
        <h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3>
        <p><b>How it's calculated:</b> This is our "weather forecast" for investors, combining risks and sentiment. A positive score suggests optimism ("risk-on"), while negative signals caution ("risk-off"). It's a blend of the three scores below.</p>
        <p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> Measures global instability by scanning news for conflict-related keywords. High scores favor safe-havens like Gold.<br><u>Key Drivers:</u> {format_articles(macro_data['geo_articles'])}</p>
        <p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Tracks mentions of 'trade war', 'tariffs', etc. High risk can hurt multinational companies.<br><u>Key Drivers:</u> {format_articles(macro_data['trade_articles'])}</p>
        <p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> Analyzes the emotional tone of news about inflation, rates, and growth (-1 bearish, +1 bullish).<br><u>Key Drivers:</u> {format_articles(macro_data['econ_articles'])}</p>
    </div>
    <div class="section"><h2>SECTOR DEEP DIVE</h2><p>Here are the top-scoring companies from different sectors, giving a cross-section of the market's strongest narratives.</p>{sector_html}</div>
    <div class="section"><h2>STOCK RADAR</h2>
        <h3>📈 Top 10 Strongest Signals</h3><p><b>How it's calculated:</b> Stocks are scored (0-100) on a blend of valuation (e.g., P/E ratio), momentum (e.g., RSI), and recent news sentiment. High scores indicate strength across multiple factors.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table>
        <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><p>These stocks are facing headwinds. This is a prompt to investigate why.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table>
    </div>
    <div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2>
        <h3>🪙 Crypto: The Digital Frontier</h3><p><b>Market Sentiment: <span style="font-weight:bold;">{context.get('crypto_sentiment', 'N/A')}</span></b> (via Fear & Greed Index). Shows investor emotion, from Extreme Fear (potential buying opportunity) to Extreme Greed (market may be due for a correction).</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 30px;">💎 Commodities: The Bedrock Assets</h3><p><b>Key Insight: <span style="font-weight:bold;">{context.get('gold_silver_ratio', 'N/A')}</span></b>. Shows how many ounces of silver it takes to buy one ounce of gold. A high number suggests silver is undervalued relative to gold, and vice-versa.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap</th></tr></thead><tbody>{commodities_html}</tbody></table>
    </div>
    <div class="section"><h2>FROM THE WIRE: Today's Top Headlines</h2>{market_news_html}</div>
    </div></body></html>
    """

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS: logging.warning("SMTP creds missing."); return
    msg = MIMEMultipart('alternative')
    msg["Subject"], msg["From"], msg["To"] = f"⛵ Your Daily Market Briefing - {datetime.date.today()}", SMTP_USER, SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(); server.login(SMTP_USER, SMTP_PASS); server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e: logging.error(f"Failed to send email: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="print", choices=["print", "email"])
    args = parser.parse_args()
    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(output=args.output))
