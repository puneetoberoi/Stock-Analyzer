import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
from asyncio_throttle import Throttler
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
MEMORY_FILE = "market_memory.json"
analyzer = SentimentIntensityAnalyzer()

# ---------- helpers ----------

async def make_robust_request(session, url, params=None, retries=3, delay=5, timeout=20):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout) as response:
                response.raise_for_status()
                return await response.text()
        except asyncio.TimeoutError: logging.warning(f"Timeout on attempt {attempt + 1}/{retries} for {url}")
        except aiohttp.ClientError as e: logging.warning(f"Request error on attempt {attempt + 1}/{retries} for {url}: {e}")
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
        return pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers=REQUEST_HEADERS, timeout=10).text))[0]["Symbol"].tolist()
    except Exception: return pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")["Symbol"].tolist() or []

def fetch_tsx_tickers_sync():
    try:
        for table in pd.read_html(StringIO(requests.get("https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index", headers=REQUEST_HEADERS, timeout=10).text)):
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
        return [{"title": row.a.text, "url": row.a['href']} for row in news_table.findAll('tr') if row.a]

async def fetch_tradingview_headlines():
    logging.info("Fetching market headlines from TradingView...")
    headlines = []
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://www.tradingview.com/news/", timeout=60000)
            try:
                accept_button = page.get_by_role("button", name="Accept all cookies")
                if await accept_button.is_visible(timeout=5000): await accept_button.click()
            except Exception: logging.info("No cookie pop-up found or handled.")
            await page.wait_for_selector("article a[href*='/news/']", timeout=30000)
            elements = await page.locator("article a[href*='/news/']").all()
            for el in elements[:5]:
                if title := await el.get_attribute("title"):
                    headlines.append({ "title": title, "source": await el.locator('span >> nth=0').inner_text(), "url": "https://www.tradingview.com" + await el.get_attribute('href')})
            await browser.close()
        except Exception as e: logging.error(f"Could not fetch TradingView headlines: {e}")
    logging.info(f"✅ Fetched {len(headlines)} headlines from TradingView.")
    return headlines

# NEW: Fetching deep context for crypto/commodities
async def fetch_coingecko_data(session, ids=["bitcoin", "ethereum", "solana", "ripple", "gold", "silver"]):
    logging.info(f"Fetching context data for {ids} from CoinGecko...")
    url = f"https://api.coingecko.com/api/v3/coins/markets"
    params = {"vs_currency": "usd", "ids": ",".join(ids), "order": "market_cap_desc"}
    content = await make_robust_request(session, url, params)
    if not content: return {}
    data = json.loads(content)
    
    # Also fetch Fear & Greed Index
    fg_content = await make_robust_request(session, "https://api.alternative.me/fng/?limit=1")
    fg_index = json.loads(fg_content)['data'][0]['value_classification'] if fg_content else "N/A"

    context_data = {item['id']: item for item in data}
    context_data['crypto_sentiment'] = fg_index
    
    # Calculate Gold/Silver Ratio
    if 'gold' in context_data and 'silver' in context_data:
        gold_price = context_data['gold'].get('current_price', 0)
        silver_price = context_data['silver'].get('current_price', 0)
        context_data['gold_silver_ratio'] = f"{gold_price/silver_price:.1f}:1" if silver_price > 0 else "N/A"
        
    return context_data

def compute_technical_indicators(series):
    if len(series.dropna()) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0))}

async def analyze_asset(throttler, session, ticker, asset_type='stock'):
    try:
        yf_ticker = yf.Ticker(ticker)
        data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
        if data.empty: return None
        info = await asyncio.to_thread(getattr, yf_ticker, 'info')
        
        tech = compute_technical_indicators(data["Close"])
        articles = await fetch_finviz_news_throttled(throttler, session, ticker)
        avg_sent = sum([analyzer.polarity_scores(a["title"]).get("compound", 0) for a in articles]) / len(articles) if articles else 0
        
        score = 50
        if tech:
            if 40 < tech.get("rsi", 50) < 65: score += 10
            if tech.get("macd", 0) > 0: score += 5
        if asset_type == 'stock':
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 35: score += 15
        score += avg_sent * 20
        
        return { "ticker": ticker, "score": score, "name": info.get('shortName', ticker), "sector": info.get('sector', 'N/A'), "summary": info.get('longBusinessSummary', 'No summary available.') }
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}", exc_info=False)
        return None

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f)

async def main(output="print"):
    # Load yesterday's data
    previous_day_memory = load_memory()

    sp500, tsx = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync), get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    crypto_tickers, commodity_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"], ["GC=F", "SI=F"]
    
    throttler = Throttler(5)
    
    async with aiohttp.ClientSession() as session:
        # Fetch all data concurrently
        asset_tasks = [analyze_asset(throttler, session, ticker) for ticker in universe + crypto_tickers + commodity_tickers]
        context_task = fetch_coingecko_data(session)
        news_task = fetch_tradingview_headlines()
        
        results, context_data, market_news = await asyncio.gather(asyncio.gather(*asset_tasks), context_task, news_task)

    stock_results = [r for r in results[:len(universe)] if r]
    crypto_results = [r for r in results[len(universe):len(universe)+len(crypto_tickers)] if r]
    commodity_results = [r for r in results[len(universe)+len(crypto_tickers):] if r]

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False) if stock_results else pd.DataFrame()
    df_crypto = pd.DataFrame(crypto_results).sort_values("score", ascending=False) if crypto_results else pd.DataFrame()
    df_commodities = pd.DataFrame(commodity_results).sort_values("score", ascending=False) if commodity_results else pd.DataFrame()

    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, df_commodities, context_data, market_news, previous_day_memory)
        send_email(html_email)
    
    # Save today's data for tomorrow
    if not df_stocks.empty:
        new_memory = {
            "previous_top_stock_name": df_stocks.iloc[0]['name'],
            "previous_top_stock_ticker": df_stocks.iloc[0]['ticker']
        }
        save_memory(new_memory)

    logging.info("✅ Done.")

def generate_html_email(df_stocks, df_crypto, df_commodities, context, market_news, memory):
    def create_stock_table_rows(df):
        return "".join([f'<tr><td style="padding: 10px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:bold; font-size: 1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])

    def create_context_table(df, ids):
        rows = ""
        for _, row in df.iterrows():
            asset_id = ids.get(row['ticker'])
            if asset_id and asset_id in context:
                asset_context = context[asset_id]
                price = f"${asset_context['current_price']:,.2f}"
                change_24h = asset_context['price_change_percentage_24h']
                change_7d = asset_context.get('price_change_percentage_7d_in_currency', 0)
                market_cap = f"${asset_context['market_cap'] / 1_000_000_000:.1f}B"
                color_24h = "#16a34a" if change_24h >= 0 else "#dc2626"
                rows += f'<tr><td style="padding: 10px; border-bottom: 1px solid #eee;"><b>{row["name"]}</b><br><span style="color:#666;font-size:0.9em;">{row["ticker"]}</span></td><td style="padding: 10px; border-bottom: 1px solid #eee;">{price}<br><span style="color:{color_24h};font-size:0.9em;">{change_24h:.2f}% (24h)</span></td><td style="padding: 10px; border-bottom: 1px solid #eee;">{market_cap}<br><span style="font-size:0.9em;">{change_7d:.2f}% (7d)</span></td></tr>'
        return rows

    # Editors note with memory
    editor_note = "Good morning. Think of the market as a big, complicated ocean. Some days it's calm, some days it's stormy. Our job isn't to predict the waves, but to build a better boat. This briefing is your daily blueprint."
    if memory.get('previous_top_stock_name'):
        editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings. Let's see what's changed on the leaderboard today."

    # Sector deep dive
    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector').apply(lambda x: x.nlargest(1, 'score')).reset_index(drop=True)
        for _, row in top_by_sector.iterrows():
            if row['sector'] == 'N/A' or row['sector'] is None: continue
            sentences = row["summary"].split('. ')
            short_summary = '. '.join(sentences[:2]) + '.'
            sector_html += f'<div style="margin-bottom: 15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size: 0.9em; color: #333; margin: 5px 0 0 0;">{short_summary}</p></div>'

    top10_html, bottom10_html = create_stock_table_rows(df_stocks.head(10)), create_stock_table_rows(df_stocks.tail(10).iloc[::-1])
    crypto_html = create_context_table(df_crypto, {"BTC-USD":"bitcoin", "ETH-USD":"ethereum", "SOL-USD":"solana", "XRP-USD":"ripple"})
    commodities_html = create_context_table(df_commodities, {"GC=F": "gold", "SI=F": "silver"})
    market_news_html = "".join([f'<div style="margin-bottom: 15px;"><b><a href="{a["url"]}" style="color: #000; text-decoration: none; font-size: 1.1em;">{a["title"]}</a></b><br><span style="color: #666; font-size: 0.9em;">{a.get("source", "N/A")}</span></div>' for a in market_news]) or "<p><i>Headlines not available.</i></p>"

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} .section h2{{font-size:1.5em;color:#111;margin-top:0;}} .section h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>{editor_note}</p></div>
    <div class="section"><h2>SECTOR DEEP DIVE</h2><p>Here, we highlight a top-scoring company from different sectors to give you a cross-section of the market's strongest narratives.</p>{sector_html}</div>
    <div class="section"><h2>STOCK RADAR</h2>
        <h3>📈 Top 10 Strongest Signals</h3><p>These stocks are firing on all cylinders, showing a strong combination of value, positive momentum, and good press.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table>
        <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><p>These stocks are facing headwinds, with weaker scores in our analysis. This is a prompt to investigate why.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table>
    </div>
    <div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2>
        <h3>🪙 Crypto: The Digital Frontier</h3><p><b>Market Sentiment: <span style="font-weight:bold;">{context.get('crypto_sentiment', 'N/A')}</span></b> (via Fear & Greed Index). This meter shows investor emotion, from Extreme Fear (potential buying opportunity) to Extreme Greed (market may be due for a correction).</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap / 7d</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 30px;">💎 Commodities: The Bedrock Assets</h3><p><b>Key Insight: <span style="font-weight:bold;">Gold/Silver Ratio at {context.get('gold_silver_ratio', 'N/A')}</span></b>. This ratio shows how many ounces of silver it takes to buy one ounce of gold. A high number suggests silver is undervalued relative to gold, and vice-versa.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:left; padding:10px;">Price / 24h</th><th style="text-align:left; padding:10px;">Market Cap / 7d</th></tr></thead><tbody>{commodities_html}</tbody></table>
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
    parser = argparse.ArgumentParser(description="Run the daily market analysis.")
    parser.add_argument("--output", default="print", choices=["print", "email"], help="Output destination.")
    args = parser.parse_args()
    asyncio.run(main(output=args.output))
