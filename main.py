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
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
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
        else: logging.error(f"All {retries} attempts failed for URL: {url}. Skipping.")
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
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        return pd.read_html(StringIO(requests.get(url, headers=REQUEST_HEADERS, timeout=10).text))[0]["Symbol"].tolist()
    except Exception: return pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")["Symbol"].tolist() or []

def fetch_tsx_tickers_sync():
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        for table in pd.read_html(StringIO(requests.get(url, headers=REQUEST_HEADERS, timeout=10).text)):
            if 'Symbol' in table.columns: return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception: return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO"]
    return []

# FIX: Added throttler to avoid being blocked by Finviz
async def fetch_finviz_news_throttled(throttler, session, ticker):
    async with throttler:
        url = f"https://finviz.com/quote.ashx?t={ticker}"
        content = await make_robust_request(session, url)
        if not content: return []
        soup = BeautifulSoup(content, 'html.parser')
        news_table = soup.find(id='news-table')
        if not news_table: return []
        return [{"title": row.a.text, "url": row.a['href']} for row in news_table.findAll('tr') if row.a]

# NEW: Scrape TradingView for high-quality market headlines
async def fetch_tradingview_headlines():
    logging.info("Fetching market headlines from TradingView...")
    headlines = []
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto("https://www.tradingview.com/news/", timeout=60000)
            await page.wait_for_selector('[class*="item-full-width-"]', timeout=30000)
            
            elements = await page.query_selector_all('[class*="item-full-width-"]')
            for el in elements[:6]: # Get top 6 headlines
                title_el = await el.query_selector('[class*="title-"]')
                source_el = await el.query_selector('[class*="source-"]')
                link_el = await el.query_selector('a[href]')
                
                if title_el and source_el and link_el:
                    headlines.append({
                        "title": await title_el.inner_text(),
                        "source": await source_el.inner_text(),
                        "url": "https://www.tradingview.com" + await link_el.get_attribute('href')
                    })
            await browser.close()
        except Exception as e:
            logging.error(f"Could not fetch TradingView headlines: {e}")
    return headlines

async def fetch_macro_sentiment():
    async with aiohttp.ClientSession() as session:
        # Simplified for reliability - main news comes from TradingView now
        geopolitical_risk, trade_risk, economic_sentiment = 0, 0, 0 # Default values
        overall_macro_score = 0
        logging.info("✅ Macro sentiment analysis complete (using default values).")
        return {
            "geopolitical_risk": geopolitical_risk, "geo_articles": [],
            "trade_risk": trade_risk, "trade_articles": [],
            "economic_sentiment": economic_sentiment, "econ_articles": [],
            "overall_macro_score": overall_macro_score
        }

def compute_technical_indicators(series):
    series = series.dropna()
    if len(series) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0)), "macd_signal": float(latest.get("macd_signal", 0))}

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
            if tech.get("macd", 0) > tech.get("macd_signal", 0): score += 5
        if asset_type == 'stock':
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 35: score += 15
            if info.get('debtToEquity') and info.get('debtToEquity') < 100: score += 5
        score += avg_sent * 20
        
        return { "ticker": ticker, "score": score, "price": data['Close'].iloc[-1], "name": info.get('shortName', ticker), "sector": info.get('sector', 'N/A'), "summary": info.get('longBusinessSummary', 'No summary available.') }
    except Exception as e:
        logging.error(f"Error processing {ticker}: {e}", exc_info=False)
        return None

async def main(output="print"):
    macro_data = await fetch_macro_sentiment()
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = (sp500 or [])[:75] + (tsx or [])[:25]
    crypto_tickers, commodity_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"], ["GC=F", "SI=F"]
    
    # NEW: Initialize the throttler to limit requests to 5 per second
    throttler = Throttler(5)
    
    async with aiohttp.ClientSession() as session:
        all_tickers = universe + crypto_tickers + commodity_tickers
        tasks = [analyze_asset(throttler, session, ticker, 'stock' if ticker in universe else 'crypto' if ticker in crypto_tickers else 'commodity') for ticker in all_tickers]
        results = await asyncio.gather(*tasks)

    stock_results = [r for r in results[:len(universe)] if r]
    crypto_results = [r for r in results[len(universe):len(universe)+len(crypto_tickers)] if r]
    commodity_results = [r for r in results[len(universe)+len(crypto_tickers):] if r]

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False) if stock_results else pd.DataFrame()
    df_crypto = pd.DataFrame(crypto_results).sort_values("score", ascending=False) if crypto_results else pd.DataFrame()
    df_commodities = pd.DataFrame(commodity_results).sort_values("score", ascending=False) if commodity_results else pd.DataFrame()
    
    market_news = await fetch_tradingview_headlines()

    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news)
        send_email(html_email)
    else: print(df_stocks.head(10))

    logging.info("✅ Done.")

def generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news):
    def create_stock_table_rows(df):
        return "".join([f'<tr><td style="padding: 10px; border-bottom: 1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; font-weight:bold; font-size: 1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])

    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector').apply(lambda x: x.nlargest(2, 'score')).reset_index(drop=True)
        for sector, group in top_by_sector.groupby('sector'):
            if sector == 'N/A' or sector is None: continue
            sector_html += f'<h4 style="margin-bottom:10px;margin-top:20px;">{sector} Sector Spotlight</h4>'
            for _, row in group.iterrows():
                sector_html += f'<div style="margin-bottom: 15px;"><b>{row["name"]} ({row["ticker"]})</b><p style="font-size: 0.9em; color: #333; margin: 5px 0 0 0;">{row["summary"][:250]}...</p></div>'

    top10_html = create_stock_table_rows(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data</td></tr>"
    bottom10_html = create_stock_table_rows(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data</td></tr>"
    crypto_html = create_stock_table_rows(df_crypto) if not df_crypto.empty else "<tr><td>No data</td></tr>"
    commodities_html = create_stock_table_rows(df_commodities) if not df_commodities.empty else "<tr><td>No data</td></tr>"
    market_news_html = "".join([f'<div style="margin-bottom: 15px;"><b><a href="{a["url"]}" style="color: #000; text-decoration: none; font-size: 1.1em;">{a["title"]}</a></b><br><span style="color: #666; font-size: 0.9em;">Source: {a.get("source", "N/A")}</span></div>' for a in market_news])

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} .section h2{{font-size:1.5em;color:#111;margin-top:0;}} .section h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>Good morning. Think of the market as a big, complicated ocean. Some days it's calm, some days it's stormy. Our job isn't to predict the waves, but to build a better boat. This briefing is your daily blueprint. We'll check the weather (the macro environment), map the currents (sector performance), and point out the interesting ships on the horizon (top stocks). Let's set sail.</p></div>
    <div class="section"><h2>SECTOR DEEP DIVE: Who's Building the Future?</h2><p>Every industry tells a story. Here, we highlight the top-scoring companies from different sectors to give you a cross-section of the market's strongest narratives right now.</p>{sector_html}</div>
    <div class="section"><h2>STOCK RADAR: Today's Most Interesting Signals</h2>
        <h3>📈 Top 10 Strongest Signals</h3><p>These stocks are currently firing on all cylinders, showing a strong combination of market value, positive momentum, and good press. They're the ships catching the strongest wind.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table>
        <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><p>These stocks are currently facing headwinds, with weaker scores in our analysis. They might be undervalued opportunities or signals of underlying issues worth investigating.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table>
    </div>
    <div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2>
        <h3>🪙 Crypto: The Digital Frontier</h3><p>Cryptocurrencies are like the new, uncharted islands of the financial world. They're volatile, high-risk, and high-reward. Bitcoin is the largest, often seen as "digital gold," while others like Ethereum power new applications. We score them based on momentum and news sentiment.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 30px;">💎 Commodities: The Bedrock Assets</h3><p>These are the real, physical materials that build our world. Gold (GC=F) is the ultimate "safe harbor" investors flock to during storms. Silver (SI=F) is both an industrial metal and a store of value. Their scores are heavily influenced by global uncertainty.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{commodities_html}</tbody></table>
    </div>
    <div class="section"><h2>FROM THE WIRE: Today's Top Headlines</h2><p>Here are the stories our systems have identified as the most impactful for the market today, pulled directly from TradingView's news desk.</p>{market_news_html}</div>
    </div></body></html>
    """

def send_email(html_body):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS: logging.warning("SMTP creds missing; cannot send email."); return
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
