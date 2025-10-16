import os, sys, argparse, time, datetime, logging, json, asyncio
import pandas as pd
import yfinance as yf
import aiohttp
from ta.momentum import RSIIndicator
from ta.trend import MACD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('yfinance').setLevel(logging.WARNING)
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
MEMORY_FILE = "market_memory.json"
WEEKLY_STATE_FILE = "weekly_state.json"
PORTFOLIO_FILE = "portfolio.json"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
analyzer = SentimentIntensityAnalyzer()

# ---------- CORE HELPERS ----------

async def make_robust_request(session, url, params=None, retries=3, delay=5, timeout=30):
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=timeout) as response:
                response.raise_for_status()
                return await response.text()
        except Exception as e:
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

def compute_technical_indicators(series):
    if len(series.dropna()) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"] = macd_obj.macd()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0))}

def load_memory():
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, 'r') as f: return json.load(f)
    return {}

def save_memory(data):
    with open(MEMORY_FILE, 'w') as f: json.dump(data, f)
    
def load_portfolio(filename=PORTFOLIO_FILE):
    if os.path.exists(filename):
        try:
            with open(filename, 'r') as f: return json.load(f)
        except json.JSONDecodeError: return []
    return []

# ---------- RELIABLE API-FIRST DATA MODULES ----------

async def fetch_news_from_api(session, query, page_size=5):
    if not NEWSAPI_KEY: return []
    url = f"https://newsapi.org/v2/everything?q=({query})&pageSize={page_size}&language=en&sortBy=relevancy&apiKey={NEWSAPI_KEY}"
    content = await make_robust_request(session, url)
    if content:
        try:
            articles = json.loads(content).get("articles", [])
            return [{"title": a['title'], "url": a['url'], "source": a['source']['name']} for a in articles]
        except json.JSONDecodeError: return []
    return []

async def fetch_macro_sentiment(session):
    logging.info("🌍 Fetching Global Macro Sentiment via NewsAPI...")
    geo_task = fetch_news_from_api(session, "war OR conflict OR geopolitics", 20)
    trade_task = fetch_news_from_api(session, '"trade war" OR tariffs OR sanctions OR trade dispute', 15)
    econ_task = fetch_news_from_api(session, '"interest rates" OR inflation OR recession OR "gdp growth"', 20)
    geo_articles, trade_articles, econ_articles = await asyncio.gather(geo_task, trade_task, econ_task)

    geopolitical_risk = min(len(geo_articles), 20) / 20 * 100
    trade_risk = min(len(trade_articles), 15) / 15 * 100
    economic_sentiment = sum(analyzer.polarity_scores(a['title']).get('compound', 0) for a in econ_articles) / len(econ_articles) if econ_articles else 0
    overall_macro_score = -(geopolitical_risk / 100 * 15) - (trade_risk / 100 * 10) + (economic_sentiment * 15)
    
    logging.info("✅ Macro sentiment analysis complete.")
    return {
        "geopolitical_risk": geopolitical_risk, "trade_risk": trade_risk, "economic_sentiment": economic_sentiment, "overall_macro_score": overall_macro_score,
        "geo_articles": geo_articles[:3], "trade_articles": trade_articles[:3], "econ_articles": econ_articles[:3]
    }

async def fetch_context_data(session):
    # This function is stable
    pass

async def analyze_stock(semaphore, session, ticker):
    async with semaphore:
        try:
            yf_ticker = yf.Ticker(ticker)
            data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
            if data.empty: return None
            info = await asyncio.to_thread(getattr, yf_ticker, 'info')

            # RELIABLE: Use NewsAPI for all ticker-specific news.
            news_query = f'"{info.get("shortName", ticker)}"'
            general_news = await fetch_news_from_api(session, news_query)
            
            avg_sent = sum(analyzer.polarity_scores(a["title"]).get("compound", 0) for a in general_news) / len(general_news) if general_news else 0
            score = 50 + (avg_sent * 20)
            if (tech := compute_technical_indicators(data["Close"])):
                if 40 < tech.get("rsi", 50) < 70: score += 15
            if info.get('trailingPE') and 0 < info.get('trailingPE') < 40: score += 15
            
            return { "ticker": ticker, "score": score, "name": info.get('shortName', ticker), "sector": info.get('sector', 'N/A'), "summary": info.get('longBusinessSummary', None)}
        except Exception as e:
            logging.error(f"Critical error in analyze_stock for {ticker}: {e}", exc_info=False)
            return None

# ---------- EMAIL TEMPLATES ----------

def generate_html_email(df_stocks, context, market_news, macro_data, memory):
    def format_articles(articles):
        if not articles: return "<p style='color:#888;'><i>No specific news drivers detected.</i></p>"
        return "<ul style='margin:0;padding-left:20px;'>" + "".join([f'<li style="margin-bottom:5px;"><a href="{a["url"]}" style="color:#1e3a8a;">{a["title"]}</a> <span style="color:#666;">({a["source"]})</span></li>' for a in articles]) + "</ul>"
    def create_stock_table(df):
        return "".join([f'<tr><td style="padding:10px;border-bottom:1px solid #eee;"><b>{row["ticker"]}</b><br><span style="color:#666;font-size:0.9em;">{row["name"]}</span></td><td style="padding:10px;border-bottom:1px solid #eee;text-align:center;font-weight:bold;font-size:1.1em;">{row["score"]:.0f}</td></tr>' for _, row in df.iterrows()])
    
    editor_note = "Good morning. This briefing is your daily blueprint for navigating the market currents."
    if memory.get('previous_top_stock_name'): editor_note += f"<br><br><b>Yesterday's Champion:</b> {memory['previous_top_stock_name']} ({memory['previous_top_stock_ticker']}) led our rankings."
    
    sector_html = ""
    if not df_stocks.empty:
        top_by_sector = df_stocks.groupby('sector', group_keys=False).apply(lambda x: x.nlargest(2, 'score'))
        for _, row in top_by_sector.iterrows():
            if not(row['sector'] and row['sector'] != 'N/A'): continue
            summary_text = '. '.join(row["summary"].split('. ')[:2]) + '.' if row["summary"] and isinstance(row["summary"], str) else "Business summary not available."
            sector_html += f'<div style="margin-bottom:15px;"><b>{row["name"]} ({row["ticker"]})</b> in <i>{row["sector"]}</i><p style="font-size:0.9em;color:#333;margin:5px 0 0 0;">{summary_text}</p></div>'

    top10_html = create_stock_table(df_stocks.head(10)) if not df_stocks.empty else "<tr><td>No data available.</td></tr>"
    bottom10_html = create_stock_table(df_stocks.tail(10).iloc[::-1]) if not df_stocks.empty else "<tr><td>No data available.</td></tr>"
    market_news_html = "".join([f'<div style="margin-bottom:15px;"><b><a href="{a["url"]}" style="color:#000;">{a["title"]}</a></b><br><span style="color:#666;font-size:0.9em;">{a.get("source", "N/A")}</span></div>' for a in market_news]) or "<p><i>Headlines not available today.</i></p>"

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;}} .container{{width:100%;max-width:700px;margin:20px auto;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} h3{{font-size:1.2em;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Weekly Market Setter</h1><p>{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>{editor_note}</p></div>
    <div class="section"><h2>THE BIG PICTURE</h2><h3>Overall Macro Score: {macro_data.get('overall_macro_score', 0):.1f} / 30</h3><p><b>🌍 Geopolitical Risk ({macro_data.get('geopolitical_risk', 0):.0f}/100):</b><br><u>Key Drivers:</u> {format_articles(macro_data.get('geo_articles',[]))}</p><p><b>🚢 Trade Risk ({macro_data.get('trade_risk', 0):.0f}/100):</b><br><u>Key Drivers:</u> {format_articles(macro_data.get('trade_articles',[]))}</p><p><b>💼 Economic Sentiment ({macro_data.get('economic_sentiment', 0):.2f}):</b><br><u>Key Drivers:</u> {format_articles(macro_data.get('econ_articles',[]))}</p></div>
    <div class="section"><h2>SECTOR DEEP DIVE</h2><p>Top-scoring companies from different sectors.</p>{sector_html}</div>
    <div class="section"><h2>STOCK RADAR</h2><h3>📈 Top 10 Strongest Signals</h3><table style="width:100%;"><thead><tr><th>Company</th><th>Score</th></tr></thead><tbody>{top10_html}</tbody></table><h3 style="margin-top:30px;">📉 Top 10 Weakest Signals</h3><table style="width:100%;"><thead><tr><th>Company</th><th>Score</th></tr></thead><tbody>{bottom10_html}</tbody></table></div>
    <div class="section"><h2>FROM THE WIRE</h2>{market_news_html}</div>
    </div></body></html>
    """

def send_email(html_body, is_monday=False):
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    SMTP_USER, SMTP_PASS = os.getenv("SMTP_USER"), os.getenv("SMTP_PASS")
    if not SMTP_USER or not SMTP_PASS: logging.warning("SMTP creds missing."); return
    subject = "⛵ Your Weekly Market Setter" if is_monday else "🔬 Your Daily Deep Dive"
    msg = MIMEMultipart('alternative')
    msg["Subject"], msg["From"], msg["To"] = f"{subject} - {datetime.date.today()}", SMTP_USER, SMTP_USER
    msg.attach(MIMEText(html_body, 'html'))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls(); server.login(SMTP_USER, SMTP_PASS); server.send_message(msg)
        logging.info("✅ Email sent successfully.")
    except Exception as e: logging.error(f"Failed to send email: {e}")

# ---------- MAIN EXECUTION LOGIC ----------

async def run_monday_mode(output):
    logging.info("🚀 Running in MONDAY MODE...")
    previous_day_memory = load_memory()
    sp500, tsx, portfolio = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync), get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync), load_portfolio()
    universe = list(set((sp500 or [])[:75] + (tsx or [])[:25] + portfolio))
    
    semaphore = asyncio.Semaphore(10) # Control concurrency
    
    async with aiohttp.ClientSession() as session:
        stock_tasks = [analyze_stock(semaphore, session, ticker) for ticker in universe]
        # We pass the session to every function that needs to make a web request
        macro_task = fetch_macro_sentiment(session)
        news_task = fetch_news_from_api(session, "stock market OR investing OR equities", 10)
        
        results, market_news, macro_data = await asyncio.gather(
            asyncio.gather(*stock_tasks), news_task, macro_task
        )
        
    if not (stock_results := [r for r in results if r]):
        logging.error("No stock data could be analyzed. Aborting Monday run."); return

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False)
    weekly_watchlist = list(set(df_stocks.head(15)['ticker'].tolist() + df_stocks.tail(15)['ticker'].tolist() + portfolio))
    
    with open(WEEKLY_STATE_FILE, 'w') as f:
        json.dump({"start_date": datetime.date.today().isoformat(), "watchlist": weekly_watchlist, "processed_tickers": []}, f, indent=2)

    if output == "email":
        # Pass empty dicts for context as this version is simplified
        html_email = generate_html_email(df_stocks, {}, market_news, macro_data, previous_day_memory)
        send_email(html_email, is_monday=True)
    
    if not df_stocks.empty:
        save_memory({"previous_top_stock_name": df_stocks.iloc[0]['name'], "previous_top_stock_ticker": df_stocks.iloc[0]['ticker'], "previous_macro_score": macro_data.get('overall_macro_score', 0)})
    logging.info("✅ Monday Market Setter run complete.")

async def run_daily_mode(output):
    # This remains a placeholder for now, as we focus on perfecting the Monday report first
    logging.info("🏃 Running in DAILY MODE (placeholder)...")
    if not os.path.exists(WEEKLY_STATE_FILE):
        logging.warning("weekly_state.json not found. Run in Monday mode first."); return
    logging.info("Daily mode is ready to be built on this stable foundation.")

async def main(mode="daily", output="print"):
    if mode == "monday": await run_monday_mode(output)
    else: await run_daily_mode(output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Market Strategist briefing.")
    parser.add_argument("--mode", default="daily", choices=["daily", "monday"], help="Run mode.")
    parser.add_argument("--output", default="print", choices=["print", "email"], help="Output destination.")
    args = parser.parse_args()
    
    yf_cache_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yf_cache")
    os.makedirs(yf_cache_path, exist_ok=True)
    yf.set_tz_cache_location(yf_cache_path)

    if os.name == 'nt': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main(mode=args.mode, output=args.output))
