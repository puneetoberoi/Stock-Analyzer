import os, sys, argparse, time, datetime, logging, json, asyncio
import requests, math
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
import aioyfinance as ayf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from io import StringIO

# ---------- config ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"}
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
# OPENAI_KEY = os.getenv("OPENAI_KEY") # NEW: Add this secret if you use the AI summary feature
analyzer = SentimentIntensityAnalyzer()
LAST_API_CALL_TIME = 0

# ---------- helpers ----------

# MODIFIED: Intelligent cooldown to respect API rate limits
async def api_cooldown():
    global LAST_API_CALL_TIME
    time_since_last_call = time.time() - LAST_API_CALL_TIME
    # Free NewsAPI plan allows ~1 call every ~1.5s. We'll be safer.
    if time_since_last_call < 2.0:
        await asyncio.sleep(2.0 - time_since_last_call)
    LAST_API_CALL_TIME = time.time()

async def make_robust_request(session, url, params=None, retries=3, delay=5):
    """MODIFIED: Async robust requests function using aiohttp session."""
    for attempt in range(retries):
        try:
            async with session.get(url, params=params, headers=REQUEST_HEADERS, timeout=15) as response:
                response.raise_for_status()
                content = await response.text()
                if content:
                    return content
                logging.warning(f"Request to {url} was successful but returned empty content.")
                return None
        except aiohttp.ClientError as e:
            logging.warning(f"HTTP error on attempt {attempt + 1}/{retries} for {url}: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(delay)
            else:
                logging.error(f"All {retries} attempts failed for URL: {url}")
    return None

# NEW: Caching mechanism for ticker lists
def get_cached_tickers(cache_file, fetch_function, *args):
    if os.path.exists(cache_file) and (time.time() - os.path.getmtime(cache_file)) < 86400: # 24 hours
        logging.info(f"Loading tickers from cache: {cache_file}")
        with open(cache_file, 'r') as f:
            return json.load(f)
    
    logging.info(f"Cache not found or expired. Fetching fresh tickers for {cache_file}.")
    tickers = fetch_function(*args)
    with open(cache_file, 'w') as f:
        json.dump(tickers, f)
    return tickers

def fetch_sp500_tickers_sync():
    # This remains sync because it's a one-off and simpler this way
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        df = tables[0]
        tickers = df["Symbol"].tolist()
        logging.info(f"✅ Wikipedia S&P 500 fetch successful — {len(tickers)} tickers found.")
        return tickers
    except Exception as e:
        logging.warning(f"⚠️ S&P 500 Wikipedia fetch failed: {e}. Trying fallback.")
        df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
        return df["Symbol"].tolist()

def fetch_tsx_tickers_sync():
    # Sync one-off fetch
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
        tables = pd.read_html(StringIO(response.text))
        for table in tables:
            if 'Symbol' in table.columns and 'Company' in table.columns:
                tickers = [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
                logging.info(f"✅ TSX fetch successful — {len(tickers)} tickers found.")
                return tickers
        raise ValueError("Constituents table not found.")
    except Exception as e:
        logging.warning(f"⚠️ TSX fetch failed: {e}. Using static fallback.")
        return ["RY.TO", "TD.TO", "ENB.TO", "BNS.TO", "CNQ.TO", "TRP.TO", "SHOP.TO"]

# MODIFIED: Returns articles along with the score
async def analyze_geopolitical_risk(session):
    logging.info("Analyzing geopolitical risks...")
    risk_keywords = ["war", "conflict", "military", "attack", "strike", "invasion", "bombing", "missile"]
    gdelt_query = "(war OR conflict OR military OR attack OR strike OR invasion OR bombing OR missile)"
    
    # GDELT doesn't need API cooldown
    content = await make_robust_request(session, "https://api.gdeltproject.org/api/v2/doc/doc", params={"query": gdelt_query, "mode": "artlist", "maxrecords": 50, "format": "json", "timespan": "7d"})
    
    if not content:
        return 0, []

    articles = json.loads(content).get("articles", [])
    
    risk_articles = []
    for article in articles:
        title = article.get("title", "").lower()
        if any(keyword in title for keyword in risk_keywords):
            risk_articles.append({"title": article.get("title"), "url": article.get("url"), "source": article.get("sourcecountry")})

    score = min((len(risk_articles) / 20.0) * 100, 100)
    logging.info(f"  Geopolitical risk score: {score:.1f}/100 ({len(risk_articles)} conflict-related articles found)")
    return score, risk_articles[:5] # Return score and top 5 articles

# NEW: AI Summarizer (placeholder)
async def summarize_with_ai(articles, topic):
    # This is where you would integrate with an LLM like GPT
    # For now, we will just format the article titles.
    # OPENAI_KEY = os.getenv("OPENAI_KEY")
    # if not OPENAI_KEY:
    #     logging.warning("OPENAI_KEY not set. Skipping AI summary.")
    #     return "AI summary disabled. Please set the OPENAI_KEY environment variable."
    
    summary_points = []
    for article in articles:
        summary_points.append(f"- [{article['title']}]({article['url']}) (Source: {article['source']})")
    
    return "\n".join(summary_points)

# MODIFIED: Centralized async macro analysis
async def fetch_macro_sentiment():
    async with aiohttp.ClientSession() as session:
        logging.info("\n🌍 Fetching Global Macro Sentiment...")
        
        # Run analyses concurrently
        geo_risk_task = analyze_geopolitical_risk(session)
        # In a real scenario, you'd make trade/economic analyses async too
        # For now, let's keep them simple to show the structure
        
        geopolitical_risk, risk_articles = await geo_risk_task
        
        # Simplified sync calls for others for now
        trade_risk = 0 # Placeholder
        economic_sentiment = 0 # Placeholder
        
        # Generate AI summary for the report
        risk_summary = await summarize_with_ai(risk_articles, "geopolitical risk")

        overall_macro_score = -(geopolitical_risk / 100 * 15) - (trade_risk / 100 * 10) + (economic_sentiment * 15)
        
        macro_data = {
            "geopolitical_risk": geopolitical_risk,
            "trade_risk": trade_risk,
            "economic_sentiment": economic_sentiment,
            "overall_macro_score": overall_macro_score,
            "risk_summary": risk_summary # NEW: Add summary to data
        }
        
        logging.info(f"\n📊 Macro Summary:")
        logging.info(f"  Overall Macro Score: {overall_macro_score:.2f}/30")
        return macro_data

# ... (Keep compute_technical_indicators, compute_momentum_volatility, scoring functions as they are) ...
# ... They are CPU-bound and don't need to be async. ...
def compute_technical_indicators(series):
    series = series.dropna()
    if len(series) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0)), "macd_signal": float(latest.get("macd_signal", 0))}

# MODIFIED: Asynchronous analysis of a single stock
async def analyze_stock(ticker, macro_data, session):
    try:
        # Using aioyfinance for async data fetching
        data = await ayf.Ticker(ticker).history(period="1y", interval="1d")
        if data.empty:
            return None
        
        info = await ayf.Ticker(ticker).info # Less reliable, use with care
        
        tech = compute_technical_indicators(data["Close"])
        fund = {"pe": info.get("trailingPE"), "de_ratio": info.get("debtToEquity")}
        
        await api_cooldown()
        news_query = f'"{info.get("longName", ticker)}"'
        news_content = await make_robust_request(session, f"https://newsapi.org/v2/everything?q={requests.utils.quote(news_query)}&pageSize=5&apiKey={NEWSAPI_KEY}")
        
        articles = json.loads(news_content).get("articles", []) if news_content else []
        sentiments = [analyzer.polarity_scores(a.get("title", "")).get("compound", 0) for a in articles]
        avg_sent = sum(sentiments) / len(sentiments) if sentiments else 0
        
        # Scoring logic remains the same (CPU-bound)
        score = 0
        if fund.get("pe") and 0 < fund["pe"] < 40: score += 20
        if tech and 40 < tech.get("rsi", 50) < 70: score += 10
        score += max(min((avg_sent * 10), 10), -10)
        # Simplified scoring for this example
        
        return {"ticker": ticker, "score": score, "sentiment": avg_sent}
    except Exception as e:
        logging.error(f"Error processing stock {ticker}: {e}", exc_info=False)
        return None

# MODIFIED: Main function is now async
async def main(output="print"):
    macro_data = await fetch_macro_sentiment()
    
    sp500 = get_cached_tickers('sp500_cache.json', fetch_sp500_tickers_sync)
    tsx = get_cached_tickers('tsx_cache.json', fetch_tsx_tickers_sync)
    universe = sp500[:100] + tsx[:50] # Reduced for speed in example
    logging.info(f"Running for {len(universe)} stock tickers...")

    async with aiohttp.ClientSession() as session:
        tasks = [analyze_stock(ticker, macro_data, session) for ticker in universe]
        results = await asyncio.gather(*tasks)
    
    stock_results = [r for r in results if r is not None]
    
    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False)
    
    # --- Generate Report ---
    # Placeholder for crypto/commodity results which would follow the same async pattern
    df_crypto = pd.DataFrame() 

    report_md = generate_markdown_report(df_stocks, df_crypto, macro_data)
    
    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, macro_data)
        send_email(html_email) # send_email remains synchronous
    else:
        print(report_md)

    with open("daily_report.md", "w", encoding='utf-8') as f:
        f.write(report_md)
    logging.info("✅ Done.")

# MODIFIED: Report generation includes the new AI summary
def generate_markdown_report(df_stocks, df_crypto, macro_data):
    md = [
        f"# Daily Market Report — {datetime.datetime.utcnow().date()}\n",
        "## 🌍 Global Macro Environment",
        f"**Overall Macro Score:** {macro_data['overall_macro_score']:.2f}/30",
        f"- Geopolitical Risk: {macro_data['geopolitical_risk']:.1f}/100\n",
        "### Key Geopolitical Drivers:",
        macro_data.get('risk_summary', "No specific risk articles found."),
        "\n## 📈 Top 30 Stocks (by composite score)",
        df_stocks.head(30)[["ticker", "score"]].to_markdown(index=False) if not df_stocks.empty else "No stock data available.",
    ]
    # ... more sections ...
    return "\n\n".join(md)
    
def generate_html_email(df_stocks, df_crypto, macro_data):
    # This function would also be updated to include a nice HTML block for the summary
    risk_summary_html = macro_data.get('risk_summary', "No specific risk articles found.").replace('\n', '<br>')
    # Example of adding it to the email
    # ... inside your html string ...
    # <h3>Key Geopolitical Drivers</h3><p>{risk_summary_html}</p>
    # ...
    return "<html><body>HTML Email Content</body></html>" # Placeholder

def send_email(html_body):
    # This can remain a synchronous function
    logging.info("Sending email...")
    # Your smtplib logic here

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the daily market analysis.")
    parser.add_argument("--output", default="print", choices=["print", "email", "slack"], help="Output destination for the report.")
    args = parser.parse_args()
    
    # MODIFIED: Run the async main function
    asyncio.run(main(output=args.output))
