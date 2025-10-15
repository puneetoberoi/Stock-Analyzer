import os, sys, argparse, time, datetime, logging, json, asyncio
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import aiohttp
from bs4 import BeautifulSoup
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
    if time_since_last_call < 1.5: await asyncio.sleep(1.5 - time_since_last_call)
    LAST_API_CALL_TIME = time.time()

async def make_robust_request(session, url, params=None, retries=3, delay=5, timeout=30):
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
        logging.info(f"Loading tickers from cache: {cache_file}")
        with open(cache_file, 'r') as f: return json.load(f)
    tickers = fetch_function()
    if tickers: # Only cache if we actually got tickers
        with open(cache_file, 'w') as f: json.dump(tickers, f)
    return tickers

# FIX: Made functions guaranteed to return a list
def fetch_sp500_tickers_sync():
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
        df = pd.read_html(StringIO(response.text))[0] # FIX: Use StringIO to address FutureWarning
        return df["Symbol"].tolist()
    except Exception as e1:
        logging.warning(f"S&P 500 Wikipedia fetch failed: {e1}. Trying fallback.")
        try:
            df = pd.read_csv("https://datahub.io/core/s-and-p-500-companies/r/constituents.csv")
            return df["Symbol"].tolist()
        except Exception as e2:
            logging.error(f"S&P 500 fallback also failed: {e2}. Returning empty list.")
            return [] # Guaranteed to return a list

def fetch_tsx_tickers_sync():
    try:
        url = "https://en.wikipedia.org/wiki/S%26P/TSX_Composite_Index"
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=10)
        response.raise_for_status()
        for table in pd.read_html(StringIO(response.text)): # FIX: Use StringIO
            if 'Symbol' in table.columns: return [str(t).split(' ')[0].replace('.', '-') + ".TO" for t in table["Symbol"].tolist()]
    except Exception as e:
        logging.warning(f"TSX fetch failed: {e}. Using small static fallback.")
        # Return a minimal list, but still a list
        return ["RY.TO", "TD.TO", "ENB.TO", "SHOP.TO"]
    logging.error("Could not find TSX tickers. Returning empty list.")
    return [] # Guaranteed to return a list

async def fetch_finviz_news_async(session, ticker):
    url = f"https://finviz.com/quote.ashx?t={ticker}"
    content = await make_robust_request(session, url)
    if not content: return []
    soup = BeautifulSoup(content, 'html.parser')
    news_table = soup.find(id='news-table')
    if not news_table: return []
    return [{"title": row.a.text, "url": row.a['href']} for row in news_table.findAll('tr') if row.a]

async def fetch_news_analysis(session, query, source_api="gdelt"):
    articles, found_articles = [], []
    keywords = query.lower().replace('"', '').split(" or ")
    if source_api == "gdelt":
        gdelt_query = f"({query})"
        content = await make_robust_request(session, "https://api.gdeltproject.org/api/v2/doc/doc", params={"query": gdelt_query, "mode": "artlist", "maxrecords": 50, "format": "json", "timespan": "7d"})
        if content: articles = json.loads(content).get("articles", [])
    elif source_api == "newsapi" and NEWSAPI_KEY:
        await api_cooldown()
        news_query = f"({query})"
        content = await make_robust_request(session, f"https://newsapi.org/v2/everything?q={requests.utils.quote(news_query)}&pageSize=30&apiKey={NEWSAPI_KEY}")
        if content: articles = json.loads(content).get("articles", [])

    for article in articles:
        title = article.get("title", "").lower()
        if any(keyword in title for keyword in keywords):
            source_name = article.get("sourcecountry", "N/A") if source_api == "gdelt" else article.get("source", {}).get("name", "N/A")
            found_articles.append({"title": article.get("title"), "url": article.get("url"), "source": source_name})
    return found_articles

async def fetch_macro_sentiment():
    async with aiohttp.ClientSession() as session:
        logging.info("🌍 Fetching Global Macro Sentiment...")
        geo_query, trade_query = "war OR conflict OR military OR attack OR invasion", '"trade war" OR tariff OR sanctions'
        
        geo_articles = await fetch_news_analysis(session, geo_query, "gdelt")
        if not geo_articles: geo_articles = await fetch_news_analysis(session, geo_query, "newsapi")
        
        trade_articles = await fetch_news_analysis(session, trade_query, "gdelt")
        if not trade_articles: trade_articles = await fetch_news_analysis(session, trade_query, "newsapi")

        econ_articles = await fetch_news_analysis(session, '"interest rates" OR inflation OR recession OR "gdp growth"', "newsapi")

        geopolitical_risk, trade_risk = min((len(geo_articles) / 20.0) * 100, 100), min((len(trade_articles) / 15.0) * 100, 100)
        economic_sentiment = sum([analyzer.polarity_scores(a['title']).get('compound', 0) for a in econ_articles]) / len(econ_articles) if econ_articles else 0
        overall_macro_score = -(geopolitical_risk / 100 * 15) - (trade_risk / 100 * 10) + (economic_sentiment * 15)
        
        logging.info("✅ Macro sentiment analysis complete.")
        return {
            "geopolitical_risk": geopolitical_risk, "geo_articles": geo_articles[:5],
            "trade_risk": trade_risk, "trade_articles": trade_articles[:5],
            "economic_sentiment": economic_sentiment, "econ_articles": econ_articles[:5],
            "overall_macro_score": overall_macro_score }

def compute_technical_indicators(series):
    series = series.dropna()
    if len(series) < 50: return None
    df = pd.DataFrame({"close": series})
    df["rsi_14"] = RSIIndicator(df["close"], window=14).rsi()
    macd_obj = MACD(df["close"], window_slow=26, window_fast=12, window_sign=9)
    df["macd"], df["macd_signal"] = macd_obj.macd(), macd_obj.macd_signal()
    latest = df.iloc[-1].fillna(0)
    return {"rsi": float(latest.get("rsi_14", 50)), "macd": float(latest.get("macd", 0)), "macd_signal": float(latest.get("macd_signal", 0))}

async def analyze_asset(ticker, session, asset_type='stock'):
    try:
        yf_ticker = yf.Ticker(ticker)
        data = await asyncio.to_thread(yf_ticker.history, period="1y", interval="1d")
        if data.empty: return None
        info = await asyncio.to_thread(getattr, yf_ticker, 'info')

        tech = compute_technical_indicators(data["Close"])
        articles = await fetch_finviz_news_async(session, ticker)
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
    universe = (sp500 or [])[:75] + (tsx or [])[:25] # FIX: Handle NoneType gracefully
    
    crypto_tickers, commodity_tickers = ["BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD"], ["GC=F", "SI=F"]
    logging.info(f"Analyzing {len(universe)} stocks, {len(crypto_tickers)} cryptos, and {len(commodity_tickers)}...")

    async with aiohttp.ClientSession() as session:
        all_tickers = universe + crypto_tickers + commodity_tickers
        tasks = [analyze_asset(ticker, session, 'stock' if ticker in universe else 'crypto' if ticker in crypto_tickers else 'commodity') for ticker in all_tickers]
        results = await asyncio.gather(*tasks)

    stock_results, crypto_results, commodity_results = [], [], []
    if results:
        stock_results = [r for r in results[:len(universe)] if r]
        crypto_results = [r for r in results[len(universe):len(universe)+len(crypto_tickers)] if r]
        commodity_results = [r for r in results[len(universe)+len(crypto_tickers):] if r]

    df_stocks = pd.DataFrame(stock_results).sort_values("score", ascending=False) if stock_results else pd.DataFrame()
    df_crypto = pd.DataFrame(crypto_results).sort_values("score", ascending=False) if crypto_results else pd.DataFrame()
    df_commodities = pd.DataFrame(commodity_results).sort_values("score", ascending=False) if commodity_results else pd.DataFrame()
    
    # FIX: Reuse existing econ_articles for market news, eliminating the final API call
    market_news = macro_data.get("econ_articles", [])

    if output == "email":
        html_email = generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news)
        send_email(html_email)
    else: print(df_stocks.head(10))

    logging.info("✅ Done.")

def generate_html_email(df_stocks, df_crypto, df_commodities, macro_data, market_news):
    def format_articles_html(articles):
        if not articles: return "<p style='color:#888;'><i>No major news drivers detected.</i></p>"
        return "<ul style='margin:0;padding-left:20px;'>" + "".join([f'<li style="margin-bottom: 5px;"><a href="{a["url"]}" style="color: #1e3a8a; text-decoration: none;">{a["title"]}</a> <span style="color: #666;">({a["source"]})</span></li>' for a in articles]) + "</ul>"
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
    market_news_html = "".join([f'<div style="margin-bottom: 15px;"><b><a href="{a["url"]}" style="color: #000; text-decoration: none; font-size: 1.1em;">{a["title"]}</a></b><br><span style="color: #666; font-size: 0.9em;">Source: {a.get("source", "N/A")}</span></div>' for a in market_news[:4]])

    return f"""
    <!DOCTYPE html><html><head><style>body{{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;padding:0;background-color:#f7f7f7;}} .container{{width:100%;max-width:700px;margin:20px auto;background-color:#fff;border:1px solid #ddd;}} .header{{background-color:#0c0a09;color:#fff;padding:30px;text-align:center;}} .section{{padding:25px;border-bottom:1px solid #ddd;}} .section h2{{font-size:1.5em;color:#111;margin-top:0;}} .section h3{{font-size:1.2em;color:#333;border-bottom:2px solid #e2e8f0;padding-bottom:5px;}}</style></head><body><div class="container">
    <div class="header"><h1>Your Daily Intelligence Briefing</h1><p style="font-size:1.1em; color:#aaa;">{datetime.date.today().strftime('%A, %B %d, %Y')}</p></div>
    <div class="section"><h2>EDITOR’S NOTE</h2><p>Good morning. Think of the market as a big, complicated ocean. Some days it's calm, some days it's stormy. Our job isn't to predict the waves, but to build a better boat. This briefing is your daily blueprint. We'll check the weather (the macro environment), map the currents (sector performance), and point out the interesting ships on the horizon (top stocks). Let's set sail.</p></div>
    <div class="section"><h2>THE BIG PICTURE: The Market Weather Report</h2>
        <h3>Overall Macro Score: {macro_data['overall_macro_score']:.1f} / 30</h3>
        <p>This is our "weather forecast" for investors. A high positive score (+10 to +30) is like a sunny day—investors feel optimistic. A deep negative score (-10 to -30) is a storm warning, suggesting caution and a flight to safety.</p>
        <p><b>🌍 Geopolitical Risk ({macro_data['geopolitical_risk']:.0f}/100):</b> This is like checking for international storms. We scan the globe for conflict news. High scores mean choppy waters ahead, which usually makes safe havens like Gold more attractive. <br><u>Key Drivers Today:</u> {format_articles_html(macro_data['geo_articles'])}</p>
        <p><b>🚢 Trade Risk ({macro_data['trade_risk']:.0f}/100):</b> Are the world's commercial shipping lanes open or closed? We look for talk of tariffs and trade wars. High scores can mean delays and higher costs for big international companies. <br><u>Key Drivers Today:</u> {format_articles_html(macro_data['trade_articles'])}</p>
        <p><b>💼 Economic Sentiment ({macro_data['economic_sentiment']:.2f}):</b> What's the mood in the boardroom? We analyze the tone of financial news about jobs, inflation, and growth. A positive number means optimism is in the air; negative means pessimism is creeping in. <br><u>Key Drivers Today:</u> {format_articles_html(macro_data['econ_articles'])}</p>
    </div>
    <div class="section"><h2>SECTOR DEEP DIVE: Who's Building the Future?</h2><p>Every industry tells a story. Here, we highlight the top-scoring companies from different sectors to give you a cross-section of the market's strongest narratives right now.</p>{sector_html}</div>
    <div class="section"><h2>STOCK RADAR: Today's Most Interesting Signals</h2>
        <h3>📈 Top 10 Strongest Signals</h3><p>These stocks are currently firing on all cylinders, showing a strong combination of market value, positive momentum, and good press. They're the ships catching the strongest wind.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{top10_html}</tbody></table>
        <h3 style="margin-top: 30px;">📉 Top 10 Weakest Signals</h3><p>These stocks are currently facing headwinds, with weaker scores in our analysis. They might be undervalued opportunities or signals of underlying issues worth investigating.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Company</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{bottom10_html}</tbody></table>
    </div>
    <div class="section"><h2>BEYOND STOCKS: Alternative Assets</h2>
        <h3>🪙 Crypto: The Digital Frontier</h3><p>Cryptocurrencies are like the new, uncharted islands of the financial world. They're volatile, high-risk, and high-reward. Bitcoin is the largest, often seen as "digital gold," while others like Ethereum power new applications. We score them based on momentum and news sentiment.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{crypto_html}</tbody></table>
        <h3 style="margin-top: 30px;">💎 Commodities: The Bedrock Assets</h3><p>These are the real, physical materials that build our world. Gold (GC=F) is the ultimate "safe harbor" investors flock to during storms. Silver (SI=F) is both an industrial metal and a store of value. Their scores are heavily influenced by global uncertainty.</p><table style="width:100%; border-collapse: collapse;"><thead><tr><th style="text-align:left; padding:10px;">Asset</th><th style="text-align:center; padding:10px;">Score</th></tr></thead><tbody>{commodities_html}</tbody></table>
    </div>
    <div class="section"><h2>FROM THE WIRE: Today's Market Narratives</h2>{market_news_html}</div>
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
    parser.add_argument("--output", default="print", choices=["print", "email", "slack"], help="Output destination.")
    args = parser.parse_args()
    asyncio.run(main(output=args.output))
