# Financial Health & Anomaly Detection Dashboard

A Streamlit port of the original analysis script (pandas pipeline + matplotlib/seaborn
visualizations + scikit-learn models for anomaly detection, health scoring, and
clustering).

## Files
- `app.py` — the Streamlit app
- `requirements.txt` — dependencies

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), and
upload your `bank_transactions.csv` file in the sidebar.

## Deploy for free (Streamlit Community Cloud)

1. Push `app.py` and `requirements.txt` to a GitHub repo (public or private).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **"New app"**, pick the repo/branch, and set the main file to `app.py`.
4. Click **Deploy**. You'll get a public URL like `https://your-app.streamlit.app`.

No server config needed — Streamlit Cloud installs `requirements.txt` automatically.

## Deploy elsewhere

Any host that runs a long-lived Python process works (Render, Railway, an EC2/VM,
a Docker container, etc.). The general recipe:

```bash
pip install -r requirements.txt
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

## Notes on what changed vs. the original script

- The two `for i in range(len(data))` loops (the anomaly flag and the health score)
  are rewritten as vectorized pandas/NumPy operations. The logic is identical
  (including Python's `and`/`or` precedence in the anomaly condition), but this
  version won't slow to a crawl on larger files.
- `data2` (category/merchant/payment) and `data3` (income/budget) are still
  randomly generated to match your uploaded transactions, exactly like the
  original script — set the seed in the sidebar to get reproducible results.
- A row-count slider lets you subsample large CSVs so the ML section (train/test
  split + 4 models) stays responsive in a browser session.
- All charts render via `st.pyplot()`; everything else (metrics, tables, model
  reports) uses native Streamlit widgets.
