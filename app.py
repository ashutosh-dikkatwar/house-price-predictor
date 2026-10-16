import os
import pickle
import warnings

import numpy as np
from flask import Flask, jsonify, request

warnings.filterwarnings("ignore")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load model (path is relative to this file so it works on Vercel too)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "linear.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Order MUST match the order used while training the model
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]

# (label, min, max) used for server-side validation
LIMITS = {
    "Square_Footage": ("Square footage", 300, 20000),
    "Num_Bedrooms": ("Bedrooms", 0, 15),
    "Num_Bathrooms": ("Bathrooms", 0, 15),
    "Year_Built": ("Year built", 1800, 2035),
    "Lot_Size": ("Lot size", 0, 100),
    "Garage_Size": ("Garage size", 0, 10),
    "Neighborhood_Quality": ("Neighborhood quality", 1, 10),
}


# Model predicts in US dollars -> convert to Indian rupees.
# Update this rate whenever you want (1 USD = ? INR).
# If your model was already trained on rupee prices, set this to 1.0
USD_TO_INR = 88.0


def indian_format(n: float) -> str:
    """Format a number with Indian digit grouping, e.g. 4,65,56,100."""
    s = str(int(round(n)))
    if len(s) <= 3:
        return s
    last3, rest = s[-3:], s[:-3]
    groups = []
    while len(rest) > 2:
        groups.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        groups.insert(0, rest)
    return ",".join(groups + [last3])


def money(x: float) -> str:
    return "\u20b9" + indian_format(x)


def money_words(x: float) -> str:
    """Easy-to-read form: '₹3.88 Crore' / '₹45.20 Lakh'."""
    if x >= 1e7:
        return f"\u20b9{x / 1e7:.2f} Crore"
    if x >= 1e5:
        return f"\u20b9{x / 1e5:.2f} Lakh"
    return money(x)


def quality_word(q: float) -> str:
    if q >= 9:
        return "excellent"
    if q >= 7:
        return "very good"
    if q >= 5:
        return "average"
    if q >= 3:
        return "below average"
    return "poor"


# ---------------------------------------------------------------------------
# Frontend (single page, light + dark theme)
# ---------------------------------------------------------------------------
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>House Price Predictor</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #f4f6fb;
    --bg-accent: radial-gradient(1200px 500px at 10% -10%, #dfe7ff 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #e9e0ff 0%, transparent 55%);
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --border: #e2e8f0;
    --input-bg: #f8fafc;
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --primary-soft: #eef2ff;
    --success: #059669;
    --danger: #dc2626;
    --danger-soft: #fef2f2;
    --shadow: 0 10px 30px rgba(15, 23, 42, .08);
  }
  :root[data-theme="dark"] {
    --bg: #0b1020;
    --bg-accent: radial-gradient(1200px 500px at 10% -10%, #1b2250 0%, transparent 60%),
                 radial-gradient(900px 500px at 100% 0%, #2a1b4d 0%, transparent 55%);
    --card: #121935;
    --text: #e8ecf8;
    --muted: #94a3c4;
    --border: #232c52;
    --input-bg: #0e1530;
    --primary: #7c83ff;
    --primary-hover: #9298ff;
    --primary-soft: #1a2150;
    --success: #34d399;
    --danger: #f87171;
    --danger-soft: #2a1620;
    --shadow: 0 10px 30px rgba(0, 0, 0, .45);
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; }
  body {
    font-family: "Inter", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); background-image: var(--bg-accent); background-attachment: fixed;
    color: var(--text); min-height: 100vh; line-height: 1.5;
    transition: background .25s, color .25s;
  }
  .wrap { max-width: 980px; margin: 0 auto; padding: 28px 18px 60px; }
  header { display: flex; align-items: center; justify-content: space-between; gap: 12px; margin-bottom: 26px; }
  .brand { display: flex; align-items: center; gap: 12px; }
  .logo {
    width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
    background: linear-gradient(135deg, var(--primary), #a855f7); color: #fff;
  }
  .brand h1 { font-size: 1.15rem; margin: 0; letter-spacing: -.01em; }
  .brand p { margin: 0; font-size: .8rem; color: var(--muted); }
  .theme-btn {
    border: 1px solid var(--border); background: var(--card); color: var(--text);
    width: 42px; height: 42px; border-radius: 12px; cursor: pointer; display: grid; place-items: center;
    box-shadow: var(--shadow); transition: transform .15s;
  }
  .theme-btn:hover { transform: translateY(-1px); }
  .grid { display: grid; grid-template-columns: 1.25fr 1fr; gap: 20px; align-items: start; }
  @media (max-width: 820px) { .grid { grid-template-columns: 1fr; } }
  .card {
    background: var(--card); border: 1px solid var(--border); border-radius: 18px;
    padding: 24px; box-shadow: var(--shadow); transition: background .25s, border-color .25s;
  }
  .card h2 { margin: 0 0 4px; font-size: 1.05rem; }
  .card .sub { margin: 0 0 20px; color: var(--muted); font-size: .88rem; }
  .fields { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 520px) { .fields { grid-template-columns: 1fr; } }
  .field.full { grid-column: 1 / -1; }
  label { display: block; font-size: .82rem; font-weight: 600; margin-bottom: 6px; }
  label small { color: var(--muted); font-weight: 500; }
  input[type=number] {
    width: 100%; padding: 11px 13px; font: inherit; color: var(--text);
    background: var(--input-bg); border: 1px solid var(--border); border-radius: 10px; outline: none;
    transition: border-color .15s, box-shadow .15s;
  }
  input[type=number]:focus { border-color: var(--primary); box-shadow: 0 0 0 3px var(--primary-soft); }
  input[type=range] { width: 100%; accent-color: var(--primary); }
  .range-row { display: flex; align-items: center; justify-content: space-between; }
  .badge { background: var(--primary-soft); color: var(--primary); font-weight: 700; font-size: .8rem; padding: 2px 10px; border-radius: 999px; }
  button.primary {
    margin-top: 22px; width: 100%; padding: 13px 16px; font: inherit; font-weight: 600; color: #fff;
    background: linear-gradient(135deg, var(--primary), #8b5cf6); border: 0; border-radius: 12px; cursor: pointer;
    transition: transform .15s, opacity .15s; display: flex; align-items: center; justify-content: center; gap: 8px;
  }
  button.primary:hover { transform: translateY(-1px); }
  button.primary:disabled { opacity: .65; cursor: wait; transform: none; }
  .spinner { width: 16px; height: 16px; border: 2px solid rgba(255,255,255,.4); border-top-color: #fff; border-radius: 50%; animation: spin .7s linear infinite; display: none; }
  button.loading .spinner { display: inline-block; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .placeholder { text-align: center; padding: 34px 10px; color: var(--muted); }
  .placeholder svg { opacity: .55; margin-bottom: 10px; }
  .result { display: none; animation: fade .35s ease; }
  @keyframes fade { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: none; } }
  .eyebrow { font-size: .75rem; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); font-weight: 600; }
  .price { font-size: 2.5rem; font-weight: 700; letter-spacing: -.03em; color: var(--success); margin: 4px 0 2px; }
  .range { color: var(--muted); font-size: .9rem; }
  .summary { margin: 18px 0; padding: 14px 16px; border-radius: 12px; background: var(--primary-soft); font-size: .93rem; }
  .stats { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .stat { border: 1px solid var(--border); border-radius: 12px; padding: 10px 12px; }
  .stat b { display: block; font-size: 1rem; }
  .stat span { font-size: .75rem; color: var(--muted); }
  .error { display: none; margin-top: 16px; padding: 12px 14px; border-radius: 10px; background: var(--danger-soft); color: var(--danger); font-size: .88rem; }
  .note { margin-top: 16px; font-size: .75rem; color: var(--muted); }
  footer { text-align: center; color: var(--muted); font-size: .78rem; margin-top: 30px; }
</style>
<script>
  // Apply saved theme (or system preference) before paint to avoid flashing
  (function () {
    var saved = null;
    try { saved = localStorage.getItem("theme"); } catch (e) {}
    var dark = saved ? saved === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
  })();
</script>
</head>
<body>
<div class="wrap">
  <header>
    <div class="brand">
      <div class="logo">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
      </div>
      <div>
        <h1>House Price Predictor</h1>
        <p>Instant estimate powered by machine learning</p>
      </div>
    </div>
    <button class="theme-btn" id="themeBtn" aria-label="Toggle theme" title="Toggle theme"></button>
  </header>

  <div class="grid">
    <form class="card" id="form" novalidate>
      <h2>Property details</h2>
      <p class="sub">Fill in the details of the house to get an estimated market value.</p>
      <div class="fields">
        <div class="field">
          <label for="Square_Footage">Square footage <small>(sq ft)</small></label>
          <input type="number" id="Square_Footage" name="Square_Footage" min="300" max="20000" step="1" value="2000" required>
        </div>
        <div class="field">
          <label for="Year_Built">Year built</label>
          <input type="number" id="Year_Built" name="Year_Built" min="1800" max="2035" step="1" value="2000" required>
        </div>
        <div class="field">
          <label for="Num_Bedrooms">Bedrooms</label>
          <input type="number" id="Num_Bedrooms" name="Num_Bedrooms" min="0" max="15" step="1" value="3" required>
        </div>
        <div class="field">
          <label for="Num_Bathrooms">Bathrooms</label>
          <input type="number" id="Num_Bathrooms" name="Num_Bathrooms" min="0" max="15" step="1" value="2" required>
        </div>
        <div class="field">
          <label for="Lot_Size">Lot size <small>(acres)</small></label>
          <input type="number" id="Lot_Size" name="Lot_Size" min="0" max="100" step="0.01" value="2.5" required>
        </div>
        <div class="field">
          <label for="Garage_Size">Garage size <small>(cars)</small></label>
          <input type="number" id="Garage_Size" name="Garage_Size" min="0" max="10" step="1" value="1" required>
        </div>
        <div class="field full">
          <div class="range-row">
            <label for="Neighborhood_Quality">Neighborhood quality <small>(1 = poor, 10 = excellent)</small></label>
            <span class="badge" id="qBadge">5</span>
          </div>
          <input type="range" id="Neighborhood_Quality" name="Neighborhood_Quality" min="1" max="10" step="1" value="5">
        </div>
      </div>
      <button class="primary" id="submitBtn" type="submit">
        <span class="spinner"></span><span id="btnText">Predict price</span>
      </button>
      <div class="error" id="error"></div>
    </form>

    <div class="card">
      <h2>Estimated value</h2>
      <p class="sub">Your prediction will appear here.</p>

      <div class="placeholder" id="placeholder">
        <svg width="54" height="54" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
        <div>Enter the property details and press <b>Predict price</b>.</div>
      </div>

      <div class="result" id="result">
        <div class="eyebrow">Estimated market value</div>
        <div class="price" id="price"></div>
        <div class="range" id="range"></div>
        <div class="summary" id="summary"></div>
        <div class="stats">
          <div class="stat"><b id="perSqft"></b><span>Price per sq ft</span></div>
          <div class="stat"><b id="age"></b><span>Age of the house</span></div>
        </div>
        <div class="note">This is an automated estimate based on past data. Actual prices can vary with market conditions, condition of the property and negotiation.</div>
      </div>
    </div>
  </div>

  <footer>Built with Flask &amp; scikit-learn</footer>
</div>

<script>
  const $ = (id) => document.getElementById(id);
  const sun = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg>';
  const moon = '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';

  function paintThemeIcon() {
    const dark = document.documentElement.getAttribute("data-theme") === "dark";
    $("themeBtn").innerHTML = dark ? sun : moon;
  }
  paintThemeIcon();
  $("themeBtn").addEventListener("click", () => {
    const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try { localStorage.setItem("theme", next); } catch (e) {}
    paintThemeIcon();
  });

  $("Neighborhood_Quality").addEventListener("input", (e) => { $("qBadge").textContent = e.target.value; });

  $("form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const err = $("error"); err.style.display = "none";
    const btn = $("submitBtn");
    btn.disabled = true; btn.classList.add("loading"); $("btnText").textContent = "Predicting...";

    const payload = {};
    new FormData($("form")).forEach((v, k) => payload[k] = v);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok || !data.ok) throw new Error(data.error || "Something went wrong.");

      $("price").textContent = data.price;
      $("range").textContent = "Likely range: " + data.low + " – " + data.high;
      $("summary").textContent = data.summary;
      $("perSqft").textContent = data.per_sqft;
      $("age").textContent = data.age;
      $("placeholder").style.display = "none";
      const r = $("result"); r.style.display = "block";
      r.style.animation = "none"; r.offsetHeight; r.style.animation = "";
      if (window.innerWidth <= 820) r.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (ex) {
      err.textContent = ex.message; err.style.display = "block";
    } finally {
      btn.disabled = false; btn.classList.remove("loading"); $("btnText").textContent = "Predict price";
    }
  });
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home():
    return HTML


@app.route("/health")
def health():
    return jsonify(status="ok")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or request.form.to_dict()

    values = []
    for key in FEATURES:
        label, lo, hi = LIMITS[key]
        raw = data.get(key)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            return jsonify(ok=False, error=f"Please enter a valid number for '{label}'."), 400
        if not (lo <= val <= hi):
            return jsonify(ok=False, error=f"'{label}' must be between {lo} and {hi}."), 400
        values.append(val)

    x = np.array([values], dtype=float)
    price = float(model.predict(x)[0]) * USD_TO_INR  # USD -> INR

    if price <= 0:
        return jsonify(
            ok=False,
            error="These inputs are outside the range the model was trained on, so it "
                  "could not produce a realistic price. Please try more typical values.",
        ), 422

    sqft, beds, baths, year, lot, garage, quality = values
    low, high = price * 0.90, price * 1.10  # rough +/-10% band
    age = max(0, 2026 - int(year))

    summary = (
        f"Based on the details you entered, a {sqft:,.0f} sq ft home with "
        f"{beds:g} bedroom{'s' if beds != 1 else ''} and {baths:g} bathroom{'s' if baths != 1 else ''}, "
        f"built in {int(year)}, on a {lot:g}-acre lot with space for {garage:g} car{'s' if garage != 1 else ''} "
        f"in a neighborhood rated {quality_word(quality)} ({quality:g}/10) "
        f"is estimated to be worth around {money_words(price)} ({money(price)}). "
        f"Most similar homes would fall between {money_words(low)} and {money_words(high)}."
    )

    return jsonify(
        ok=True,
        price=money(price),
        low=money(low),
        high=money(high),
        per_sqft=money(price / sqft),
        age="New build" if age == 0 else f"{age} year{'s' if age != 1 else ''}",
        summary=summary,
    )


if __name__ == "__main__":
    app.run(debug=True)
