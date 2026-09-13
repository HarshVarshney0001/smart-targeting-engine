import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import io

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Smart Targeting Engine", page_icon="🎯", layout="wide")

# ---------- PATHS ----------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

T_LEARNER_PATH = MODEL_DIR / "t_learner.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_control_model.pkl"
SEGMENTATION_CSV_PATH = MODEL_DIR / "segmentation_results.csv"
QINI_SCORES_PATH = MODEL_DIR / "qini_scores.pkl"
QINI_PLOT_PATH = MODEL_DIR / "qini_curve.png"
HILLSTROM_DATA_PATH = DATA_DIR / "hillstrom.csv"

# ---------- FEATURE ORDER (must exactly match training in uplift_models.py) ----------
FEATURE_COLS = [
    "recency", "history", "mens", "womens", "newbie",
    "history_segment_2) $100 - $200", "history_segment_3) $200 - $350",
    "history_segment_4) $350 - $500", "history_segment_5) $500 - $750",
    "history_segment_6) $750 - $1,000", "history_segment_7) $1,000 +",
    "zip_code_Surburban", "zip_code_Urban",
    "channel_Phone", "channel_Web",
]

HISTORY_BUCKETS = [
    (0, 100, "1) $0 - $100"),
    (100, 200, "2) $100 - $200"),
    (200, 350, "3) $200 - $350"),
    (350, 500, "4) $350 - $500"),
    (500, 750, "5) $500 - $750"),
    (750, 1000, "6) $750 - $1,000"),
    (1000, float("inf"), "7) $1,000 +"),
]

# Raw category spelling must match the original Hillstrom dataset exactly
# (the dataset itself contains the typo "Surburban").
ZIP_DISPLAY_TO_RAW = {"Rural": "Rural", "Suburban": "Surburban", "Urban": "Urban"}
VALID_ZIP_MAP = {"rural": "Rural", "suburban": "Surburban", "surburban": "Surburban", "urban": "Urban"}
CHANNEL_OPTIONS = ["Phone", "Web", "Multichannel"]
VALID_CHANNEL_MAP = {"phone": "Phone", "web": "Web", "multichannel": "Multichannel"}

REQUIRED_BULK_COLS = {"recency", "history", "mens", "womens", "newbie", "zip_code", "channel"}
LARGE_FILE_ROW_THRESHOLD = 5000

# Variable cost assumption (documented in budget_optimization.py — synthetic,
# based on customer's historical channel preference, not real cost data)
COST_MULTICHANNEL_BASELINE = 0.10
COST_WEB = 0.15
COST_PHONE = 0.50

SEGMENT_MEANING = {
    "Persuadables": "Ye customer type wo hai jo **email milne pe hi convert karega, bina email ke nahi karega** — sabse valuable customer hai kyunki marketing effort ise genuinely influence kar raha hai, paisa waste nahi ho raha.",
    "Sure Things": "Ye customer **bina email ke bhi convert kar lega** — email bhejna iske liye zyada matlab nahi rakhta, budget kahin aur use karna behtar hai.",
    "Lost Causes": "Ye customer **na email se na bina email ke convert karega** — targeting se koi real fayda nahi hai.",
    "Sleeping Dogs": "Email is customer ko **ulta irritate karke door bhaga sakta hai** — inhe target karna nuksaan de sakta hai.",
}

RECOMMENDATION_MEANING = {
    "Persuadables": "Haan, isko email bhejo — email se real fayda hoga.",
    "Sure Things": "Skip karo — ye waise bhi convert kar lega, budget bachao.",
    "Lost Causes": "Skip karo — na targeting se, na uske bina, koi fark nahi padega.",
    "Sleeping Dogs": "Avoid karo — email bhejne se ulta nuksaan ho sakta hai.",
}


# ---------- CACHED LOADERS ----------
@st.cache_resource
def load_models():
    t_learner = joblib.load(T_LEARNER_PATH)
    baseline_model = joblib.load(BASELINE_MODEL_PATH)
    return t_learner, baseline_model


@st.cache_data
def load_thresholds():
    df = pd.read_csv(SEGMENTATION_CSV_PATH)
    baseline_median = float(np.median(df["baseline_conversion_prob"]))
    uplift_median_positive = float(np.median(df.loc[df["uplift_score"] >= 0, "uplift_score"]))
    return baseline_median, uplift_median_positive


@st.cache_data
def load_avg_order_value():
    # Used only to translate uplift into an illustrative $ ROI figure.
    # NOTE: this is an estimate (mean spend among historical converters),
    # not a guaranteed revenue number — flagged clearly in the UI.
    try:
        raw = pd.read_csv(HILLSTROM_DATA_PATH)
        converters = raw[raw["conversion"] == 1]
        return float(converters["spend"].mean())
    except Exception:
        return 100.0  # fallback assumption if raw data isn't available in deployment


@st.cache_data
def load_qini_info():
    try:
        scores = joblib.load(QINI_SCORES_PATH)
        return scores
    except Exception:
        return None


# ---------- FEATURE ENGINEERING (shared by single-customer + bulk CSV) ----------
def bucket_history(history_value):
    for low, high, label in HISTORY_BUCKETS:
        if low <= history_value < high:
            return label
    return HISTORY_BUCKETS[-1][2]


def encode_features(raw_df):
    """raw_df must have columns: recency, history, mens, womens, newbie, zip_code, channel"""
    df = raw_df.copy()
    df["history_segment"] = df["history"].apply(bucket_history)
    df = pd.get_dummies(df, columns=["history_segment", "zip_code", "channel"], drop_first=False)
    df = df.reindex(columns=FEATURE_COLS, fill_value=0)
    df = df.astype(float)  # avoid object-dtype arrays from mixed bool/int dummy columns
    return df


def get_cost(channel):
    if channel == "Phone":
        return COST_PHONE
    elif channel == "Web":
        return COST_WEB
    return COST_MULTICHANNEL_BASELINE


def segment_customer(uplift, baseline_prob, baseline_median, uplift_median_positive):
    if uplift < 0:
        return "Sleeping Dogs"
    elif baseline_prob >= baseline_median and uplift < uplift_median_positive:
        return "Sure Things"
    elif uplift >= uplift_median_positive:
        return "Persuadables"
    else:
        return "Lost Causes"


def get_recommendation(segment):
    mapping = {
        "Persuadables": "TARGET ✅ — high incremental value",
        "Sure Things": "SKIP ⚪ — will convert regardless, saves budget",
        "Lost Causes": "SKIP ⚪ — unlikely to convert either way",
        "Sleeping Dogs": "AVOID ⛔ — targeting may reduce conversion",
    }
    return mapping[segment]


def generate_explanation(uplift, baseline_prob, segment, cost, roi, channel, avg_order_value):
    baseline_pct = baseline_prob * 100
    uplift_pct = uplift * 100

    text = f"""
**1. Baseline Conversion Probability: {baseline_prob:.4f} ({baseline_pct:.2f}%)**
Matlab: agar customer ko koi email NAHI bhejte, to bhi khud se naturally shopping karne ke chances **{baseline_pct:.2f}%** hain. Ye uska "bina kisi push ke" natural buying tendency hai.

**2. Uplift Score: {uplift:.4f}**
Matlab: agar customer ko email bhejte ho, to uske conversion chances mein **{uplift_pct:.2f}% ka EXTRA (incremental) boost** milega — email ki wajah se genuinely zyada chance banta hai ki wo khareede. Ye batata hai ki email ka is customer pe **kitna real asar** hoga, sirf ye nahi ki wo convert karega ya nahi.

**3. Segment: {segment}**
{SEGMENT_MEANING[segment]}

**4. Recommendation**
{RECOMMENDATION_MEANING[segment]}

**5. Estimated Cost: ${cost:.2f}**
Matlab: is customer ko treat karne mein **${cost:.2f}** kharcha aayega — "{channel}" historical channel hone ki wajah se (Step 6 ki cost assumption ke hisaab se).

**6. Estimated ROI: {roi:.1f}x**
Matlab: jitna paisa is customer pe kharch kar rahe ho (${cost:.2f}), uske against uplift se jo value milegi (uplift × avg order value ${avg_order_value:.2f}), wo us kharche ka **~{roi:.1f} guna** hai.
"""
    return text


# ---------- VALIDATION HELPERS ----------
def normalize_binary(value):
    """Returns 0/1, or None if unrecognized (caller decides fallback)."""
    if pd.isna(value):
        return None
    val_str = str(value).strip().lower()
    if val_str in ["1", "yes", "y", "true"]:
        return 1
    elif val_str in ["0", "no", "n", "false"]:
        return 0
    return None


def validate_and_clean_bulk_df(raw_df):
    """
    Cleans a bulk-upload dataframe, defaulting/dropping bad values instead of
    crashing. Returns (cleaned_df, warnings_list, dropped_row_count).
    """
    warnings = []
    df = raw_df.copy()
    original_len = len(df)

    # --- numeric coercion for recency/history ---
    for col in ["recency", "history"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    bad_numeric_rows = df.index[df["recency"].isna() | df["history"].isna()].tolist()
    if bad_numeric_rows:
        warnings.append(
            f"'recency'/'history' column mein {len(bad_numeric_rows)} row(s) non-numeric ya khaali thi "
            f"(row #: {[r + 1 for r in bad_numeric_rows]}) — ye rows result se hata di gayi hain."
        )
    df = df.dropna(subset=["recency", "history"]).reset_index(drop=True)

    if df.empty:
        return df, warnings, original_len

    # --- negative value clipping ---
    neg_mask = (df["recency"] < 0) | (df["history"] < 0)
    if neg_mask.any():
        bad_rows = df.index[neg_mask].tolist()
        warnings.append(
            f"{neg_mask.sum()} row(s) mein negative recency/history thi (row #: {[r + 1 for r in bad_rows]}) "
            f"— 0 pe clip kar diya gaya hai."
        )
        df["recency"] = df["recency"].clip(lower=0)
        df["history"] = df["history"].clip(lower=0)

    # --- binary columns: mens, womens, newbie ---
    for col in ["mens", "womens", "newbie"]:
        cleaned = df[col].apply(normalize_binary)
        bad_mask = cleaned.isna()
        if bad_mask.any():
            bad_rows = df.index[bad_mask].tolist()
            warnings.append(
                f"Column '{col}': {bad_mask.sum()} row(s) mein unrecognized value thi "
                f"(row #: {[r + 1 for r in bad_rows]}) — default 'No' (0) maan liya gaya hai."
            )
            cleaned = cleaned.fillna(0)
        df[col] = cleaned.astype(int)

    # --- zip_code normalization ---
    def clean_zip(v):
        return VALID_ZIP_MAP.get(str(v).strip().lower())

    cleaned_zip = df["zip_code"].apply(clean_zip)
    bad_mask = cleaned_zip.isna()
    if bad_mask.any():
        bad_rows = df.index[bad_mask].tolist()
        bad_values = df.loc[bad_mask, "zip_code"].unique().tolist()
        warnings.append(
            f"Column 'zip_code': {bad_mask.sum()} row(s) mein unrecognized value thi "
            f"(values: {bad_values}, row #: {[r + 1 for r in bad_rows]}) — default 'Rural' maan liya gaya hai."
        )
        cleaned_zip = cleaned_zip.fillna("Rural")
    df["zip_code"] = cleaned_zip

    # --- channel normalization ---
    def clean_channel(v):
        return VALID_CHANNEL_MAP.get(str(v).strip().lower())

    cleaned_channel = df["channel"].apply(clean_channel)
    bad_mask = cleaned_channel.isna()
    if bad_mask.any():
        bad_rows = df.index[bad_mask].tolist()
        bad_values = df.loc[bad_mask, "channel"].unique().tolist()
        warnings.append(
            f"Column 'channel': {bad_mask.sum()} row(s) mein unrecognized value thi "
            f"(values: {bad_values}, row #: {[r + 1 for r in bad_rows]}) — default 'Multichannel' maan liya gaya hai."
        )
        cleaned_channel = cleaned_channel.fillna("Multichannel")
    df["channel"] = cleaned_channel

    dropped_count = original_len - len(df)
    return df, warnings, dropped_count


# ---------- LOAD EVERYTHING ----------
try:
    t_learner, baseline_model = load_models()
    baseline_median, uplift_median_positive = load_thresholds()
    avg_order_value = load_avg_order_value()
    qini_info = load_qini_info()
    models_loaded = True
    load_error = None
except Exception as e:
    models_loaded = False
    load_error = str(e)

# ---------- SIDEBAR ----------
with st.sidebar:
    st.header("🎯 Smart Targeting Engine")
    st.write(
        "Causal AI (uplift modeling) se batata hai **kise** marketing email bhejni chahiye — "
        "sirf 'kaun convert karega' nahi, balki 'kis par email ka genuinely asar padega'."
    )
    st.markdown("**Dataset:** Hillstrom E-Mail Analytics (42,613 customers)")
    if models_loaded and qini_info is not None:
        st.markdown(
            f"**Model:** {qini_info.get('best_model', 'T-Learner')} "
            f"(Qini = {qini_info.get('t_learner_qini', 0):.3f})"
        )
    st.markdown("---")
    st.caption("Built by Harsh — Smart Targeting Engine")

# ---------- HEADER ----------
st.title("🎯 Smart Targeting Engine")
st.caption("Causal AI for Marketing — Uplift Modeling on the Hillstrom E-Mail Dataset")

if not models_loaded:
    st.error(
        "⚠️ Models ya data load nahi ho paaye. Please check ki 'models/' aur 'data/' "
        "folders app.py ke saath sahi jagah pe hain."
    )
    with st.expander("Technical details"):
        st.code(load_error)
    st.stop()

with st.expander("📊 Model Performance (Qini Curve) — kaise pata chala ye best model hai"):
    if QINI_PLOT_PATH.exists():
        st.image(str(QINI_PLOT_PATH), caption="Qini Curve: T-Learner vs X-Learner (5-fold CV, out-of-fold)")
        st.caption(
            "Qini curve ye dikhata hai ki agar customers ko model ke uplift-score se rank karke top-N ko "
            "target karein, to actual incremental conversions (blue line) random targeting (grey dashed line) "
            "se kitni zyada milti hain. Jitna bada gap curve aur random-line ke beech, utna behtar model."
        )
    else:
        st.info("Qini curve image nahi mili — 'models/qini_curve.png' check karein.")

tab1, tab2 = st.tabs(["👤 Single Customer Check", "📂 Bulk CSV Upload"])

# ================= TAB 1: SINGLE CUSTOMER =================
with tab1:
    st.subheader("Check one customer")

    with st.form("single_customer_form"):
        col1, col2 = st.columns(2)

        with col1:
            recency = st.number_input("Recency (months since last purchase)", min_value=0, max_value=36, value=5)
            history = st.number_input("Total historical spend ($)", min_value=0.0, value=250.0, step=10.0)
            zip_display = st.selectbox("Zip code type", ["Rural", "Suburban", "Urban"])

        with col2:
            mens = st.selectbox("Purchased mens item?", ["No", "Yes"])
            womens = st.selectbox("Purchased womens item?", ["No", "Yes"])
            newbie = st.selectbox("New customer?", ["No", "Yes"])
            channel = st.selectbox("Historical channel", CHANNEL_OPTIONS)

        submitted = st.form_submit_button("Predict")

    if submitted:
        if recency < 0 or history < 0:
            st.error("Recency aur history negative nahi ho sakte. Please sahi values daalein.")
        else:
            try:
                with st.spinner("Predicting..."):
                    raw_row = pd.DataFrame([{
                        "recency": recency,
                        "history": history,
                        "mens": 1 if mens == "Yes" else 0,
                        "womens": 1 if womens == "Yes" else 0,
                        "newbie": 1 if newbie == "Yes" else 0,
                        "zip_code": ZIP_DISPLAY_TO_RAW[zip_display],
                        "channel": channel,
                    }])

                    X = encode_features(raw_row).values

                    uplift = float(t_learner.predict(X).flatten()[0])
                    baseline_prob = float(baseline_model.predict_proba(X)[:, 1][0])
                    segment = segment_customer(uplift, baseline_prob, baseline_median, uplift_median_positive)
                    recommendation = get_recommendation(segment)
                    cost = get_cost(channel)
                    expected_value = uplift * avg_order_value
                    roi = expected_value / cost if cost > 0 else 0

                st.markdown("### Result")
                c1, c2, c3 = st.columns(3)
                c1.metric("Segment", segment)
                c2.metric("Uplift Score", f"{uplift:.4f}")
                c3.metric("Baseline Probability", f"{baseline_prob:.2%}")

                c4, c5, c6 = st.columns(3)
                c4.metric("Estimated Cost", f"${cost:.2f}")
                c5.metric("Estimated ROI", f"{roi:.1f}x")
                c6.metric("Recommendation", recommendation.split(" ")[0])

                st.write(f"**Recommendation:** {recommendation}")
                st.caption(
                    f"ROI is illustrative: uplift x estimated avg order value (${avg_order_value:.2f}, "
                    f"based on historical converter spend) / contact cost. Not a guaranteed revenue figure."
                )

                with st.expander("ℹ️ Ye numbers ka matlab kya hai? (Explanation)"):
                    st.markdown(generate_explanation(uplift, baseline_prob, segment, cost, roi, channel, avg_order_value))

            except Exception as e:
                st.error(f"Prediction mein kuch galat ho gaya. Please dobara try karein. Details: {e}")

# ================= TAB 2: BULK CSV =================
with tab2:
    st.subheader("Upload a customer list")

    template_df = pd.DataFrame([
        {"recency": 2, "history": 100, "mens": 1, "womens": 0, "newbie": 1, "zip_code": "Rural", "channel": "Web"},
        {"recency": 10, "history": 500, "mens": 0, "womens": 1, "newbie": 0, "zip_code": "Urban", "channel": "Phone"},
        {"recency": 6, "history": 250, "mens": 1, "womens": 0, "newbie": 0, "zip_code": "Suburban", "channel": "Multichannel"},
    ])
    template_buffer = io.StringIO()
    template_df.to_csv(template_buffer, index=False)
    st.download_button(
        "📥 Download CSV template",
        data=template_buffer.getvalue(),
        file_name="customer_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader(
        "Upload CSV (columns: recency, history, mens, womens, newbie, zip_code, channel)", type="csv"
    )

    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"CSV padhi nahi ja saki — file format check karein. Details: {e}")
            st.stop()

        if raw_df.empty:
            st.error("Uploaded CSV khaali hai — kam se kam ek customer row honi chahiye.")
            st.stop()

        missing = REQUIRED_BULK_COLS - set(raw_df.columns)
        if missing:
            st.error(
                f"CSV mein ye required columns missing hain: {', '.join(missing)}. "
                f"Please template download karke format match karein."
            )
            st.stop()

        if len(raw_df) > LARGE_FILE_ROW_THRESHOLD:
            st.warning(f"{len(raw_df)} rows mili — processing mein thoda time lag sakta hai, please wait.")

        with st.spinner(f"Predicting for {len(raw_df)} customers..."):
            cleaned_df, cleaning_warnings, dropped_count = validate_and_clean_bulk_df(raw_df)

            if cleaned_df.empty:
                st.error(
                    "Saari rows invalid nikli (recency/history numeric nahi thi) — koi bhi row process nahi ho payi. "
                    "Please CSV check karein ya template use karein."
                )
                st.stop()

            X = encode_features(cleaned_df).values
            uplift_scores = t_learner.predict(X).flatten()
            baseline_probs = baseline_model.predict_proba(X)[:, 1]

            segments = [
                segment_customer(u, b, baseline_median, uplift_median_positive)
                for u, b in zip(uplift_scores, baseline_probs)
            ]
            costs = cleaned_df["channel"].apply(get_cost).values
            recommendations = [get_recommendation(s) for s in segments]

            results_df = cleaned_df.copy()
            results_df["uplift_score"] = uplift_scores.round(4)
            results_df["baseline_probability"] = baseline_probs.round(4)
            results_df["segment"] = segments
            results_df["cost"] = costs
            results_df["recommendation"] = recommendations

        if dropped_count > 0:
            st.warning(f"⚠️ {dropped_count} row(s) ko result se hata diya gaya (invalid recency/history). Detail neeche dekhein.")
        for w in cleaning_warnings:
            st.warning(w)

        n_total = len(results_df)
        n_target = int((results_df["segment"] == "Persuadables").sum())
        expected_conversions = results_df.loc[results_df["segment"] == "Persuadables", "uplift_score"].sum()
        total_cost = results_df.loc[results_df["segment"] == "Persuadables", "cost"].sum()

        st.markdown("### Summary")
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Total customers processed", n_total)
        s2.metric("Recommended to target", f"{n_target} ({n_target / n_total:.0%})")
        s3.metric("Expected incremental conversions", f"{expected_conversions:.2f}")
        s4.metric("Total cost (Persuadables only)", f"${total_cost:.2f}")

        st.markdown("### Segment distribution")
        segment_order = ["Persuadables", "Sure Things", "Lost Causes", "Sleeping Dogs"]
        segment_counts = results_df["segment"].value_counts().reindex(segment_order, fill_value=0)
        st.bar_chart(segment_counts)

        st.markdown("### Full results")

        SEGMENT_COLORS = {
            "Persuadables": "background-color: rgba(44, 160, 44, 0.25)",   # green
            "Sure Things": "background-color: rgba(127, 127, 127, 0.25)",  # grey
            "Lost Causes": "background-color: rgba(127, 127, 127, 0.15)",  # light grey
            "Sleeping Dogs": "background-color: rgba(214, 39, 40, 0.25)",  # red
        }

        def highlight_segment(row):
            style = SEGMENT_COLORS.get(row["segment"], "")
            return [style] * len(row)

        try:
            styled_results = results_df.style.apply(highlight_segment, axis=1)
            st.dataframe(styled_results, use_container_width=True)
        except Exception:
            # Fallback if styling isn't supported in this Streamlit version/environment
            st.dataframe(results_df, use_container_width=True)

        output_buffer = io.StringIO()
        results_df.to_csv(output_buffer, index=False)
        st.download_button(
            "📥 Download results CSV",
            data=output_buffer.getvalue(),
            file_name="targeting_results.csv",
            mime="text/csv",
        )

st.markdown("---")
st.caption("Smart Targeting Engine — Causal AI Marketing Project | Built with CausalML, scikit-learn & Streamlit")