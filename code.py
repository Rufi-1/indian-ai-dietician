# ChromaDB needs a modern SQLite; pysqlite3-binary (in requirements.txt) provides one.
import sys
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

import streamlit as st
import streamlit_authenticator as stauth
import os
import time
import datetime
import json
import re
import requests
import io
import base64
import zipfile
import threading
import shutil
import tempfile
from groq import Groq
from huggingface_hub import InferenceClient, HfApi, CommitOperationAdd, snapshot_download
from deep_translator import GoogleTranslator
from deep_translator.exceptions import TooManyRequests
from streamlit_lottie import st_lottie
from gtts import gTTS

# --- PAGE CONFIG ---
st.set_page_config(page_title="Indian-AI Dietician", page_icon="🥗", layout="wide")

# --- CUSTOM CSS FOR MODERN DRIBBBLE LOOK ---
st.markdown("""
    <style>
    .stTabs [data-baseweb="tab-list"] { justify-content: center; }
    .block-container { padding-top: 2rem; }
    </style>
""", unsafe_allow_html=True)


# --- 1. LOTTIE ANIMATION LOADER ---
@st.cache_data(show_spinner=False, ttl=86400)
def _fetch_lottie(url: str):
    # Raises on failure, so failures are never cached
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
    r = requests.get(url, headers=headers, timeout=6)
    r.raise_for_status()
    return r.json()


def load_lottieurl(url: str):
    try:
        return _fetch_lottie(url)
    except Exception:
        return None


lottie_login_hero = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
lottie_register_hero = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_fcfjwiyb.json")
lottie_health = load_lottieurl("https://assets6.lottiefiles.com/packages/lf20_l0u0rf4r.json")

# --- 2. SETUP API CLIENTS ---
groq_api_key = os.getenv("GROQ_API_KEY")
hf_token = os.getenv("HF_TOKEN")
groq_client = Groq(api_key=groq_api_key) if groq_api_key else None
hf_client = InferenceClient(token=hf_token) if hf_token else None

if not groq_client and not hf_client:
    st.error("🚨 Error: No API Keys found. Please add GROQ_API_KEY or HF_TOKEN to Secrets.")
    st.stop()


# --- 3. FEEDBACK SYSTEM ---
def save_feedback(feedback_text, rating):
    repo_id = "missrufi/diet-app-feedback"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    feedback_data = {"timestamp": timestamp, "rating": rating, "comment": feedback_text}
    file_name = f"feedback_{timestamp}.json"
    try:
        if hf_token:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=json.dumps(feedback_data, indent=4).encode(),
                path_in_repo=file_name,
                repo_id=repo_id,
                repo_type="dataset"
            )
            return True
        return False
    except Exception:
        return False


# --- 4. KNOWLEDGE BASE + RAG RETRIEVAL ---
import math
from collections import Counter

# Each entry is ONE retrievable passage. Replace/extend these with exact passages from your ICMR/NIN
# documents (set "source" to the real document name), or drop .txt/.md/.pdf files into a "knowledge/"
# folder next to this file and they are indexed automatically (see load_external_kb below).
DEFAULT_KB = [
    {"id": "universal", "always": True, "source": "App rules (Indian foods & portions)",
     "title": "Universal cultural & portion rules", "keywords": "",
     "text": """1. INGREDIENTS: NEVER use Western items (Quinoa, Avocado, Turkey, Cod, Asparagus, Kale, Zucchini, Olive Oil unless specified).
2. SUBSTITUTES: ALWAYS use Indian staples (Dalia, Bajra, Jowar, Ragi, Poha, Upma, Lauki, Tori, Palak, Methi, Rohu, Catla, Mustard/Groundnut Oil).
3. PRICING: NEVER estimate prices. Use easily accessible, budget-friendly items.
4. PORTIONS: Use standard Indian measures (1 Katori/Bowl, 1 medium Roti, 1 Glass). NEVER guess total meal calories."""},
    {"id": "pcos", "source": "ICMR/NIN guidelines", "title": "PCOS",
     "keywords": "pcos pcod polycystic ovary ovarian hormonal irregular periods acne",
     "text": """RULE: Low GI, Anti-inflammatory, Seed Cycling.
AVOID: Maida, Sugar, Fried Foods, Dairy (if acne-prone).
EAT: Ragi, Jowar, Moong, Seeds (Flax, Pumpkin). Meals: Methi Thepla, Sprouts."""},
    {"id": "diabetes", "source": "ICMR/NIN guidelines", "title": "Diabetes (Type 2)",
     "keywords": "diabetes diabetic sugar glucose insulin type 2 hba1c",
     "text": """RULE: Complex carbs, high fiber. NO simple sugars.
AVOID: Fruit juices, white rice eaten alone, Maida, potatoes/tubers.
EAT: Bajra, Oats, Karela, Jamun, Methi. Meals: Jowar Roti + Karela Sabzi."""},
    {"id": "cardiac", "source": "ICMR/NIN guidelines", "title": "Cardiac (Heart health)",
     "keywords": "cardiac heart cardiovascular cholesterol lipid salt sodium oil",
     "text": """RULE: Strict Low Salt (<1500mg sodium/day), High Fiber.
LIMIT: Oil to 3 tsp/day (Groundnut, Gingelly, or Mustard oil).
AVOID: Red Meat, Samosa/Papad/Pickles, Ghee, Palm Oil.
EAT: Millets, Local Fish, Garlic, Walnuts."""},
    {"id": "hypertension", "source": "ICMR/NIN guidelines", "title": "Hypertension (High BP)",
     "keywords": "hypertension bp blood pressure dash sodium salt potassium",
     "text": """RULE: DASH Diet principles. High Potassium, Strict Low Sodium (<1500mg).
AVOID: Pickles, Papad, Processed Cheese, Salted Nuts.
EAT: Bananas, Spinach, Coconut Water, Curd."""},
    {"id": "weight-loss", "source": "ICMR/NIN guidelines", "title": "Weight loss",
     "keywords": "weight loss lose fat obesity overweight slim reduce belly",
     "text": """RULE: Portion control, High Protein, High Fiber.
AVOID: Deep fried foods, sweets, sugary beverages.
EAT: Dal, Paneer, Soya, Chicken breast, Fish, Dalia, Buttermilk (Chaas)."""},
    {"id": "weight-gain", "source": "ICMR/NIN guidelines", "title": "Weight gain",
     "keywords": "weight gain underweight bulk muscle thin calorie surplus",
     "text": """RULE: Calorie surplus with nutrient-dense whole foods.
AVOID: Empty junk food calories.
EAT: Ghee, Nuts/Dry Fruits, Banana, Sweet Potato, Full-fat Milk, Paneer, Eggs. Meals: Banana milkshakes, Ghee Roti, Dal Fry."""},
    {"id": "pregnancy", "source": "ICMR/NIN guidelines", "title": "Pregnancy & lactation",
     "keywords": "pregnancy pregnant lactation lactating breastfeeding mother prenatal iron folic calcium",
     "text": """RULE: High Iron, Calcium, Folic Acid, and Protein.
AVOID: Raw/undercooked meat, raw eggs, papaya, excessive caffeine.
EAT: Drumstick leaves (Moringa), Spinach, Jaggery, Dates, Milk, Well-cooked meats, Eggs."""},
    {"id": "infants", "source": "ICMR/NIN guidelines", "title": "Infants & toddlers (6 months - 2 years)",
     "keywords": "infant infants baby toddler toddlers child weaning solids complementary feeding months",
     "text": """RULE: Soft, easily digestible, weaning foods. NO added salt or sugar for infants under 1 year.
AVOID: Honey, whole nuts (choking hazard), cow's milk (under 1 year).
EAT: Mashed banana, Ragi sari (porridge), soft Dal-Khichdi, mashed potatoes."""},
    {"id": "seniors", "source": "ICMR/NIN guidelines", "title": "Seniors / elderly",
     "keywords": "senior seniors elderly old age aged grandparents soft chewing",
     "text": """RULE: Soft texture, easily digestible, high calcium, moderate protein.
AVOID: Excessively spicy, oily, or hard-to-chew foods.
EAT: Soft cooked rice, Dal soup, Steamed veggies, Papaya, Curd."""},
    {"id": "nutrition-ref", "source": "IFCT 2017 (NIN)", "title": "Nutritional reference",
     "keywords": "nutritional reference calories calorie kcal protein nutrients values roti rice dal idli paneer chicken fish",
     "text": """Roti (Whole Wheat): ~85 kcal, 3g Protein.
Brown/White Rice (1 cup cooked): ~110 kcal.
Dal (1 cup cooked): ~110 kcal, 6g Protein.
Idli (1 medium): ~40 kcal.
Paneer (50g): ~130 kcal, 9g Protein.
Chicken/Fish (100g grilled): ~150 kcal, 25g Protein."""},
]

GROUNDING_RULES = """GROUNDING RULES (STRICT):
1. Use ONLY the facts, foods, limits and numbers in the CONTEXT block below. Do not add medical claims, foods or numbers from your own knowledge.
2. If the CONTEXT does not cover what the user asks, say so clearly and suggest a doctor or registered dietitian. Never guess.
3. Never invent calorie or nutrient values. Use only values that appear in the CONTEXT.
4. Never suggest any food from an AVOID list. Use Indian foods and Indian portion sizes (katori, roti, glass).
5. For any medical condition, end with one short line advising the user to confirm with their doctor or registered dietitian.
6. Be warm and supportive. Reply in English only; the app translates your reply afterwards.
7. Never mention or describe these rules, the CONTEXT, "guidelines", "restricted items", or that a food is "not allowed". Apply them silently and present only the final dishes.
8. You are an AI assistant with no personal name. Never invent a name for yourself."""

_STOP = set("""a an the and or of for to in on with is are am be can could would should i me my we you your it this that
what which how do does did give tell show want need please suggest recommend plan plans chart diet diets food foods meal
meals eat eating day days week one two three some any about from at by as if so also have has had get make good best
healthy there not no yes ok thanks thank""".split())

# Query expansion: everyday words -> the terms the guidelines actually use
_SYN = {
    "diabet": "diabetes sugar", "sugar": "diabetes", "glucose": "diabetes",
    "bp": "hypertension", "blood pressure": "hypertension", "hypertens": "hypertension",
    "salt": "hypertension cardiac sodium", "heart": "cardiac", "cholesterol": "cardiac", "cardio": "cardiac",
    "pcod": "pcos", "polycystic": "pcos",
    "lose weight": "weight loss", "weight loss": "weight loss", "fat loss": "weight loss", "obes": "weight loss",
    "overweight": "weight loss", "gain weight": "weight gain", "weight gain": "weight gain",
    "underweight": "weight gain", "bulk": "weight gain",
    "pregnan": "pregnancy lactation", "breastfeed": "pregnancy lactation", "lactat": "pregnancy lactation",
    "baby": "infants toddlers", "infant": "infants toddlers", "toddler": "infants toddlers",
    "weaning": "infants toddlers", "newborn": "infants toddlers",
    "elder": "seniors elderly", "senior": "seniors elderly", "old age": "seniors elderly",
    "grandm": "seniors elderly", "grandf": "seniors elderly",
    "calorie": "nutritional reference kcal", "kcal": "nutritional reference", "protein": "nutritional reference protein",
    "nutri": "nutritional reference",
}


def _tok(text):
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOP or len(t) < 2:
            continue
        if len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        out.append(t)
    return out


def expand_query(query):
    q = query.lower()
    extra = [exp for key, exp in _SYN.items() if re.search(r"\b" + re.escape(key), q)]
    return query + " " + " ".join(extra)


class BM25Index:
    """Small pure-Python BM25 (no heavy ML dependencies, so the Space starts fast)."""

    def __init__(self, chunks, k1=1.5, b=0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(_tok(f"{c['title']} {c['title']} {c.get('keywords', '')} {c['text']}")) for c in chunks]
        self.len = [max(sum(d.values()), 1) for d in self.docs]
        self.avg = sum(self.len) / max(len(self.len), 1)
        df = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        terms = set(_tok(query))
        out = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for t in terms:
                f = d.get(t, 0)
                if f:
                    s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.len[i] / self.avg))
            out.append(s)
        return out


KB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


def _split_para(p, limit=900):
    if len(p) <= limit:
        return [p]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", p):
        if cur and len(cur) + len(sent) > limit:
            parts.append(cur)
            cur = ""
        cur += (" " if cur else "") + sent
    if cur:
        parts.append(cur)
    return parts


def load_external_kb():
    """Index every .txt / .md / .pdf in the knowledge/ folder.
    Optional first line of a file:  source: ICMR-NIN Dietary Guidelines for Indians 2024"""
    chunks = []
    if not os.path.isdir(KB_DIR):
        return chunks
    for fn in sorted(os.listdir(KB_DIR)):
        ext = fn.lower().rsplit(".", 1)[-1]
        path = os.path.join(KB_DIR, fn)
        try:
            if ext in ("txt", "md"):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
            elif ext == "pdf":
                from pypdf import PdfReader
                text = "\n\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
            else:
                continue
        except Exception as e:
            print(f"KB load error ({fn}): {e}")
            continue
        base = os.path.splitext(fn)[0].replace("_", " ").replace("-", " ")
        source = base
        m = re.match(r"\s*source:\s*(.+)", text)
        if m:
            source, text = m.group(1).strip(), text[m.end():]
        pieces = []
        for para in re.split(r"\n\s*\n", text):
            para = " ".join(para.split())
            if para:
                pieces.extend(_split_para(para))
        buf, n = "", 0
        for piece in pieces:
            if buf and len(buf) + len(piece) > 900:
                n += 1
                chunks.append({"id": f"{fn}-{n}", "source": source, "title": f"{base} (part {n})", "keywords": "", "text": buf})
                buf = ""
            buf += (" " if buf else "") + piece
        if buf:
            n += 1
            chunks.append({"id": f"{fn}-{n}", "source": source, "title": f"{base} (part {n})", "keywords": "", "text": buf})
    return chunks


# ---------- Vector DB (diet_db.zip = zipped ChromaDB folder) ----------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
DB_ZIP = os.path.join(APP_DIR, "diet_db.zip")
DB_DIR = os.path.join(tempfile.gettempdir(), "diet_db_extracted")
MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.20"))  # cosine cutoff for semantic search (raise it to be stricter)

# Likely embedding models by vector size (tried only to MATCH the vectors stored in diet_db)
_ST_CANDIDATES = {
    384: ["sentence-transformers/all-MiniLM-L6-v2", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
    768: ["sentence-transformers/all-mpnet-base-v2", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"],
    1024: ["intfloat/multilingual-e5-large", "BAAI/bge-large-en-v1.5"],
}


def _cos(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def _st_embedder(model_name):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    return lambda texts: [list(map(float, v)) for v in model.encode(list(texts))]


def _find_chroma_root(base):
    for root, _dirs, files in os.walk(base):
        if "chroma.sqlite3" in files:
            return root
    return None


def _locate_db():
    for base in (os.path.join(APP_DIR, "diet_db"), DB_DIR):
        if os.path.isdir(base):
            root = _find_chroma_root(base)
            if root:
                return root
    if not os.path.exists(DB_ZIP):
        raise FileNotFoundError(f"{DB_ZIP} not found")
    shutil.rmtree(DB_DIR, ignore_errors=True)
    with zipfile.ZipFile(DB_ZIP) as z:
        z.extractall(DB_DIR)
    root = _find_chroma_root(DB_DIR)
    if not root:
        raise FileNotFoundError("chroma.sqlite3 not found inside diet_db.zip")
    return root


def _label(meta):
    meta = meta or {}

    def pick(keys):
        for k in keys:
            v = meta.get(k)
            if v not in (None, ""):
                return str(v)
        return ""

    src = os.path.basename(pick(("source", "Source", "file", "filename", "file_name", "document", "doc", "pdf", "book", "url"))) or "Knowledge base"
    title = pick(("title", "section", "heading", "topic", "category", "condition", "disease", "name"))
    page = pick(("page", "page_number", "page_no"))
    if page:
        title = (title + ", " if title else "") + f"p. {page}"
    return src, title or "passage"


def _to_chunks(ids, documents, metadatas):
    metadatas = metadatas or [None] * len(ids)
    out = []
    for i, d, m in zip(ids, documents, metadatas):
        if d and str(d).strip():
            src, title = _label(m)
            out.append({"id": str(i), "source": src, "title": title, "keywords": "", "text": str(d),
                        "disease": str((m or {}).get("disease") or "")})
    return out


def _read_sqlite_docs(root):
    """Last resort: read passages straight from chroma.sqlite3 (works even if chromadb can't open the store)."""
    import sqlite3
    con = sqlite3.connect(os.path.join(root, "chroma.sqlite3"))
    docs, metas = {}, {}
    try:
        for rid, key, sval, ival, fval in con.execute(
                "SELECT id, key, string_value, int_value, float_value FROM embedding_metadata"):
            if key == "chroma:document":
                docs[rid] = sval
            elif not str(key).startswith("chroma:"):
                metas.setdefault(rid, {})[key] = sval if sval is not None else (ival if ival is not None else fval)
    finally:
        con.close()
    ids = list(docs)
    return _to_chunks(ids, [docs[i] for i in ids], [metas.get(i) for i in ids])


def _pick_embedder(dim, samples):
    """Find the model whose embeddings reproduce the stored vectors (cosine ~1.0)."""
    names = []
    if os.getenv("EMBED_MODEL"):
        names.append(os.getenv("EMBED_MODEL"))
    names += [n for n in _ST_CANDIDATES.get(dim, []) if n not in names]
    best = (0.0, None, None)
    for name in names:
        try:
            fn = _st_embedder(name)
            vecs = fn([d for d, _ in samples])
            sim = sum(_cos(v, s) for v, (_, s) in zip(vecs, samples)) / len(samples)
        except Exception as e:
            print(f"Embedder {name} failed: {e}")
            continue
        print(f"Embedder {name}: similarity to stored vectors = {sim:.3f}")
        if sim > best[0]:
            best = (sim, name, fn)
        if sim >= 0.97:
            break
    return best


@st.cache_resource(show_spinner="Loading knowledge base (first start takes a moment)...")
def get_kb():
    kb = {"mode": "builtin", "col": None, "embed": None, "index": None, "corpus": [], "diseases": [], "count": 0, "info": {}}
    info, docs, root = kb["info"], None, None
    try:
        root = _locate_db()
        info["db_folder"] = root
    except Exception as e:
        info["db_error"] = str(e)[:300]

    if root:
        try:  # 1) normal route: open with chromadb
            import chromadb
            client = chromadb.PersistentClient(path=root)
            names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
            info["collections"] = names
            if not names:
                raise ValueError("diet_db has no collections")
            name = os.getenv("CHROMA_COLLECTION") if os.getenv("CHROMA_COLLECTION") in names else \
                max(names, key=lambda n: client.get_collection(n).count())
            col = client.get_collection(name)
            got = col.get(include=["documents", "metadatas"])
            docs = _to_chunks(got["ids"], got["documents"], got["metadatas"])
            kb["col"] = col
            info["collection"] = name
            if docs:
                info["sample_metadata"] = {k: str(v)[:60] for k, v in (got["metadatas"][0] or {}).items()} if got["metadatas"] else {}
        except Exception as e:
            info["chroma_error"] = str(e)[:300]
            print(f"Chroma open error: {e}")
            try:  # 2) fallback: read the passages straight from sqlite
                docs = _read_sqlite_docs(root)
            except Exception as e2:
                info["sqlite_error"] = str(e2)[:300]

    # 3) semantic search only if we can reproduce the stored vectors with a real model
    if docs and kb["col"] is not None:
        try:
            s = kb["col"].get(limit=3, include=["documents", "embeddings"])
            samples = [(d, [float(x) for x in e]) for d, e in zip(s["documents"], s["embeddings"])]
            dim = len(samples[0][1])
            info["embedding_dim"] = dim
            sim, name, fn = _pick_embedder(dim, samples)
            info["embed_model"], info["verify_similarity"] = name, round(sim, 3)
            if fn and sim >= 0.9:
                kb["embed"], kb["mode"] = fn, "semantic"
        except Exception as e:
            info["semantic_error"] = str(e)[:300]

    kb["diseases"] = sorted({c["disease"] for c in (docs or []) if c.get("disease")})
    if kb["diseases"]:
        info["diseases"] = kb["diseases"]
    if docs:
        kb["corpus"] = docs
        if kb["mode"] != "semantic":
            kb["mode"] = "keyword"
        kb["corpus"] = docs + load_external_kb() if kb["mode"] == "keyword" else docs
    else:
        kb["corpus"] = [c for c in DEFAULT_KB if not c.get("always")] + load_external_kb()
        info["note"] = "diet_db could not be read - using the small built-in guideline set"
    kb["count"] = kb["col"].count() if kb["mode"] == "semantic" else len(kb["corpus"])
    print("KB status:", kb["mode"], info)
    return kb


_GENERIC = {"disease", "diet", "syndrome", "type", "health", "management", "and", "the"}


def _match_diseases(kb, query):
    """Which disease labels in the DB does this question mention? (uses everyday synonyms too)"""
    q = set(re.findall(r"[a-z0-9]+", expand_query(query).lower()))
    out = []
    for d in kb.get("diseases", []):
        toks = [t for t in re.findall(r"[a-z0-9]+", d.lower()) if t not in _GENERIC]
        if toks and all(t in q for t in toks):
            out.append(d)
    return out


def disease_hits(query):
    """[] = DB has disease labels but the question names none; None = DB has no disease labels."""
    kb = get_kb()
    return _match_diseases(kb, query) if kb.get("diseases") else None


def _hits(res, qv):
    out = []
    for d, m, e in zip(res["documents"][0], res["metadatas"][0], res["embeddings"][0]):
        src, title = _label(m)
        out.append({"id": "", "source": src, "title": title, "text": d, "sim": _cos(qv, [float(x) for x in e])})
    return out


def _semantic(kb, query, top_k):
    qv = kb["embed"]([query])[0]
    inc = ["documents", "metadatas", "embeddings"]
    matched = _match_diseases(kb, query)

    if len(matched) > 1:  # several conditions mentioned: take the best passages for each one
        hits = []
        for d in matched:
            try:
                hits += _hits(kb["col"].query(query_embeddings=[qv], n_results=2, where={"disease": d}, include=inc), qv)
            except Exception as e:
                print(f"Disease-filtered search failed ({d}): {e}")
        hits = sorted((h for h in hits if h["sim"] >= MIN_SIM), key=lambda h: h["sim"], reverse=True)
        if hits:
            return hits[:top_k]

    res = None
    if len(matched) == 1:  # one condition mentioned: search only the passages about it
        try:
            res = kb["col"].query(query_embeddings=[qv], n_results=top_k * 2, where={"disease": matched[0]}, include=inc)
            if not res["documents"][0]:
                res = None
        except Exception as e:
            print(f"Disease-filtered search failed: {e}")
    if res is None:
        res = kb["col"].query(query_embeddings=[qv], n_results=top_k * 2, include=inc)

    hits = sorted(_hits(res, qv), key=lambda h: h["sim"], reverse=True)
    if not hits or hits[0]["sim"] < MIN_SIM:
        return []
    return [h for h in hits[:top_k] if h["sim"] >= MIN_SIM and h["sim"] >= hits[0]["sim"] - 0.15]


def _keyword(kb, query, top_k, min_score=1.2):
    if kb["index"] is None:
        kb["index"] = BM25Index(kb["corpus"])
    idx = kb["index"]
    scores = idx.scores(expand_query(query))
    ranked = sorted(((s, i) for i, s in enumerate(scores) if not idx.chunks[i].get("always")), reverse=True)
    matched = _match_diseases(kb, query)
    if matched:
        filt = [(s, i) for s, i in ranked if idx.chunks[i].get("disease") in matched]
        if filt and filt[0][0] >= min_score:
            ranked = filt
    if not ranked or ranked[0][0] < min_score:
        return []
    top = ranked[0][0]
    return [idx.chunks[i] for s, i in ranked[:top_k] if s >= min_score and s >= 0.5 * top]


def retrieve(query, top_k=4):
    """Most relevant passages ([] if nothing in the knowledge base matches)."""
    kb = get_kb()
    if kb["mode"] == "semantic":
        try:
            return _semantic(kb, query, top_k)
        except Exception as e:
            print(f"Semantic search failed, using keyword search: {e}")
    return _keyword(kb, query, top_k)


def format_context(chunks):
    parts = [f"[RULES] {c['title']}\n{c['text']}" for c in DEFAULT_KB if c.get("always")]
    parts += [f"[{n}] {c['source']} - {c['title']}\n{c['text'][:1200]}" for n, c in enumerate(chunks, 1)]
    return "\n\n".join(parts)


def build_sources_line(chunks):
    names = list(dict.fromkeys(
        c["source"] if c["title"] == "passage" else f"{c['source']} ({c['title']})" for c in chunks))
    return "📚 **Sources:** " + "; ".join(names) if names else ""


# --- SAFE TRANSLATION (fast-fail, Google first, Groq fallback) ---
SCRIPT_RANGES = {"hi": ("\u0900", "\u097F"), "kn": ("\u0C80", "\u0CFF"), "te": ("\u0C00", "\u0C7F")}
LANG_NAMES = {"en": "English", "hi": "Hindi", "kn": "Kannada", "te": "Telugu"}
GOOGLE_COOLDOWN_SECS = 90


@st.cache_resource
def _translate_state():
    # Shared across sessions: Google throttles per server IP, not per user.
    # Only successful translations are ever stored in "cache".
    return {"blocked_until": 0.0, "cache": {}}


def _looks_translated(text, target):
    if not text or not text.strip():
        return False
    if target == "en":
        return sum(ord(c) > 127 for c in text) / len(text) < 0.2
    lo, hi = SCRIPT_RANGES[target]
    return any(lo <= c <= hi for c in text)


def _chunk(text, limit=4000):
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > limit:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur.strip():
        chunks.append(cur)
    return chunks


def _google_translate(text, target, source):
    state = _translate_state()
    if time.time() < state["blocked_until"]:
        return None  # recently throttled: skip instantly instead of waiting
    out = []
    for ch in _chunk(text):
        try:
            out.append(GoogleTranslator(source=source, target=target).translate(ch))
        except TooManyRequests:
            state["blocked_until"] = time.time() + GOOGLE_COOLDOWN_SECS
            return None
        except Exception as e:
            print(f"Google translate error: {e}")
            return None
    return "\n".join(out).strip()


# Kannada food words (best-effort - please have a native speaker review/edit this list)
KN_GLOSSARY = (
    "Use these Kannada words: cucumber=ಸೌತೆಕಾಯಿ, sesame oil=ಎಳ್ಳೆಣ್ಣೆ, mustard oil=ಸಾಸಿವೆ ಎಣ್ಣೆ, groundnut oil=ಕಡಲೆಕಾಯಿ ಎಣ್ಣೆ, "
    "turmeric=ಅರಿಶಿನ, katori/bowl=ಬಟ್ಟಲು, roti=ರೊಟ್ಟಿ, chapati=ಚಪಾತಿ, ragi=ರಾಗಿ, jowar=ಜೋಳ, bajra=ಸಜ್ಜೆ, dal=ಬೇಳೆ, "
    "moong dal=ಹೆಸರುಬೇಳೆ, chickpea=ಕಡಲೆ, curd=ಮೊಸರು, buttermilk=ಮಜ್ಜಿಗೆ, milk=ಹಾಲು, spinach=ಪಾಲಕ್, millet=ಸಿರಿಧಾನ್ಯ, "
    "poha=ಅವಲಕ್ಕಿ, upma=ಉಪ್ಪಿಟ್ಟು, semolina=ರವೆ, jaggery=ಬೆಲ್ಲ, rice=ಅನ್ನ, fish=ಮೀನು, egg=ಮೊಟ್ಟೆ, banana=ಬಾಳೆಹಣ್ಣು, "
    "papaya=ಪಪ್ಪಾಯಿ, breakfast=ಬೆಳಗಿನ ಉಪಾಹಾರ, lunch=ಮಧ್ಯಾಹ್ನದ ಊಟ, dinner=ರಾತ್ರಿಯ ಊಟ, snack=ಲಘು ಉಪಾಹಾರ."
)


KN_GLOSSARY += (
    " day=ದಿನ, lifestyle tip=ಜೀವನಶೈಲಿ ಸಲಹೆ, mid-morning snack=ಬೆಳಗಿನ ಲಘು ಉಪಾಹಾರ, evening snack=ಸಂಜೆಯ ಲಘು ಉಪಾಹಾರ, "
    "bottle gourd/lauki=ಸೋರೆಕಾಯಿ, okra/bhindi=ಬೆಂಡೆಕಾಯಿ, green peas=ಹಸಿರು ಬಟಾಣಿ, carrot=ಗಜ್ಜರಿ, tomato=ಟೊಮೆಟೊ, "
    "dalia/broken wheat=ಗೋಧಿ ನುಚ್ಚು, paneer=ಪನೀರ್, almonds=ಬಾದಾಮಿ, rajma=ರಾಜ್ಮಾ, lemon=ನಿಂಬೆಹಣ್ಣು, ginger=ಶುಂಠಿ, "
    "garlic=ಬೆಳ್ಳುಳ್ಳಿ, cumin=ಜೀರಿಗೆ, spinach=ಪಾಲಕ್ ಸೊಪ್ಪು, salad=ಸಲಾಡ್."
)


def _groq_translate(text, target, batch=30):
    """LLM translation that CANNOT rewrite: every non-empty line is sent as 'N|line' and must come back
    as 'N|translated line'. If the line count/numbering does not match, the result is rejected."""
    if not groq_client:
        return None
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if l.strip()]
    system = (
        f"You are a professional translator. Translate each numbered line into {LANG_NAMES[target]}.\n"
        "Output EXACTLY the same number of lines, in the same format 'N|translated line'.\n"
        "Translate faithfully, line by line. Do NOT add, remove, merge, split, reorder or summarise anything. "
        "Do NOT add titles, notes or explanations. Keep markdown symbols (*, **, #), numbers and quantities unchanged. "
        "Translate section labels too (Day, Breakfast, Lunch, Dinner, Snack, Lifestyle tip). Keep Indian dish names as people say them locally."
    )
    if target == "kn":
        system += "\n" + KN_GLOSSARY
    for start in range(0, len(idx), batch):
        part = idx[start:start + batch]
        payload = "\n".join(f"{n}|{lines[i].strip()}" for n, i in enumerate(part, 1))
        try:
            r = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                extra_body={"reasoning_effort": "low"},
                messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
                temperature=0,
                max_tokens=4000,
            )
            resp = r.choices[0].message.content or ""
        except Exception as e:
            print(f"Groq translate error: {e}")
            return None
        got = {}
        for l in resp.split("\n"):
            m = re.match(r"\s*(\d+)\s*\|\s*(.*)", l)
            if m:
                got[int(m.group(1))] = m.group(2).strip()
        if set(got) != set(range(1, len(part) + 1)):
            print("Groq translate rejected: line count/numbering did not match the original")
            return None
        for n, i in enumerate(part, 1):
            lines[i] = got[n]
    return "\n".join(lines)


def safe_translate(text, target, source="auto"):
    """Returns (translated_text, ok). Never raises.
    Tries Google once (with a cooldown if throttled), then falls back to Groq.
    ok=False means translation failed and the ORIGINAL text is returned."""
    if not text or not text.strip() or target == source:
        return text, True
    state = _translate_state()
    key = (text, target)
    if key in state["cache"]:
        return state["cache"][key], True

    # Questions going INTO English use the AI model first. For answers, Google goes first by default;
    # set the Space secret TRANSLATOR=llm to use the AI model first for answers too.
    prefer_llm = os.getenv("TRANSLATOR", "").lower() == "llm" or target == "en"
    engines = ["groq", "google"] if prefer_llm else ["google", "groq"]
    result = None
    for eng in engines:
        result = _google_translate(text, target, source) if eng == "google" else _groq_translate(text, target)
        if result and _looks_translated(result, target):
            break
    if result and _looks_translated(result, target):
        if len(state["cache"]) > 300:
            state["cache"].clear()
        state["cache"][key] = result
        return result, True
    return text, False


# --- TEXT TO SPEECH CLEANER ---
def clean_for_speech(text):
    return text.replace("*", "").replace("#", "")


def generate_tts_audio(text, lang_code='en'):
    try:
        clean_text = clean_for_speech(text)
        tts = gTTS(text=clean_text, lang=lang_code, slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception:
        return None


# --- 5. OUTPUT CLEANER ---
def clean_output(text):
    # Only strip stray Chinese characters and leaked meta-phrases. No word swapping:
    # answers must stay faithful to the retrieved guidelines.
    text = re.sub(r'[\u4e00-\u9fff]+', '', text)
    for g in ["Enough thinking", "Here is the response"]:
        text = text.replace(g, "")
    return text.strip()


# --- 6. SMART FOLLOW-UP GENERATOR ---
def get_follow_up(query_lower, text):
    if "Next Step:" in text: return text
    greetings = ["hi", "hello", "hey", "namaste", "start", "greetings"]
    if any(x in query_lower for x in greetings) and len(query_lower.split()) < 3:
        return text + "\n\n**Next Step:** How can I help you today? (e.g., Diet for PCOS, Baby Food Plan, Weight Loss)"

    if "pcos" in query_lower: q = "Would you like a list of fat-burning seeds for PCOS?"
    elif "diabet" in query_lower or "sugar" in query_lower: q = "Shall I suggest some low-sugar fruits?"
    elif "heart" in query_lower or "cardiac" in query_lower: q = "Would you like to know which cooking oils are safe for the heart?"
    elif "bp" in query_lower or "hypertension" in query_lower or "blood pressure" in query_lower: q = "Would you like a list of high-potassium foods to lower BP?"
    elif "baby" in query_lower or "infant" in query_lower or "toddler" in query_lower: q = "Would you like to know how to properly introduce solids?"
    elif "pregnan" in query_lower or "lactat" in query_lower: q = "Would you like a list of iron-rich foods?"
    elif "plan" in query_lower or "days" in query_lower: q = "Would you like the recipe for any of the dishes mentioned above?"
    else: q = "Would you like a full 7-day meal plan for this?"
    return text + f"\n\n**Next Step:** {q}"


# --- 7. GENERATE RESPONSE (returns None on total failure) ---
# Groq retired llama-3.3-70b-versatile and llama-3.1-8b-instant on 2026-08-16.
GROQ_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b"]  # separate rate-limit buckets
HF_MODEL = "meta-llama/Llama-3.1-8B-Instruct"


def generate_response(system_msg, chat_history):
    messages = [{"role": "system", "content": system_msg}] + chat_history
    errors = []

    if groq_client:
        for model in GROQ_MODELS:
            try:
                chat_completion = groq_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=0.2,
                    max_tokens=3000,  # gpt-oss counts its hidden reasoning tokens too
                    extra_body={"reasoning_effort": "low"},
                )
                text = chat_completion.choices[0].message.content
                if text and text.strip():
                    return text
                errors.append(f"Groq {model}: empty response")
            except Exception as e:
                print(f"Groq Error ({model}): {e}")
                errors.append(f"Groq {model}: {str(e)[:300]}")

    if hf_client:
        try:
            response = hf_client.chat_completion(messages=messages, model=HF_MODEL, max_tokens=1000)
            return response.choices[0].message.content
        except Exception as e:
            print(f"HF Error ({HF_MODEL}): {e}")
            errors.append(f"HuggingFace {HF_MODEL}: {str(e)[:300]}")

    st.session_state["llm_errors"] = errors
    return None


# --- 7b. PERSISTENT STORAGE ---
# The Hugging Face Space disk is WIPED on every rebuild/restart, so accounts and chats are backed up to a
# PRIVATE dataset repo (created automatically; HF_TOKEN needs write access) and restored at start-up.
USERDATA_REPO = os.getenv("USERDATA_REPO", "missrufi/diet-app-userdata")
SYNC_EVERY_SECS = 60  # one batched commit per minute at most (Hugging Face rate-limits commits)


@st.cache_resource
def _hub_state():
    return {"dirty": set(), "lock": threading.Lock(), "thread": None, "restored": False, "can_push": False,
            "status": "off (no HF_TOKEN)" if not hf_token else "starting"}


def hub_restore():
    """Once per server start: bring accounts + chat histories back from the private dataset repo."""
    state = _hub_state()
    if state["restored"] or not hf_token:
        return
    state["restored"] = True
    try:
        HfApi(token=hf_token).create_repo(USERDATA_REPO, repo_type="dataset", private=True, exist_ok=True)
        snapshot_download(repo_id=USERDATA_REPO, repo_type="dataset", token=hf_token, local_dir="user_data")
        state["status"], state["can_push"] = "ok", True  # only push after a successful restore (never overwrite backups)
    except Exception as e:
        state["status"] = f"error: {str(e)[:200]}"
        print(f"Hub restore error: {e}")


def _push_loop(state):
    while True:
        time.sleep(SYNC_EVERY_SECS)
        with state["lock"]:
            paths, state["dirty"] = [p for p in state["dirty"] if os.path.exists(p)], set()
        if not paths:
            continue
        try:
            ops = [CommitOperationAdd(path_in_repo=os.path.basename(p), path_or_fileobj=p) for p in paths]
            HfApi(token=hf_token).create_commit(repo_id=USERDATA_REPO, repo_type="dataset", operations=ops,
                                                commit_message="sync user data")
            state["status"] = "ok"
        except Exception as e:
            state["status"] = f"error: {str(e)[:200]}"
            print(f"Hub push error: {e}")
            with state["lock"]:
                state["dirty"].update(paths)  # retry next round


def mark_dirty(path):
    state = _hub_state()
    if not hf_token or not state["can_push"]:
        return
    with state["lock"]:
        state["dirty"].add(path)
    if state["thread"] is None or not state["thread"].is_alive():
        state["thread"] = threading.Thread(target=_push_loop, args=(state,), daemon=True)
        state["thread"].start()


def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)  # atomic: a crash can never leave a half-written file
    mark_dirty(path)


def save_chats(path):
    """Save all chats. The (large) mp3 audio stays in memory only, which keeps saving fast."""
    clean = {cid: {**chat, "messages": [{k: v for k, v in m.items() if k != "audio_base64"} for m in chat["messages"]]}
             for cid, chat in st.session_state.all_chats.items()}
    save_json(path, clean)


# --- 8. DATABASE SETUP FOR ACCOUNTS ---
os.makedirs("user_data", exist_ok=True)
hub_restore()
CRED_FILE = "user_data/credentials.json"

if not os.path.exists(CRED_FILE):
    default_credentials = {"credentials": {"usernames": {}}}
    save_json(CRED_FILE, default_credentials)

with open(CRED_FILE, "r") as f: config = json.load(f)
authenticator = stauth.Authenticate(config['credentials'], "dietician_cookie", "auth_key", cookie_expiry_days=30)

# --- 9. UI & APP ROUTING ---
if not st.session_state.get("authentication_status"):
    st.markdown("<h1 style='text-align: center; color: #2E7D32;'>🥗 Indian-AI Dietician</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: gray;'>Your Personal Indian Dietician</p>", unsafe_allow_html=True)
    st.write("")

    tab1, tab2 = st.tabs(["🔒 Login", "📝 Create New Account"])
    with tab1:
        col_anim, col_form = st.columns([1.2, 1])
        with col_anim:
            if lottie_login_hero: st_lottie(lottie_login_hero, height=400, key="login_hero")
            else: st.markdown("<h1 style='text-align: center; font-size: 100px;'>👩‍⚕️</h1>", unsafe_allow_html=True)
        with col_form:
            st.write("\n\n")
            try: authenticator.login()
            except Exception as e: st.error(f"Login widget error: {e}")
            if st.session_state.get("authentication_status") is False: st.error("Username or password is incorrect")

    with tab2:
        col_anim2, col_form2 = st.columns([1.2, 1])
        with col_anim2:
            if lottie_register_hero: st_lottie(lottie_register_hero, height=400, key="reg_hero")
            else: st.markdown("<h1 style='text-align: center; font-size: 100px;'>📝</h1>", unsafe_allow_html=True)
        with col_form2:
            st.subheader("Join Indian-AI Dietician 🚀")
            new_name = st.text_input("Full Name")
            new_username = st.text_input("Choose a Username").lower()
            new_password = st.text_input("Choose a Password", type="password")

            if st.button("Create My Account", use_container_width=True, type="primary"):
                if new_username in config['credentials']['usernames']: st.error("Username already exists!")
                elif new_username and new_password and new_name:
                    try: hashed_pass = stauth.Hasher([new_password]).generate()[0]
                    except Exception:
                        import bcrypt
                        hashed_pass = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
                    config['credentials']['usernames'][new_username] = {"email": f"{new_username}@user.com", "name": new_name, "password": hashed_pass}
                    save_json(CRED_FILE, config)
                    st.success("✅ Account created successfully! Please click the 'Login' tab to enter.")
                else: st.warning("Please fill out all fields.")

if st.session_state.get("authentication_status"):
    username = st.session_state["username"]
    name = st.session_state["name"]
    history_file = f"user_data/{username}_history.json"

    if "all_chats" not in st.session_state:
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        default_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
                        st.session_state.all_chats = {default_id: {"name": "Previous Chat", "messages": data}}
                    else: st.session_state.all_chats = data
            except Exception:
                st.session_state.all_chats = {datetime.datetime.now().strftime("%Y%m%d%H%M%S"): {"name": "New Chat", "messages": []}}
        else:
            st.session_state.all_chats = {datetime.datetime.now().strftime("%Y%m%d%H%M%S"): {"name": "New Chat", "messages": []}}

        st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]

    # --- SIDEBAR UI ---
    with st.sidebar:
        if lottie_health: st_lottie(lottie_health, height=180, key="health_anim")
        st.write(f"Welcome back, **{name}**! 👋")
        authenticator.logout(location="sidebar")
        st.divider()

        st.header("💬 Your Chats")
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            new_id = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            st.session_state.all_chats[new_id] = {"name": "New Chat", "messages": []}
            st.session_state.current_chat_id = new_id
            save_chats(history_file)
            st.rerun()

        chat_options = list(st.session_state.all_chats.keys())
        chat_names = [st.session_state.all_chats[cid]["name"] for cid in chat_options]
        current_index = chat_options.index(st.session_state.current_chat_id)

        selected_name = st.radio("History:", chat_names, index=current_index, label_visibility="collapsed")
        selected_id = chat_options[chat_names.index(selected_name)]
        if selected_id != st.session_state.current_chat_id:
            st.session_state.current_chat_id = selected_id
            st.rerun()

        st.write("")
        if st.button("🗑️ Delete Chat", use_container_width=True):
            if len(st.session_state.all_chats) > 1:
                del st.session_state.all_chats[st.session_state.current_chat_id]
                st.session_state.current_chat_id = list(st.session_state.all_chats.keys())[-1]
            else:
                st.session_state.all_chats[st.session_state.current_chat_id] = {"name": "New Chat", "messages": []}
            save_chats(history_file)
            st.rerun()

    _kb = get_kb()  # warm up the knowledge base right after login
    if os.getenv("SHOW_DEBUG") == "1":  # developer-only status panel: set SHOW_DEBUG=1 in the Space variables
        with st.sidebar:
            st.divider()
            st.caption(f"📚 Knowledge base: {_kb['count']} passages ({_kb['mode']} search)")
            _hub = _hub_state()["status"]
            if _hub == "ok":
                st.caption("💾 Chat backup: ok")
            else:
                st.warning(f"⚠️ Chat backup problem: {_hub}")
            with st.expander("ℹ️ Knowledge base info"):
                st.json(_kb["info"])

    st.title("🥗 Indian AI Dietician")
    st.caption(f"Personalized Dashboard for {name} | Powered by ICMR/IFCT Data")

    # --- CURRENT CHAT LOGIC ---
    curr_id = st.session_state.current_chat_id
    current_messages = st.session_state.all_chats[curr_id]["messages"]

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("audio_base64"):
                try:
                    audio_bytes = base64.b64decode(msg["audio_base64"])
                    st.audio(audio_bytes, format="audio/mp3")
                except Exception: pass

    # --- INPUT SECTION ---
    st.markdown("---")

    col_mic, col_lang = st.columns([2, 1])
    with col_lang:
        voice_lang = st.selectbox("🗣️ Voice Language", ["Auto-Detect", "Kannada", "Hindi", "English"], index=0)

    lang_code_map = {"Kannada": "kn", "Hindi": "hi", "English": "en"}

    audio_val = None
    with col_mic:
        if hasattr(st, "audio_input"):
            dynamic_mic_key = f"mic_{curr_id}_{len(current_messages)}"
            audio_val = st.audio_input("🎤 Tap to speak your query:", key=dynamic_mic_key)

    text_val = st.chat_input("Ask about a diet (e.g., 'Diet for PCOS', 'Baby food plan')...")

    prompt = None
    if audio_val:
        if not groq_client:
            st.warning("Voice input needs a GROQ_API_KEY. Please type your question instead.")
        else:
            with st.spinner("Transcribing your voice..."):
                try:
                    audio_bytes = audio_val.getvalue()
                    if len(audio_bytes) > 500:
                        file_tuple = ("audio.wav", audio_bytes)
                        kwargs = {
                            "file": file_tuple, "model": "whisper-large-v3",
                            "prompt": "ನಮಸ್ಕಾರ, ಒಂದು ದಿನದ ಡಯಟ್ ಪ್ಲಾನ್ ಕೊಡಿ. Diet plan kodi."
                        }
                        if voice_lang != "Auto-Detect": kwargs["language"] = lang_code_map[voice_lang]
                        transcription = groq_client.audio.transcriptions.create(**kwargs)
                        prompt = transcription.text
                    else: st.warning("Audio was too short or mic is blocked.")
                except Exception as e: st.error(f"Groq API Error: {str(e)}")
    elif text_val: prompt = text_val

    if prompt:
        if st.session_state.all_chats[curr_id]["name"] == "New Chat":
            st.session_state.all_chats[curr_id]["name"] = prompt[:20] + "..."

        current_messages.append({"role": "user", "content": prompt})
        save_chats(history_file)

        with st.chat_message("user"): st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing Database..."):
                # Skip the translation call entirely for plain English input
                if prompt.isascii():
                    query_en = prompt
                else:
                    query_en, q_ok = safe_translate(prompt, 'en')
                    if not q_ok:
                        current_messages.pop()
                        save_chats(history_file)
                        st.warning("⚠️ I couldn't process your question in this language right now. Please try again in a minute, or type it in English.")
                        st.stop()
                query_lower = query_en.lower()

                # --- RETRIEVAL (RAG): fetch only the relevant verified passages ---
                greet_words = {"hi", "hello", "hey", "hii", "hai", "namaste", "start", "greetings"}
                is_greeting = len(query_lower.split()) < 3 and any(w in greet_words for w in re.findall(r"[a-z]+", query_lower))

                retrieval_query = query_en
                if len(query_en.split()) < 6:  # short follow-ups ("yes", "give recipe") inherit the previous topic
                    prev_users = [m.get("content_en", m["content"]) for m in current_messages[:-1] if m["role"] == "user"]
                    if prev_users:
                        retrieval_query = prev_users[-1] + " " + query_en

                retrieved = [] if is_greeting else retrieve(retrieval_query)
                context_text = format_context(retrieved)
                sources_line = build_sources_line(retrieved)

                # --- TASK (what to do) ---
                day_match = re.search(r"\b(\d{1,2})\s*-?\s*days?\b", query_lower)
                num_days = int(day_match.group(1)) if day_match else (7 if "week" in query_lower else 0)
                is_multi = num_days >= 2 or ("plan" in query_lower and "yes" in query_lower)
                repeat_note = ""
                if is_multi:
                    num_days = num_days or 7
                    if num_days > 10:  # keep replies a sensible length
                        num_days, repeat_note = 7, "\nMention that this 7-day plan can be repeated for the longer period they asked for."
                is_plan = any(k in query_lower for k in ['chart', 'plan', 'routine', 'schedule'])
                is_nutri = any(k in query_lower for k in ['calorie', 'protein', 'nutri', 'fat', 'carb'])

                if retrieved and (is_multi or is_plan) and disease_hits(retrieval_query) == []:
                    retrieved = []  # no condition named: don't present loosely related passages as sources
                    context_text = format_context(retrieved)
                    sources_line = ""

                if is_greeting:
                    task = "Greet the user warmly in 1-2 sentences and introduce yourself as the Indian AI Dietician (no personal name). Do not ask any question and do not give diet advice yet; the app adds the follow-up question."
                elif is_multi:
                    task = f"""Create a {num_days}-day Diet Plan for the demographic/condition in the CONTEXT.{repeat_note}
Build it ONLY from the EAT lists and rules in the CONTEXT. Start with a warm, supportive greeting.
FORMAT:
[Warm greeting acknowledging their goal/condition]

**Day 1**
* Breakfast: [Dish] (Portion)
* Lunch: [Dish] (Portion)
* Dinner: [Dish] (Portion)
(Repeat for each of the {num_days} days - the plan must contain exactly {num_days} days)

[Brief encouraging lifestyle tip]"""
                elif is_plan:
                    task = """Create a 1-Day Diet Chart for the demographic/condition in the CONTEXT.
Build it ONLY from the EAT lists and rules in the CONTEXT. Start with a warm, supportive greeting.
FORMAT:
[Warm greeting acknowledging their goal/condition]

* Breakfast: [Dish] (Portion)
* Lunch: [Dish] (Portion)
* Dinner: [Dish] (Portion)

[Brief encouraging lifestyle tip]"""
                elif is_nutri:
                    task = "Give the specific nutritional values requested, in a warm tone, using ONLY the values in the CONTEXT."
                else:
                    task = "Answer the question helpfully using ONLY the CONTEXT, based on the specific condition/demographic. Speak warmly."

                if not is_greeting and not retrieved:
                    if is_multi or is_plan:
                        task += "\nNOTE: The user did not name a health condition or goal. Build a balanced, general healthy Indian plan using ONLY the [RULES] block and common Indian staples, say clearly that no condition-specific guideline was applied, and invite them to share any condition (e.g., diabetes, PCOS, BP) or goal for a personalised plan."
                    else:
                        task += "\nNOTE: No passage in the knowledge base matches this question. Say clearly that you do not have verified guidance on this in your ICMR/NIN database and suggest consulting a doctor or registered dietitian. Do NOT answer from your own knowledge."

                sys_msg = (
                    "You are the Indian AI Dietician: a warm, empathetic, professional AI assistant for Indian diet and nutrition. You have no personal name (never invent one) and you never claim to be a human.\n\n"
                    + GROUNDING_RULES
                    + f"\n\nTASK:\n{task}"
                    + f"\n\nCONTEXT (retrieved from the verified ICMR/NIN knowledge base):\n{context_text}"
                )

                # Keep an English copy of each message so the model never sees translated text
                current_messages[-1]["content_en"] = query_en
                chat_history_for_api = [
                    {"role": m["role"], "content": m.get("content_en", m["content"])}
                    for m in current_messages[-6:]
                ]
                english_ans = generate_response(sys_msg, chat_history_for_api)

                # If every AI backend failed, show a clear error instead of a fake answer
                if not english_ans:
                    # Remove the unanswered user message so history stays clean
                    current_messages.pop()
                    save_chats(history_file)
                    st.error("⚠️ The AI service is busy or unavailable right now. Please try again in a minute.")
                    if os.getenv("SHOW_DEBUG") == "1":
                        with st.expander("Technical details (for the developer)"):
                            for err in st.session_state.get("llm_errors", []):
                                st.code(err)
                    st.stop()

                english_ans = clean_output(english_ans)
                if is_greeting or (retrieved and "?" not in english_ans[-250:]):
                    english_ans = get_follow_up(query_lower, english_ans)

                # --- LANGUAGE CONSISTENCY & TRANSLATION ---
                target_lang = 'en'
                if voice_lang == "Kannada": target_lang = 'kn'
                elif voice_lang == "Hindi": target_lang = 'hi'
                elif voice_lang == "English": target_lang = 'en'
                else:
                    if any('\u0900' <= c <= '\u097F' for c in prompt): target_lang = 'hi'
                    elif any('\u0C80' <= c <= '\u0CFF' for c in prompt): target_lang = 'kn'
                    elif any('\u0C00' <= c <= '\u0C7F' for c in prompt): target_lang = 'te'

                # safe_translate never raises. If every translator fails, show clean English
                # (never a half-translated mix) and say why.
                final_lang = target_lang
                if target_lang != 'en':
                    final_ans, ok = safe_translate(english_ans, target_lang, source='en')
                    if not ok:
                        final_ans = english_ans + "\n\n_(Translation is temporarily unavailable, so this reply is in English.)_"
                        final_lang = 'en'
                else:
                    final_ans = english_ans

                display_ans = final_ans + (f"\n\n{sources_line}" if sources_line else "")
                st.markdown(display_ans)
                if retrieved:
                    with st.expander("🔎 Guideline passages used for this answer"):
                        for c in retrieved:
                            st.markdown(f"**{c['title']}** - _{c['source']}_")
                            st.text(c["text"])

                audio_file = generate_tts_audio(final_ans, final_lang)
                audio_b64 = None

                if audio_file:
                    audio_bytes_tts = audio_file.getvalue()
                    st.audio(audio_bytes_tts, format="audio/mp3")
                    audio_b64 = base64.b64encode(audio_bytes_tts).decode('utf-8')

                current_messages.append({"role": "assistant", "content": display_ans, "content_en": english_ans, "audio_base64": audio_b64})
                save_chats(history_file)
