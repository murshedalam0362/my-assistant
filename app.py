import streamlit as st
import os, json, glob, requests, urllib.parse
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="আমার অ্যাসিস্ট্যান্ট", page_icon="📚")

DATA = "data"
BOOKS = f"{DATA}/books"
FAQ_FILE = f"{DATA}/faq.json"
CFG_FILE = f"{DATA}/config.json"
MODEL = "gemini-2.5-flash"
os.makedirs(BOOKS, exist_ok=True)

BLOCKED = ["নবী", "রাসূল", "রাসুল", "সাহাবী", "সাহাবি", "মুহাম্মদ", "nabi", "rasul",
           "rasool", "prophet", "muhammad", "mohammad", "sahaba", "companion of"]


def load_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def gemini(prompt, system=None):
    key = st.secrets["GEMINI_API_KEY"]
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL}:generateContent?key={key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    try:
        r = requests.post(url, json=body, timeout=90)
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        detail = ""
        try:
            detail = str(data.get("error", data))[:300]
        except Exception:
            detail = str(e)[:300]
        if st.session_state.get("admin"):
            return f"⚠️ এরর (শুধু অ্যাডমিন দেখছেন): {detail}"
        return "⚠️ এখন উত্তর দেওয়া যাচ্ছে না। একটু পরে আবার চেষ্টা করুন।"


@st.cache_data
def load_chunks(sig):
    out = []
    for f in sorted(glob.glob(f"{BOOKS}/*")):
        name = os.path.basename(f)
        try:
            if f.lower().endswith(".pdf"):
                for i, p in enumerate(PdfReader(f).pages, 1):
                    t = (p.extract_text() or "").strip()
                    if t:
                        out.append((name, i, t[:3000]))
            elif f.lower().endswith(".txt"):
                t = open(f, encoding="utf-8", errors="ignore").read()
                for i in range(0, len(t), 1500):
                    out.append((name, i // 1500 + 1, t[i:i + 1500]))
        except Exception:
            pass
    return out


def book_sig():
    return tuple((f, os.path.getmtime(f)) for f in sorted(glob.glob(f"{BOOKS}/*")))


def best_matches(query, texts, k=4, threshold=0.05):
    if not texts:
        return []
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    m = v.fit_transform(texts + [query])
    s = cosine_similarity(m[-1], m[:-1])[0]
    idx = s.argsort()[::-1][:k]
    return [(i, s[i]) for i in idx if s[i] > threshold]


SYSTEM_BOOK = (
    "তুমি একটি বই-ভিত্তিক সহকারী। শুধুমাত্র নিচে দেওয়া অংশগুলো থেকে উত্তর দেবে। "
    "নিজের জ্ঞান থেকে কিছু যোগ করবে না, কোনো রেফারেন্স বানাবে না। "
    "উত্তরের শেষে প্রতিটি তথ্যের জন্য [বইয়ের নাম, পৃষ্ঠা] দেখাবে এবং মূল লাইনটি উদ্ধৃত করবে। "
    "অংশগুলোতে উত্তর না থাকলে শুধু বলবে: 'আমার কাছে থাকা বইগুলোতে এর উত্তর পাওয়া যায়নি।' "
    "ব্যবহারকারীর ভাষায় উত্তর দেবে।"
)


def answer_from_books(q):
    faq = load_json(FAQ_FILE, [])
    if faq:
        hits = best_matches(q, [x["q"] for x in faq], k=1, threshold=0.6)
        if hits:
            return faq[hits[0][0]]["a"]
    chunks = load_chunks(book_sig())
    hits = best_matches(q, [c[2] for c in chunks], k=4)
    if not hits:
        return "আমার কাছে থাকা বইগুলোতে এর উত্তর পাওয়া যায়নি।"
    ctx = "\n\n".join(f"[বই: {chunks[i][0]}, পৃষ্ঠা: {chunks[i][1]}]\n{chunks[i][2]}" for i, _ in hits)
    return gemini(f"অংশসমূহ:\n{ctx}\n\nপ্রশ্ন: {q}", SYSTEM_BOOK)


def chat_box(key, fn):
    hist = st.session_state.setdefault(key, [])
    for role, text in hist:
        st.chat_message(role).write(text)
    q = st.chat_input("এখানে প্রশ্ন লিখুন...", key=key + "_in")
    if q:
        st.chat_message("user").write(q)
        with st.spinner("খুঁজছি..."):
            a = fn(q)
        st.chat_message("assistant").write(a)
        hist += [("user", q), ("assistant", a)]


# ---------------- sidebar: admin login ----------------
cfg = load_json(CFG_FILE, {"general_chat": True, "image_gen": True})
is_admin = st.session_state.get("admin", False)
with st.sidebar:
    st.header("🔐 অ্যাডমিন")
    if not is_admin:
        pw = st.text_input("পাসওয়ার্ড", type="password")
        if st.button("লগইন"):
            if pw and pw == st.secrets.get("ADMIN_PASSWORD"):
                st.session_state["admin"] = True
                st.rerun()
            else:
                st.error("ভুল পাসওয়ার্ড")
    else:
        st.success("লগইন করা আছে")
        if st.button("লগআউট"):
            st.session_state["admin"] = False
            st.rerun()

# ---------------- tabs ----------------
names = ["📚 বই থেকে প্রশ্ন"]
if cfg["general_chat"]:
    names.append("➕ সাধারণ চ্যাট")
if cfg["image_gen"]:
    names.append("🖼️ ছবি বানান")
if is_admin:
    names.append("⚙️ অ্যাডমিন প্যানেল")
tabs = dict(zip(names, st.tabs(names)))

with tabs["📚 বই থেকে প্রশ্ন"]:
    st.caption("শুধু আমার সংরক্ষিত বই ও প্রশ্ন-উত্তর থেকে রেফারেন্সসহ উত্তর দেওয়া হবে।")
    chat_box("book_chat", answer_from_books)

if "➕ সাধারণ চ্যাট" in tabs:
    with tabs["➕ সাধারণ চ্যাট"]:
        st.caption("এখানে যেকোনো সাধারণ প্রশ্ন করা যাবে। (দ্বীনি মাসআলার জন্য বই-ট্যাব বা আলেমের কাছে যান)")
        chat_box("gen_chat", lambda q: gemini(q, "তুমি একজন সহায়ক সহকারী। ব্যবহারকারীর ভাষায় উত্তর দাও।"))

if "🖼️ ছবি বানান" in tabs:
    with tabs["🖼️ ছবি বানান"]:
        st.caption("নবী-রাসূল ও সাহাবীদের ছবি বানানো বন্ধ রাখা হয়েছে।")
        prompt = st.text_input("কেমন ছবি চান? (ইংরেজিতে লিখলে ভালো হয়)")
        if st.button("ছবি বানাও") and prompt:
            if any(w in prompt.lower() for w in BLOCKED):
                st.warning("দুঃখিত, এই ধরনের ছবি বানানো হয় না। মসজিদ, প্রকৃতি বা ক্যালিগ্রাফি চেষ্টা করুন।")
            else:
                with st.spinner("ছবি তৈরি হচ্ছে... (১ মিনিট পর্যন্ত লাগতে পারে)"):
                    img, err = None, ""
                    url = ("https://image.pollinations.ai/prompt/" + urllib.parse.quote(prompt)
                           + "?width=768&height=768&nologo=true&model=flux")
                    for _ in range(2):
                        try:
                            r = requests.get(url, timeout=120, headers={"User-Agent": "Mozilla/5.0"})
                            if r.status_code == 200 and r.headers.get("content-type", "").startswith("image"):
                                img = r.content
                                break
                            err = f"status {r.status_code}: {r.text[:200]}"
                        except Exception as e:
                            err = str(e)[:200]
                if img:
                    st.image(img)
                    st.download_button("⬇️ ডাউনলোড", img, "image.jpg", "image/jpeg")
                else:
                    st.error("ছবি বানানো যায়নি, একটু পরে আবার চেষ্টা করুন।")
                    if is_admin:
                        st.code(err or "unknown error")

if is_admin:
    with tabs["⚙️ অ্যাডমিন প্যানেল"]:
        st.subheader("📖 বই আপলোড (PDF বা TXT)")
        ups = st.file_uploader("বই বাছাই করুন", type=["pdf", "txt"], accept_multiple_files=True)
        if ups and st.button("সেভ করুন"):
            for u in ups:
                with open(f"{BOOKS}/{u.name}", "wb") as f:
                    f.write(u.getbuffer())
            st.success("বই সেভ হয়েছে")
        for f in sorted(glob.glob(f"{BOOKS}/*")):
            c1, c2 = st.columns([4, 1])
            c1.write(os.path.basename(f))
            if c2.button("🗑️", key=f):
                os.remove(f)
                st.rerun()

        st.divider()
        st.subheader("❓ নিজের প্রশ্ন-উত্তর (FAQ)")
        fq = st.text_input("প্রশ্ন")
        fa = st.text_area("উত্তর (রেফারেন্সসহ)")
        faq = load_json(FAQ_FILE, [])
        if st.button("FAQ যোগ করুন") and fq and fa:
            faq.append({"q": fq, "a": fa})
            save_json(FAQ_FILE, faq)
            st.success("যোগ হয়েছে")
            st.rerun()
        for i, x in enumerate(faq):
            c1, c2 = st.columns([4, 1])
            c1.write(f"**{x['q']}**")
            if c2.button("🗑️", key=f"faq{i}"):
                faq.pop(i)
                save_json(FAQ_FILE, faq)
                st.rerun()

        st.divider()
        st.subheader("🔧 সেটিংস")
        g = st.toggle("➕ সাধারণ চ্যাট চালু", cfg["general_chat"])
        im = st.toggle("🖼️ ছবি বানানো চালু", cfg["image_gen"])
        if (g, im) != (cfg["general_chat"], cfg["image_gen"]):
            save_json(CFG_FILE, {"general_chat": g, "image_gen": im})
            st.rerun()
