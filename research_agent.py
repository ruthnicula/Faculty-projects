# -*- coding: utf-8 -*-
"""
Research Agent (stiri recente in romana) - GRATIS
- Cauta stiri recente folosind Google News RSS (free, fara API key)
- Filtreaza pe cuvinte-cheie
- Face rezumat in romana cu Ollama local (free/offline, dupa ce ai tras un model)

Cerinte:
1) Python 3.9+
2) Ollama instalat si pornit: https://ollama.com
3) Un model mic pentru PC modest (recomandat):
   - ollama pull llama3.2:3b
   sau
   - ollama pull qwen2.5:3b

Instalare pachete:
   pip install requests feedparser

Rulare:
   python research_agent_ro_news.py
"""

import re
import time
import json
import requests
import feedparser
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

# =========================
# CONFIG
# =========================

# Model Ollama
OLLAMA_MODEL = "llama3.2:3b" 

# Cate rezultate sa ia din RSS inainte de filtrare
MAX_RSS_ITEMS = 30

# Cate stiri finale sa afiseze
TOP_K = 8

# Cat de "recente" sa fie (in ore). Google News RSS e deja recent, asta e un filtru suplimentar.
RECENCY_HOURS = 72

# Timeout pentru request-uri
HTTP_TIMEOUT = 15


# =========================
# 1) GOOGLE NEWS RSS (gratis)
# =========================

def build_google_news_rss_url(query: str) -> str:
    """
    Google News RSS pentru cautari in romana, Romania.
    Exemple:
      https://news.google.com/rss/search?q=energie&hl=ro&gl=RO&ceid=RO:ro
    """
    # Inlocuieste spatii cu +
    q = query.strip()
    q = re.sub(r"\s+", "+", q)
    return f"https://news.google.com/rss/search?q={q}&hl=ro&gl=RO&ceid=RO:ro"


def fetch_rss(url: str) -> List[Dict[str, Any]]:
    """
    Descarca RSS si returneaza o lista de item-uri standardizate.
    """
    feed = feedparser.parse(url)
    items = []

    for e in feed.entries[:MAX_RSS_ITEMS]:
        title = getattr(e, "title", "").strip()
        link = getattr(e, "link", "").strip()
        summary = getattr(e, "summary", "").strip()

        # published_parsed e de obicei disponibil
        published_dt = None
        if getattr(e, "published_parsed", None):
            published_dt = datetime.fromtimestamp(time.mktime(e.published_parsed), tz=timezone.utc)

        source = ""
        # Google News RSS pune adesea sursa in title (ex: "Titlu - Sursa")
        if " - " in title:
            possible_source = title.split(" - ", 1)[-1].strip()
            if len(possible_source) <= 60:
                source = possible_source

        items.append({
            "title": title,
            "link": link,
            "summary": clean_html(summary),
            "published_utc": published_dt.isoformat() if published_dt else None,
            "source": source
        })

    return items


def clean_html(text: str) -> str:
    # curata tag-uri simple
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================
# 2) FILTRARE + SCOR
# =========================

def is_recent(published_utc: Optional[str], hours: int) -> bool:
    if not published_utc:
        return True  # daca nu stim data, nu excludem
    try:
        dt = datetime.fromisoformat(published_utc.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        delta_hours = (now - dt).total_seconds() / 3600
        return delta_hours <= hours
    except Exception:
        return True


def keyword_score(item: Dict[str, Any], query: str) -> int:
    """
    Scor simplu pe baza aparitiei cuvintelor din query in titlu/rezumat.
    """
    q_words = [w.lower() for w in re.findall(r"[A-Za-zĂÂÎȘȚăâîșț0-9]+", query) if len(w) >= 3]
    hay = f"{item.get('title','')} {item.get('summary','')}".lower()

    score = 0
    for w in q_words:
        if w in hay:
            score += 2
    # bonus pt titlu
    title_lower = item.get("title", "").lower()
    for w in q_words:
        if w in title_lower:
            score += 2
    return score


def dedupe_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Dedupe pe link + titlu.
    """
    seen = set()
    out = []
    for it in items:
        key = (it.get("link", ""), it.get("title", ""))
        if key in seen:
            continue
        seen.add(key)
        out.append(it)
    return out


def select_top_news(items: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
    # Filtru recenta
    filtered = [it for it in items if is_recent(it.get("published_utc"), RECENCY_HOURS)]
    filtered = dedupe_items(filtered)

    # Scorare
    for it in filtered:
        it["score"] = keyword_score(it, query)

    # Sorteaza: score desc + data desc
    def sort_key(it):
        published = it.get("published_utc")
        # daca lipseste, pune foarte vechi
        ts = 0
        if published:
            try:
                ts = datetime.fromisoformat(published.replace("Z", "+00:00")).timestamp()
            except Exception:
                ts = 0
        return (it.get("score", 0), ts)

    filtered.sort(key=sort_key, reverse=True)

    # Pastreaza top_k, dar daca scor=0 la toate, tot luam primele top_k (stiri "generale" pe tema)
    return filtered[:top_k]


# =========================
# 3) REZUMAT CU OLLAMA (local)
# =========================

def ollama_chat(messages: List[Dict[str, str]], model: str = OLLAMA_MODEL, temperature: float = 0.2) -> str:
    """
    Apeleaza Ollama local via HTTP.
    Necesita ollama sa ruleze local (default: http://localhost:11434)
    """
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature
        }
    }
    r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data["message"]["content"]


def summarize_news_with_ollama(query: str, items: List[Dict[str, Any]]) -> str:
    """
    Rezuma intr-un format util: top stiri + idei cheie + trenduri.
    Rezumatul se bazeaza pe titlu+snippet (nu pe full article).
    """
    # Pregateste context compact
    lines = []
    for i, it in enumerate(items, 1):
        pub = it.get("published_utc") or "necunoscut"
        src = it.get("source") or "sursa necunoscuta"
        title = it.get("title", "")
        summ = it.get("summary", "")
        link = it.get("link", "")
        lines.append(f"{i}. [{src}] ({pub})\n   Titlu: {title}\n   Snippet: {summ}\n   Link: {link}\n")

    context = "\n".join(lines)

    system = (
        "Esti un research assistant care rezuma stiri RECENTE in romana.\n"
        "Reguli:\n"
        "- Foloseste doar informatiile din lista de stiri primita (titlu+snippet).\n"
        "- Nu inventa detalii care nu apar in snippet.\n"
        "- Structureaza clar: Top stiri, Idei-cheie, Trenduri/observatii.\n"
        "- Pastreaza linkurile.\n"
    )

    user = (
        f"Query utilizator: {query}\n\n"
        "Mai jos ai stiri (titlu+snippet+link). Fa un rezumat util:\n"
        "- 5-8 bullet points pentru 'Top stiri' (fiecare cu link)\n"
        "- 3-5 idei-cheie (ce se repeta)\n"
        "- 2-3 posibile implicatii (daca se pot deduce prudent, fara speculatii mari)\n\n"
        f"STIRI:\n{context}"
    )

    return ollama_chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        model=OLLAMA_MODEL
    )


# =========================
# 4) CLI AGENT
# =========================

def research_news_agent_loop():
    print("=== Research Agent: Stiri recente in romana (Google News RSS + Ollama) ===")
    print("Scrie un subiect (ex: 'energie Romania', 'inflatie', 'AI', 'Netflix'), sau 'exit'.")

    while True:
        query = input("\nSubiect: ").strip()
        if not query:
            continue
        if query.lower() in ("exit", "quit", "q"):
            break

        rss_url = build_google_news_rss_url(query)
        print(f"\n[1/3] Caut RSS: {rss_url}")

        try:
            items = fetch_rss(rss_url)
        except Exception as e:
            print(f"Eroare la fetch RSS: {e}")
            continue

        if not items:
            print("Nu am gasit rezultate in RSS.")
            continue

        top_items = select_top_news(items, query=query, top_k=TOP_K)

        print(f"[2/3] Am selectat {len(top_items)} stiri relevante/recente.\n")
        for i, it in enumerate(top_items, 1):
            print(f"{i}. {it.get('title','(fara titlu)')}")
            print(f"   Link: {it.get('link','')}")
            if it.get("published_utc"):
                print(f"   Data(UTC): {it.get('published_utc')}")
            if it.get("source"):
                print(f"   Sursa: {it.get('source')}")
            print()

        print("[3/3] Generez rezumat cu Ollama local...")
        try:
            summary = summarize_news_with_ollama(query, top_items)
            print("\n=== REZUMAT ===\n")
            print(summary)
        except requests.exceptions.ConnectionError:
            print(
                "\nNu pot conecta la Ollama (http://localhost:11434).\n"
                "Asigura-te ca Ollama este pornit.\n"
                "Test rapid: in terminal -> ollama run llama3.2:3b\n"
            )
        except Exception as e:
            print(f"Eroare la rezumat: {e}")


if __name__ == "__main__":
    research_news_agent_loop()
