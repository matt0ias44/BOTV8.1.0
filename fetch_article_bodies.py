#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Télécharge et nettoie les articles liés aux news (via l’URL).
Utilise trafilatura comme parser principal et readability en fallback.
"""

import os
import time
import requests
import pandas as pd

try:
    import trafilatura
except ImportError:
    trafilatura = None

try:
    from readability import Document
    from lxml import html
except ImportError:
    Document = None
    html = None


def fetch_article_body(url: str, timeout: int = 10) -> str:
    """Récupère et nettoie un article complet depuis son URL. Retourne le texte (string) ou "" si échec."""
    if not url or not isinstance(url, str):
        return ""

    try:
        resp = requests.get(url, timeout=timeout, headers={
            "User-Agent": "Mozilla/5.0 (compatible; CryptoNewsBot/1.0)"
        })
        if resp.status_code != 200 or not resp.text:
            return ""
        html_content = resp.text

        # 1. Essayer trafilatura
        if trafilatura:
            text = trafilatura.extract(html_content, include_comments=False, include_tables=False)
            if text and len(text) > 100:
                return text.strip()

        # 2. Fallback readability
        if Document and html:
            doc = Document(html_content)
            parsed = html.fromstring(doc.summary())
            text = " ".join(parsed.itertext())
            if text and len(text) > 100:
                return text.strip()

        return ""
    except Exception as e:
        print(f"[fetch_article_body] Erreur sur {url}: {e}")
        return ""


def fetch_article_bodies(csv_in: str, csv_out: str, url_col: str = "url") -> pd.DataFrame:
    """Lit un CSV (avec au moins une colonne `url`), télécharge les articles et ajoute une colonne `body`."""
    if not os.path.exists(csv_in):
        raise FileNotFoundError(f"{csv_in} introuvable")

    df = pd.read_csv(csv_in)
    if url_col not in df.columns:
        raise ValueError(f"Colonne {url_col} absente du CSV")

    bodies = []
    for i, row in df.iterrows():
        url = row.get(url_col, "")
        text = fetch_article_body(url)
        bodies.append(text)
        print(f"[{i+1}/{len(df)}] {url} -> {len(text)} caractères")
        time.sleep(1)  # éviter de spammer

    df["body"] = bodies
    df.to_csv(csv_out, index=False)
    print(f"[OK] Articles sauvegardés dans {csv_out}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True, help="CSV input avec colonne 'url'")
    parser.add_argument("--outfile", required=True, help="CSV output avec colonne 'body'")
    args = parser.parse_args()

    fetch_article_bodies(args.infile, args.outfile)
