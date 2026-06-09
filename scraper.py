import html
import json
import re
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

DDRAGON_BASE = "https://ddragon.leagueoflegends.com"
UNIVERSE_ROOT = "https://universe-meeps.leagueoflegends.com/v1/en_us"
UNIVERSE_BASE = f"{UNIVERSE_ROOT}/champions"
LOCALE = "en_US"
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "data"
REQUEST_DELAY = 0.1
MAX_CHAMPIONS = None
SCRAPE_UNIVERSE = True
HEADERS = {
    "User-Agent": ""
}

STAT_LABELS = [
    ("attackrange", "Attack range"),
    ("movespeed", "Move speed"),
]


def get_json(url):
    response = requests.get(url, headers=HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def get_latest_version():
    return get_json(f"{DDRAGON_BASE}/api/versions.json")[0]


def get_all_champions(version):
    url = f"{DDRAGON_BASE}/cdn/{version}/data/{LOCALE}/champion.json"
    return get_json(url)["data"]


def get_champion_detail(version, champion_id):
    url = f"{DDRAGON_BASE}/cdn/{version}/data/{LOCALE}/champion/{champion_id}.json"
    return get_json(url)["data"][champion_id]


def clean_text(raw):
    if not raw:
        return ""
    unescaped = html.unescape(raw)
    without_tags = BeautifulSoup(unescaped, "html.parser").get_text(separator=" ")
    return re.sub(r"\s+", " ", without_tags).strip()


def clean_html_paragraphs(raw_html):
    if not raw_html:
        return ""
    soup = BeautifulSoup(html.unescape(raw_html), "html.parser")
    paragraphs = [re.sub(r"\s+", " ", p.get_text(separator=" ")).strip()
                  for p in soup.find_all("p")]
    paragraphs = [p for p in paragraphs if p]
    if not paragraphs:
        return clean_text(raw_html)
    return "\n\n".join(paragraphs)


def universe_slug(name):
    return re.sub(r"[^a-z0-9]", "", name.lower())


def get_universe_slugs():
    data = get_json(f"{UNIVERSE_ROOT}/champion-browse/index.json")
    return {champ["name"]: champ["slug"] for champ in data.get("champions", [])}


def scrape_universe_lore(slug):
    url = f"{UNIVERSE_BASE}/{slug}/index.json"
    response = requests.get(url, headers=HEADERS, timeout=30)
    if response.status_code != 200:
        return ""
    biography = response.json().get("champion", {}).get("biography", {})
    return clean_html_paragraphs(biography.get("full", "") or biography.get("short", ""))


def meaningful_range(range_burn):
    if not range_burn or range_burn == "self":
        return None
    values = range_burn.split("/")
    if all(v.strip() in ("25000", "0") for v in values):
        return None
    return range_burn


def build_ddragon_document(detail):
    name = detail["name"]
    title = detail.get("title", "")
    lore = clean_text(detail.get("lore", "")) or clean_text(detail.get("blurb", ""))
    roles = ", ".join(detail.get("tags", []))
    resource = detail.get("partype", "")

    lines = [f"# {name} — {title}", "", f"Roles: {roles}"]
    if resource:
        lines.append(f"Resource: {resource}")
    lines += ["", "## Lore", lore, ""]

    stats = detail.get("stats", {})
    if stats:
        lines.append("## Base Stats")
        for key, label in STAT_LABELS:
            if key in stats:
                lines.append(f"- {label}: {stats[key]}")
        lines.append("")

    passive = detail.get("passive", {})
    if passive:
        lines += [f"## Passive — {passive.get('name', '')}",
                  clean_text(passive.get("description", "")), ""]

    for key, spell in zip(["Q", "W", "E", "R"], detail.get("spells", [])):
        lines.append(f"## {key} — {spell.get('name', '')}")
        lines.append(clean_text(spell.get("description", "")))
        range_burn = meaningful_range(spell.get("rangeBurn"))
        if range_burn:
            lines.append(f"Range: {range_burn}")
        if spell.get("cooldownBurn"):
            lines.append(f"Cooldown: {spell['cooldownBurn']}")
        if spell.get("costBurn"):
            lines.append(f"Cost: {spell['costBurn']}")
        lines.append("")

    ally_tips = detail.get("allytips", [])
    if ally_tips:
        lines.append("## Tips (playing as)")
        lines += [f"- {clean_text(tip)}" for tip in ally_tips]
        lines.append("")

    enemy_tips = detail.get("enemytips", [])
    if enemy_tips:
        lines.append("## Tips (playing against)")
        lines += [f"- {clean_text(tip)}" for tip in enemy_tips]
        lines.append("")

    return "\n".join(lines).strip()


def safe_filename(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")


def save_champion(name, ddragon_text):
    path = OUTPUT_DIR / f"{safe_filename(name)}.txt"
    path.write_text(ddragon_text, encoding="utf-8")
    if not path.exists():
        raise IOError(f"write reported success but file is missing: {path}")
    return path


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print(f"Saving documents to: {OUTPUT_DIR}")
    version = get_latest_version()
    print(f"Latest Data Dragon version: {version}")

    champions = get_all_champions(version)
    items = sorted(champions.items())
    if MAX_CHAMPIONS:
        items = items[:MAX_CHAMPIONS]
    print(f"Processing {len(items)} of {len(champions)} champions")

    universe_slugs = {}
    if SCRAPE_UNIVERSE:
        try:
            universe_slugs = get_universe_slugs()
            print(f"Loaded {len(universe_slugs)} Universe slugs")
        except Exception as error:
            print(f"  universe slug list failed, using derived slugs: {error}")

    index = []
    for position, (champion_id, summary) in enumerate(items, start=1):
        name = summary["name"]
        print(f"[{position}/{len(items)}] {name}")

        try:
            detail = get_champion_detail(version, champion_id)
            ddragon_text = build_ddragon_document(detail)
        except Exception as error:
            print(f"  ddragon failed: {error}")
            continue
        time.sleep(REQUEST_DELAY)

        universe_text = ""
        if SCRAPE_UNIVERSE:
            slug = universe_slugs.get(name) or universe_slug(name)
            try:
                universe_text = scrape_universe_lore(slug)
                if universe_text:
                    print(f"  universe lore: {len(universe_text)} chars")
                else:
                    print("  universe lore: none found (slug may differ)")
            except Exception as error:
                print(f"  universe failed: {error}")
            time.sleep(REQUEST_DELAY)

        document = ddragon_text
        if universe_text:
            document += "\n\n## Full lore (Universe)\n" + universe_text
        path = save_champion(name, document)
        size = path.stat().st_size
        print(f"  saved {path} ({size} bytes)")
        index.append({"champion": name, "id": champion_id, "file": path.name})

    index_path = OUTPUT_DIR / "index.json"
    index_path.write_text(
        json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"  saved {index_path} ({index_path.stat().st_size} bytes)")
    print(f"Done. Saved {len(index)} documents + index.json to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
