import json
from pathlib import Path
from typing import List, Dict, Optional
import requests
from bs4 import BeautifulSoup as BS


BASE_URL = "https://www.statsf1.com"
URL_TEMPLATE = BASE_URL + "/fr/{}.aspx"
DEFAULT_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/140.0.0.0 Safari/537.36"


def fetch_page(url: str, ua: Optional[str] = None) -> str:
    """Fetch page HTML. If `ua` is provided, send it as User-Agent header; otherwise rely on requests default."""
    headers = {"User-Agent": ua} if ua else None
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    return resp.text


def check_robots(base_url: str = BASE_URL, user_agent: Optional[str] = None, target_path: str = "/fr/") -> Dict[str, Optional[str]]:
    """Check robots.txt for the provided user_agent and target_path.

    Returns dict {"ok": bool, "crawl_delay": str|None, "reason": str|None}

    Logic (simplified / conservative):
    - Parse robots.txt into groups by User-agent.
    - If there is a group matching the exact user_agent token, use its Disallow rules.
    - Else if there is a group for '*', use its Disallow rules.
    - Else allow by default.
    - A path is disallowed if any Disallow entry is a non-empty prefix of the target_path (prefix match).
    """
    robots_url = base_url.rstrip("/") + "/robots.txt"
    try:
        resp = requests.get(robots_url, timeout=8)
        if resp.status_code != 200:
            return {"ok": True, "crawl_delay": None, "reason": "no robots.txt or unreachable"}
        text = resp.text

        groups: List[Dict] = []
        group: Dict = {"agents": [], "disallow": [], "crawl-delay": None}

        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(":", 1)
            if len(parts) != 2:
                continue
            key = parts[0].strip().lower()
            val = parts[1].strip()
            if key == "user-agent":
                # start a new group if current group already has agents
                if group["agents"]:
                    groups.append(group)
                    group = {"agents": [], "disallow": [], "crawl-delay": None}
                group["agents"].append(val.lower())
            elif key == "disallow":
                group["disallow"].append(val)
            elif key == "crawl-delay":
                group["crawl-delay"] = val

        if group and group["agents"]:
            groups.append(group)

        ua = (user_agent or "").lower()

        # find exact match group
        target_group = None
        for g in groups:
            for a in g["agents"]:
                if ua and a == ua:
                    target_group = g
                    break
            if target_group:
                break

        # fallback to '*' group
        if not target_group:
            for g in groups:
                if "*" in g["agents"]:
                    target_group = g
                    break

        # if still no group, allow
        if not target_group:
            return {"ok": True, "crawl_delay": None, "reason": "no matching group"}

        disallows = [d for d in target_group.get("disallow", []) if d is not None]
        # empty disallow entry means allow all (spec says empty means allow everything)
        # treat non-empty entries as prefixes
        for d in disallows:
            if d == "":
                continue
            dp = d if d.startswith("/") else "/" + d
            # prefix match: if the disallow path is a prefix of target_path => blocked
            if target_path.startswith(dp):
                return {"ok": False, "crawl_delay": target_group.get("crawl-delay"), "reason": f"disallowed by {d}"}

        return {"ok": True, "crawl_delay": target_group.get("crawl-delay"), "reason": None}
    except Exception as e:
        return {"ok": True, "crawl_delay": None, "reason": f"error:{e}"}


def _get_text_or_none(el) -> Optional[str]:
    return el.get_text(" ", strip=True) if el else None


def parse_gpsaison(soup: BS) -> List[Dict]:
    section = soup.find(id="ctl00_CPH_Main_P_GPSaison")
    if not section:
        return []
    results = []
    for tr in section.find_all("tr"):
        tds = tr.find_all(["td", "th"])
        if not tds:
            continue
        a = tr.find("a")
        name = _get_text_or_none(a) if a else None
        href = a.get("href") if a and a.has_attr("href") else None
        cols = [td.get_text(" ", strip=True) for td in tds]
        date_text = None
        for c in cols:
            if any(ch.isdigit() for ch in c):
                date_text = c
                break
        results.append({"gp_name": name, "link": href, "date_text": date_text, "cols": cols})
    if results:
        return results
    for item in section.find_all(["li", "p", "div"], recursive=False):
        text = item.get_text(" ", strip=True)
        a = item.find("a")
        name = _get_text_or_none(a) if a else None
        href = a.get("href") if a and a.has_attr("href") else None
        results.append({"gp_name": name or text, "link": href, "date_text": text, "cols": [text]})
    if results:
        return results
    for a in section.find_all("a"):
        results.append({"gp_name": _get_text_or_none(a), "link": a.get("href"), "date_text": None, "cols": []})
    return results


def parse_champ_table(soup: BS, table_id: str) -> Dict:
    tbl = soup.find(id=table_id)
    if not tbl:
        return {"manches": [], "rows": []}

    manches: List[str] = []
    rows: List[Dict] = []

    # Find header row: prefer a row that contains cells with class 'manche' or 'titre'.
    header_row = None
    for tr in tbl.find_all("tr"):
        if tr.find(class_="manche") or tr.find(class_="titre"):
            header_row = tr
            break
        # fallback: detect a non-data row where second cell is not a digit
        cells = tr.find_all(["th", "td"])
        if not cells:
            continue
        texts = [_get_text_or_none(c) or "" for c in cells]
        if len(texts) >= 2 and not texts[1].strip().isdigit():
            header_row = tr
            break

    if header_row:
        # Prefer explicit 'manche' class cells (robust for this site's markup)
        manche_cells = header_row.find_all(class_="manche")
        if manche_cells:
            manches = [_get_text_or_none(c) for c in manche_cells]
        else:
            # Fallback: try to infer manche columns from header texts.
            header_cells = header_row.find_all(["th", "td"])
            texts = [_get_text_or_none(c) for c in header_cells]
            # Attempt to align with a data row to compute the correct slice
            first_data_tds = None
            for tr in tbl.find_all("tr"):
                tds = tr.find_all("td")
                if tds and len(tds) > 2:
                    first_data_tds = tds
                    break
            if first_data_tds:
                total_cols = len(first_data_tds)
                # data rows layout: pos(1) + name(1) + manches(n) + total(1)
                manche_count = max(0, total_cols - 3)
                if manche_count > 0 and len(texts) >= (2 + manche_count + 1):
                    manches = texts[2:2 + manche_count]
                else:
                    manches = texts[2:-1] if len(texts) >= 3 else []
            else:
                manches = texts[2:-1] if len(texts) >= 3 else []

    # Parse data rows
    for tr in tbl.find_all("tr"):
        tds = tr.find_all("td")
        if not tds:
            continue
        cells = [_get_text_or_none(td) for td in tds]
        if len(cells) < 2:
            continue
        pos = cells[0]
        name = cells[1]
        if len(cells) >= 3:
            results = cells[2:-1] if len(cells) > 3 else []
            total = cells[-1]
        else:
            results = []
            total = ""
        rows.append({"pos": pos, "name": name, "results": results, "total": total})

    return {"manches": manches, "rows": rows}


def parse_grid(html_or_soup) -> List[Dict]:
    """Parse the starting grid table with id `ctl00_CPH_Main_TBL_Grille`.

    Returns a list of dicts: [{"position": int, "driver": str, "constructor": str, "engine": str}, ...]

    The statsf1 grid is laid out as many <div id="GrdX"> elements containing markup like:
      1. <a ...><span class="CurDriver">L.NORRIS</span></a><br>
      <a ...><span class="CurChpConstructor">McLaren</span></a>&nbsp;<a ...><span class="CurEngine">Mercedes</span></a>

    The function is tolerant to class name variants (CurDriver, CurChpDriver, CurConstructor, CurChpConstructor).
    """
    soup = html_or_soup if isinstance(html_or_soup, BS) else BS(html_or_soup, "html.parser")
    tbl = soup.find(id="ctl00_CPH_Main_TBL_Grille")
    if not tbl:
        return []

    entries: List[Dict] = []

    # find all divs with id starting with 'Grd'
    for div in tbl.find_all("div"):
        did = div.get("id", "")
        if not did.lower().startswith("grd"):
            continue

        text_html = str(div)
        # position: try to extract leading number before a dot or strong tag
        pos = None
        # attempt 1: look for <strong>NUMBER</strong>
        strong = div.find("strong")
        if strong:
            try:
                pos = int(strong.get_text(strip=True))
            except Exception:
                pos = None

        # attempt 2: leading text like '1.' or '1. '
        if pos is None:
            import re

            m = re.search(r"^(\s*|<br>)*([0-9]{1,2})\s*\.", div.get_text())
            if m:
                try:
                    pos = int(m.group(2))
                except Exception:
                    pos = None

        # driver name
        driver_el = div.find(class_=lambda c: c and "curdriver" in c.lower())
        if not driver_el:
            # some use CurChpDriver etc. fallback to first <a> text
            a = div.find("a")
            driver_name = _get_text_or_none(a) if a else None
        else:
            driver_name = driver_el.get_text(strip=True)

        # constructor
        constructor_el = div.find(class_=lambda c: c and ("constructor" in c.lower() or "chpconstructor" in c.lower()))
        constructor = constructor_el.get_text(strip=True) if constructor_el else None

        # engine
        engine_el = div.find(class_=lambda c: c and "curengine" in c.lower())
        engine = engine_el.get_text(strip=True) if engine_el else None

        if driver_name is None and not constructor and not engine:
            # skip empty / irrelevant divs
            continue

        entries.append({"position": pos, "driver": driver_name, "constructor": constructor, "engine": engine})

    # sort by numeric position when available
    def _pos_key(o):
        return o.get("position") if isinstance(o.get("position"), int) else 999

    entries.sort(key=_pos_key)
    return entries


def save_grid_json(year: int, gp_slug: str, grid: List[Dict], out_dir: Path = Path("exports")) -> Path:
    """Save grid list to `exports/grid_{year}_{gp_slug}.json`"""
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_slug = gp_slug.replace("/", "_").replace("\\", "_")
    out_path = out_dir / f"grid_{year}_{safe_slug}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({"year": year, "gp": gp_slug, "grid": grid}, f, ensure_ascii=False, indent=2)
    return out_path


def build_season_json(year: int, html: str) -> Dict:
    soup = BS(html, "html.parser")
    gps = parse_gpsaison(soup)
    drivers_table = parse_champ_table(soup, "ctl00_CPH_Main_TBL_CHP_Drv")
    manches = drivers_table.get("manches") or []
    driver_rows = drivers_table.get("rows") or []
    gps_entries = []
    for idx, manche in enumerate(manches):
        gp_info = gps[idx] if idx < len(gps) else {"gp_name": manche, "date_text": None}
        gp_name = gp_info.get("gp_name") or manche or f"GP_{idx+1}"
        date_text = gp_info.get("date_text")
        drivers_list = []
        for dr in driver_rows:
            name = dr.get("name")
            results = dr.get("results") or []
            points = None
            if idx < len(results):
                points = results[idx] or None
            drivers_list.append({"name": name, "points": points})
        gps_entries.append({"gp_name": gp_name, "date": date_text, "drivers": drivers_list})
    return {"season": year, "gps": gps_entries}


def save_season_json(year: int, data: Dict, out_dir: Path = Path("exports")) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"statsf1_season_{year}.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out_path


def scrape_season(year: int, respect_robots: bool = True, user_agent: Optional[str] = DEFAULT_UA) -> Path:
    url = URL_TEMPLATE.format(year)
    if respect_robots:
        r = check_robots(user_agent=user_agent)
        if not r.get("ok"):
            raise RuntimeError(f"robots.txt disallows scraping {url}: {r.get('reason')}")
    html = fetch_page(url, ua=user_agent)
    data = build_season_json(year, html)
    path = save_season_json(year, data)
    return path


if __name__ == "__main__":
    import sys
    years = list(range(1950, 2025))
    if len(sys.argv) > 1:
        try:
            years = [int(x) for x in sys.argv[1:]]
        except ValueError:
            print("Arguments must be years (e.g. 2024 2025)")
            raise
    for y in years:
        print(f"→ Scraping season {y} — building JSON")
        try:
            p = scrape_season(y)
            print(f"Saved: {p}")
        except Exception as e:
            print(f"Error scraping {y}: {e}")
