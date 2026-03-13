#!/usr/bin/env python3
"""Polymarket Gamma API helpers for deep sports market discovery."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"
TIMEOUT = 25

SPORT_TAG_SLUGS = {
    "ufc": "ufc",
    "nba": "nba",
    "nfl": "nfl",
    "nhl": "hockey",
    "soccer": "soccer",
}

EURO_CLUB_SOCCER_KEYWORDS = [
    "uefa champions league",
    "champions league",
    "uefa europa league",
    "europa league",
    "uefa conference league",
    "conference league",
    "premier league",
    "la liga",
    "serie a",
    "bundesliga",
    "ligue 1",
    "eredivisie",
    "primeira liga",
    "belgian pro league",
    "scottish premiership",
    "turkish super lig",
]

SOCCER_EXCLUDE_KEYWORDS = [
    "more markets",
    "sidemen",
    "youtube allstars",
    "mls",
    "liga mx",
    "nwsl",
    "national team",
    "world cup",
    "qualifier",
]


def _parse_jsonish(val: Any) -> Any:
    if isinstance(val, (list, dict)):
        return val
    if not isinstance(val, str):
        return val
    txt = val.strip()
    if not txt:
        return val
    try:
        return json.loads(txt)
    except Exception:
        return val


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


def parse_matchup_from_title(title: str) -> Tuple[Optional[str], Optional[str]]:
    if not title:
        return None, None
    text = str(title).strip()
    # For sports titles like "UFC Fight Night: A vs. B (...)"
    if ":" in text:
        text = text.split(":")[-1].strip()
    text = re.sub(r"\(.*?\)", "", text).strip()
    m = re.search(r"(.+?)\s+vs\.?\s+(.+)", text, flags=re.IGNORECASE)
    if not m:
        m = re.search(r"(.+?)\s+v\.?\s+(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None, None
    return m.group(1).strip(), m.group(2).strip()


def _is_europe_club_soccer_event(title: str) -> bool:
    t = str(title or "").lower()
    if " vs" not in t:
        return False
    if any(x in t for x in SOCCER_EXCLUDE_KEYWORDS):
        return False
    return any(k in t for k in EURO_CLUB_SOCCER_KEYWORDS)


@dataclass
class MarketRow:
    sport: str
    event_id: str
    event_slug: str
    title: str
    start_date: Optional[str]
    end_date: Optional[str]
    liquidity: float
    volume: float
    market_id: str
    market_slug: str
    outcome_a: str
    outcome_draw: Optional[str]
    outcome_b: str
    price_a_cents: float
    price_draw_cents: Optional[float]
    price_b_cents: float
    source: str

    def as_dict(self) -> Dict[str, Any]:
        return {
            "sport": self.sport,
            "event_id": self.event_id,
            "event_slug": self.event_slug,
            "title": self.title,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "liquidity": self.liquidity,
            "volume": self.volume,
            "market_id": self.market_id,
            "market_slug": self.market_slug,
            "outcome_a": self.outcome_a,
            "outcome_draw": self.outcome_draw,
            "outcome_b": self.outcome_b,
            "price_a_cents": self.price_a_cents,
            "price_draw_cents": self.price_draw_cents,
            "price_b_cents": self.price_b_cents,
            "source": self.source,
        }


class GammaClient:
    def __init__(self, base_url: str = GAMMA_BASE, session: Optional[requests.Session] = None):
        self.base_url = base_url.rstrip("/")
        self.s = session or requests.Session()

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        url = f"{self.base_url}{path}"
        resp = self.s.get(url, params=params or {}, timeout=TIMEOUT)
        resp.raise_for_status()
        return resp.json()

    def list_tags(self, limit: int = 500, max_pages: int = 8) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        offset = 0
        for _ in range(max_pages):
            rows = self._get("/tags", {"limit": limit, "offset": offset})
            if not rows:
                break
            out.extend(rows)
            if len(rows) < limit:
                break
            offset += limit
        return out

    def list_sports(self, limit: int = 500, max_pages: int = 8) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        offset = 0
        for _ in range(max_pages):
            rows = self._get("/sports", {"limit": limit, "offset": offset})
            if not rows:
                break
            out.extend(rows)
            if len(rows) < limit:
                break
            offset += limit
        return out

    def fetch_events(
        self,
        tag_slug: str,
        pages: int = 5,
        page_size: int = 100,
        active_only: bool = True,
        closed: bool = False,
        archived: bool = False,
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for p in range(max(1, pages)):
            params = {
                "limit": page_size,
                "offset": p * page_size,
                "tag_slug": tag_slug,
                "closed": str(closed).lower(),
                "archived": str(archived).lower(),
            }
            if active_only:
                params["active"] = "true"
            rows = self._get("/events", params)
            if not rows:
                break
            out.extend(rows)
            if len(rows) < page_size:
                break
        return out

    def _extract_main_market(
        self,
        event: Dict[str, Any],
        title_team_a: Optional[str],
        title_team_b: Optional[str],
        sport_key: str,
    ) -> Optional[MarketRow]:
        markets = event.get("markets") or []
        if not markets:
            return None

        t_a_n = _normalize_name(title_team_a or "")
        t_b_n = _normalize_name(title_team_b or "")

        candidates: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
        for m in markets:
            outcomes = _parse_jsonish(m.get("outcomes"))
            prices = _parse_jsonish(m.get("outcomePrices"))
            if not isinstance(outcomes, list) or not isinstance(prices, list):
                continue
            if len(outcomes) != len(prices):
                continue

            score = _to_float(m.get("liquidity"), 0.0)
            normalized = [_normalize_name(o) for o in outcomes]

            # Two-way team-v-team.
            if len(outcomes) == 2:
                o1 = str(outcomes[0]).strip()
                o2 = str(outcomes[1]).strip()
                if {o1.lower(), o2.lower()} == {"yes", "no"}:
                    continue
                if {o1.lower(), o2.lower()} == {"over", "under"}:
                    continue
                p1 = _to_float(prices[0], default=-1.0)
                p2 = _to_float(prices[1], default=-1.0)
                if p1 < 0 or p2 < 0:
                    continue

                payload = {
                    "outcome_a": o1,
                    "outcome_draw": None,
                    "outcome_b": o2,
                    "price_a_cents": p1 * 100.0,
                    "price_draw_cents": None,
                    "price_b_cents": p2 * 100.0,
                }

                if t_a_n and t_b_n:
                    o1n = _normalize_name(o1)
                    o2n = _normalize_name(o2)
                    if o1n == t_a_n and o2n == t_b_n:
                        score += 1_000_000.0
                    elif o1n == t_b_n and o2n == t_a_n:
                        # reorder so A follows title team A
                        payload = {
                            "outcome_a": o2,
                            "outcome_draw": None,
                            "outcome_b": o1,
                            "price_a_cents": p2 * 100.0,
                            "price_draw_cents": None,
                            "price_b_cents": p1 * 100.0,
                        }
                        score += 900_000.0
                    else:
                        continue
                candidates.append((score, m, payload))
                continue

            # Soccer 3-way (Team A / Draw / Team B).
            if sport_key == "soccer" and len(outcomes) == 3:
                draw_idx = None
                for i, on in enumerate(normalized):
                    if on in {"draw", "tie", "x"}:
                        draw_idx = i
                        break
                if draw_idx is None:
                    continue
                team_idx = [i for i in range(3) if i != draw_idx]
                oa = str(outcomes[team_idx[0]]).strip()
                ob = str(outcomes[team_idx[1]]).strip()
                od = str(outcomes[draw_idx]).strip()
                pa = _to_float(prices[team_idx[0]], default=-1.0)
                pb = _to_float(prices[team_idx[1]], default=-1.0)
                pdw = _to_float(prices[draw_idx], default=-1.0)
                if pa < 0 or pb < 0 or pdw < 0:
                    continue

                payload = {
                    "outcome_a": oa,
                    "outcome_draw": od,
                    "outcome_b": ob,
                    "price_a_cents": pa * 100.0,
                    "price_draw_cents": pdw * 100.0,
                    "price_b_cents": pb * 100.0,
                }
                if t_a_n and t_b_n:
                    oan = _normalize_name(oa)
                    obn = _normalize_name(ob)
                    if oan == t_a_n and obn == t_b_n:
                        score += 1_200_000.0
                    elif oan == t_b_n and obn == t_a_n:
                        payload = {
                            "outcome_a": ob,
                            "outcome_draw": od,
                            "outcome_b": oa,
                            "price_a_cents": pb * 100.0,
                            "price_draw_cents": pdw * 100.0,
                            "price_b_cents": pa * 100.0,
                        }
                        score += 1_100_000.0
                    else:
                        continue
                candidates.append((score, m, payload))
                continue

        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, chosen, payload = candidates[0]
        return MarketRow(
            sport="",
            event_id=str(event.get("id", "")),
            event_slug=str(event.get("slug", "")),
            title=str(event.get("title", "")),
            start_date=event.get("startDate"),
            end_date=event.get("endDate"),
            liquidity=_to_float(event.get("liquidity"), 0.0),
            volume=_to_float(event.get("volume"), 0.0),
            market_id=str(chosen.get("id", "")),
            market_slug=str(chosen.get("slug", "")),
            outcome_a=payload["outcome_a"],
            outcome_draw=payload["outcome_draw"],
            outcome_b=payload["outcome_b"],
            price_a_cents=payload["price_a_cents"],
            price_draw_cents=payload["price_draw_cents"],
            price_b_cents=payload["price_b_cents"],
            source="events",
        )

    def fetch_game_markets(
        self,
        sport: str,
        pages: int = 6,
        page_size: int = 100,
        query: Optional[str] = None,
        min_liquidity: float = 0.0,
        soccer_club_only: bool = True,
    ) -> List[MarketRow]:
        sport_key = sport.strip().lower()
        tag_slug = SPORT_TAG_SLUGS.get(sport_key, sport_key)
        events = self.fetch_events(tag_slug=tag_slug, pages=pages, page_size=page_size, active_only=True, closed=False, archived=False)
        out: List[MarketRow] = []

        qn = _normalize_name(query) if query else ""
        for e in events:
            title = str(e.get("title", "")).strip()
            if qn and qn not in _normalize_name(title):
                continue
            if sport_key == "soccer" and soccer_club_only and not _is_europe_club_soccer_event(title):
                continue

            t1, t2 = parse_matchup_from_title(title)
            if sport_key in {"nba", "nfl", "nhl", "ufc"} and (not t1 or not t2):
                continue
            row = self._extract_main_market(e, t1, t2, sport_key=sport_key)
            if not row:
                continue
            if row.liquidity < min_liquidity:
                continue
            row.sport = sport_key
            out.append(row)

        # Dedup by event_id + market_id in case pagination overlaps.
        dedup: Dict[Tuple[str, str], MarketRow] = {}
        for r in out:
            dedup[(r.event_id, r.market_id)] = r
        return list(dedup.values())
