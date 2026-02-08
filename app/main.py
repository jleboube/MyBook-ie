import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from starlette.requests import Request

APP_NAME = "Bookie Swarm"
APP_VERSION = "0.2.0"
SPORT_KEYWORDS = ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "SOCCER", "MMA", "UFC", "TENNIS", "GOLF"]
LEAGUE_PREFIXES = ["NCAAB", "NCAAF", "SOCCER", "TENNIS", "NBA", "NFL", "MLB", "NHL", "MMA", "UFC", "GOLF", "WNBA"]
STAT_CODE_MAP = {
    "PTS": "Points",
    "AST": "Assists",
    "REB": "Rebounds",
    "PRA": "Points+Rebounds+Assists",
    "PR": "Points+Rebounds",
    "PA": "Points+Assists",
    "RA": "Rebounds+Assists",
    "BLK": "Blocks",
    "STL": "Steals",
    "TOV": "Turnovers",
    "THREES": "Three-Pointers Made",
    "FGM": "Field Goals Made",
    "FGA": "Field Goal Attempts",
    "FTM": "Free Throws Made",
    "FTA": "Free Throw Attempts",
    "REC": "Receptions",
    "RECYDS": "Receiving Yards",
    "RSHYDS": "Rushing Yards",
    "PASSYDS": "Passing Yards",
    "PASSATT": "Passing Attempts",
    "PASSCMP": "Completions",
    "PASSTDS": "Passing Touchdowns",
    "RUSHTDS": "Rushing Touchdowns",
    "RECTDS": "Receiving Touchdowns",
    "ANYTD": "Anytime Touchdown",
    "FIRSTTD": "First Touchdown Scorer",
    "2TD": "Two+ Touchdowns",
    "PASSINT": "Interceptions Thrown",
    "TDS": "Touchdowns",
    "TD": "Touchdowns",
    "NCAAMBGAME": "Game Winner",
    "NCAAWBGAME": "Game Winner",
    "NBAGAME": "Game Winner",
    "NFLGAME": "Game Winner",
    "MLBGAME": "Game Winner",
    "NHLGAME": "Game Winner",
    "SB": "Game Winner",
}

logger = logging.getLogger("bookie_swarm")
logging.basicConfig(level=logging.INFO)

BASE_DIR = Path(__file__).resolve().parent


class ExecuteRequest(BaseModel):
    id: str
    sportsbook: Literal["fanduel", "draftkings", "kalshi"]
    action: Literal["queue", "place"] = "queue"


class SwarmControl(BaseModel):
    interval_seconds: int = Field(default=20, ge=5, le=300)


@dataclass
class Opportunity:
    id: str
    sport: str
    event: str
    market_type: Literal["moneyline", "spread", "total", "parlay", "prop"]
    sportsbook: Literal["fanduel", "draftkings", "kalshi"]
    selection: str
    odds_american: int
    implied_probability: float
    model_probability: float
    edge: float
    confidence: float
    rationale: str
    timestamp: float
    leg_details: Optional[List[Dict[str, Any]]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "sport": self.sport,
            "event": self.event,
            "market_type": self.market_type,
            "sportsbook": self.sportsbook,
            "selection": self.selection,
            "odds_american": self.odds_american,
            "implied_probability": round(self.implied_probability, 4),
            "model_probability": round(self.model_probability, 4),
            "edge": round(self.edge, 4),
            "confidence": round(self.confidence, 4),
            "rationale": self.rationale,
            "timestamp": self.timestamp,
            "leg_details": self.leg_details or [],
        }


def probability_to_american(prob: float) -> int:
    prob = min(max(prob, 0.01), 0.99)
    if prob >= 0.5:
        return int(round(-(prob / (1 - prob)) * 100))
    return int(round(((1 - prob) / prob) * 100))


def classify_sport(text: str) -> str:
    upper = text.upper()
    for key in SPORT_KEYWORDS:
        if key in upper:
            if key == "UFC":
                return "MMA"
            return key
    if "NCAAM" in upper or "CBB" in upper:
        return "NCAAB"
    if "NCAAF" in upper or "CFB" in upper:
        return "NCAAF"
    return "Sports"


def classify_market_type(title: str) -> Literal["moneyline", "spread", "total", "parlay", "prop"]:
    t = title.lower()
    if " over " in t or " under " in t or "total" in t:
        return "total"
    if "spread" in t or re.search(r"\b[+-]\d+(\.\d+)?\b", t):
        return "spread"
    if ",yes " in t or "same game" in t or " sgp" in t:
        return "parlay"
    prop_keywords = [
        "yards",
        "touchdown",
        "points",
        "rebounds",
        "assists",
        "goals",
        "strikeouts",
        "saves",
        "receptions",
        "attempts",
        "passes",
        "rushing",
    ]
    if any(k in t for k in prop_keywords):
        return "prop"
    return "moneyline"


def clamp_prob_from_cents(value: Any) -> Optional[float]:
    try:
        iv = int(value)
    except (TypeError, ValueError):
        return None
    if iv <= 0 or iv >= 100:
        return None
    return iv / 100.0


def split_csv(value: Any) -> List[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return [x.strip() for x in value.split(",") if x.strip()]


def normalize_stat_code_from_prefix(prefix: str) -> str:
    stripped = prefix
    for league in LEAGUE_PREFIXES:
        if stripped.startswith(league):
            stripped = stripped[len(league):]
            break
    return stripped or prefix


def stat_label_from_leg_ticker(leg_ticker: str) -> str:
    first = leg_ticker.split("-", 1)[0].upper()
    if first.startswith("KX"):
        first = first[2:]
    code = normalize_stat_code_from_prefix(first)
    return STAT_CODE_MAP.get(code, code)


def parse_leg_threshold(leg_ticker: str) -> Optional[int]:
    match = re.search(r"-(\d+)$", leg_ticker)
    if not match:
        return None
    return int(match.group(1))


def parse_parlay_leg_details(title: str, custom_strike: Any) -> List[Dict[str, Any]]:
    if not isinstance(custom_strike, dict):
        return []
    associated_markets = split_csv(custom_strike.get("Associated Markets"))
    if not associated_markets:
        return []

    title_parts = [seg.strip() for seg in title.split(",") if seg.strip()]
    cleaned_title_parts = []
    for seg in title_parts:
        lowered = seg.lower()
        if lowered.startswith("yes "):
            cleaned_title_parts.append(seg[4:].strip())
        else:
            cleaned_title_parts.append(seg)

    legs: List[Dict[str, Any]] = []
    for idx, market_ticker in enumerate(associated_markets):
        stat_label = stat_label_from_leg_ticker(market_ticker)
        threshold = parse_leg_threshold(market_ticker)
        display_leg = cleaned_title_parts[idx] if idx < len(cleaned_title_parts) else market_ticker
        legs.append(
            {
                "index": idx + 1,
                "display_leg": display_leg,
                "stat": stat_label,
                "threshold": threshold,
                "market_ticker": market_ticker,
            }
        )
    return legs


def load_kalshi_private_key_material() -> str:
    raw = os.getenv("KALSHI_API_PRIVATE_KEY", "").strip()
    if raw:
        return raw
    path = os.getenv("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not path:
        return ""
    candidates = [Path(path)]
    # If user supplied a host path in .env, try mounted secret basename in container.
    candidates.append(Path("/run/secrets") / Path(path).name)
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
    return ""


class KalshiOddsProvider:
    def __init__(self):
        self.base = os.getenv("KALSHI_API_BASE", "https://api.elections.kalshi.com").rstrip("/")
        self.timeout = float(os.getenv("KALSHI_HTTP_TIMEOUT", "20"))

    def fetch(self) -> List[Dict[str, Any]]:
        url = f"{self.base}/trade-api/v2/markets"
        params = {"limit": 300}

        with httpx.Client(timeout=self.timeout) as client:
            resp = client.get(url, params=params)
            resp.raise_for_status()
            payload = resp.json()

        markets = payload.get("markets", [])
        rows: List[Dict[str, Any]] = []
        for market in markets:
            ticker = str(market.get("ticker", ""))
            event_ticker = str(market.get("event_ticker", ""))
            title = str(market.get("title", "")).strip()
            corpus = f"{ticker} {event_ticker} {title}"

            if "SPORT" not in corpus.upper() and not any(k in corpus.upper() for k in SPORT_KEYWORDS):
                continue

            custom_strike = market.get("custom_strike")
            associated_markets = split_csv(custom_strike.get("Associated Markets")) if isinstance(custom_strike, dict) else []
            is_multi_leg = len(associated_markets) > 1
            market_type = "parlay" if is_multi_leg else classify_market_type(title)
            sport = classify_sport(corpus)
            leg_details = parse_parlay_leg_details(title, custom_strike) if is_multi_leg else []
            rows.append(
                {
                    "event_id": ticker,
                    "sport": sport,
                    "event": title or event_ticker,
                    "sportsbook": "kalshi",
                    "market_type": market_type,
                    "selection": f"YES - {title}" if title else f"YES - {ticker}",
                    "yes_ask": market.get("yes_ask"),
                    "no_ask": market.get("no_ask"),
                    "yes_bid": market.get("yes_bid"),
                    "no_bid": market.get("no_bid"),
                    "last_price": market.get("last_price"),
                    "volume": int(market.get("volume", 0) or 0),
                    "liquidity": int(market.get("liquidity", 0) or 0),
                    "close_time": market.get("close_time"),
                    "status": market.get("status"),
                    "leg_details": leg_details,
                    "ts": time.time(),
                }
            )
        return rows


class SportAgent:
    def __init__(self, sport: str):
        self.sport = sport
        self.name = f"{sport.lower()}_agent"

    def run(self, snapshots: List[Dict[str, Any]]) -> List[Opportunity]:
        out: List[Opportunity] = []
        for row in snapshots:
            if row["sport"] != self.sport:
                continue
            if row["market_type"] not in ("moneyline", "spread", "total"):
                continue
            opportunity = build_opportunity(row, self.name)
            if opportunity:
                out.append(opportunity)
        return out


class PropAgent:
    name = "prop_agent"

    def run(self, snapshots: List[Dict[str, Any]]) -> List[Opportunity]:
        out: List[Opportunity] = []
        for row in snapshots:
            if row["market_type"] != "prop":
                continue
            opportunity = build_opportunity(row, self.name)
            if opportunity:
                out.append(opportunity)
        return out


class ParlayAgent:
    name = "parlay_agent"

    def run(self, snapshots: List[Dict[str, Any]]) -> List[Opportunity]:
        out: List[Opportunity] = []
        for row in snapshots:
            if row["market_type"] != "parlay":
                continue
            opportunity = build_opportunity(row, self.name)
            if opportunity:
                out.append(opportunity)
        return out


def build_opportunity(row: Dict[str, Any], agent_name: str) -> Optional[Opportunity]:
    implied = clamp_prob_from_cents(row.get("yes_ask"))
    if implied is None:
        implied = clamp_prob_from_cents(row.get("last_price"))
    if implied is None:
        return None

    model = None
    no_ask_prob = clamp_prob_from_cents(row.get("no_ask"))
    if no_ask_prob is not None:
        model = 1.0 - no_ask_prob
    if model is None:
        no_bid_prob = clamp_prob_from_cents(row.get("no_bid"))
        if no_bid_prob is not None:
            model = 1.0 - no_bid_prob
    if model is None:
        model = clamp_prob_from_cents(row.get("yes_bid"))
    if model is None:
        model = implied

    edge = model - implied
    if edge < 0.005:
        return None

    liquidity = int(row.get("liquidity", 0))
    volume = int(row.get("volume", 0))
    liq_score = min(1.0, liquidity / 5000.0)
    vol_score = min(1.0, volume / 2000.0)
    confidence = min(0.98, 0.45 + (edge * 3.5) + (liq_score * 0.2) + (vol_score * 0.15))

    return Opportunity(
        id=row["event_id"],
        sport=row["sport"],
        event=row["event"],
        market_type=row["market_type"],
        sportsbook="kalshi",
        selection=row["selection"],
        odds_american=probability_to_american(implied),
        implied_probability=implied,
        model_probability=model,
        edge=edge,
        confidence=confidence,
        rationale=f"{agent_name} found positive YES-side edge from live Kalshi orderbook prices.",
        timestamp=row["ts"],
        leg_details=row.get("leg_details", []),
    )


class BookieOrchestrator:
    def __init__(self):
        self.provider = KalshiOddsProvider()
        self.sport_agents = [SportAgent(s) for s in ["NFL", "NBA", "MLB", "NHL", "NCAAF", "NCAAB", "SOCCER", "MMA", "TENNIS", "GOLF", "Sports"]]
        self.prop_agent = PropAgent()
        self.parlay_agent = ParlayAgent()

        self.running = False
        self.interval_seconds = 20
        self._task: Optional[asyncio.Task] = None

        self.last_scan_at: Optional[float] = None
        self.last_error: Optional[str] = None
        self.snapshots: List[Dict[str, Any]] = []
        self.opportunities: Dict[str, Opportunity] = {}
        self.recommendations: List[Opportunity] = []
        self.execution_queue: List[Dict[str, Any]] = []

    def start(self, interval_seconds: int = 20) -> None:
        if self.running:
            self.interval_seconds = interval_seconds
            return
        self.interval_seconds = interval_seconds
        self.running = True
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _loop(self) -> None:
        while self.running:
            self.scan_once()
            await asyncio.sleep(self.interval_seconds)

    def scan_once(self) -> None:
        try:
            snapshots = self.provider.fetch()
            all_opps: List[Opportunity] = []

            for agent in self.sport_agents:
                all_opps.extend(agent.run(snapshots))
            all_opps.extend(self.prop_agent.run(snapshots))
            all_opps.extend(self.parlay_agent.run(snapshots))

            deduped: Dict[str, Opportunity] = {}
            for opp in all_opps:
                prev = deduped.get(opp.id)
                if not prev or opp.edge > prev.edge:
                    deduped[opp.id] = opp

            ranked = sorted(
                deduped.values(),
                key=lambda o: (o.edge * 0.7) + (o.confidence * 0.3),
                reverse=True,
            )

            self.snapshots = snapshots
            self.last_scan_at = time.time()
            self.last_error = None
            self.opportunities = deduped
            self.recommendations = ranked[:25]
        except Exception as exc:
            logger.exception("Live Kalshi scan failed")
            # Fail closed: never fallback to mock data.
            self.last_scan_at = time.time()
            self.last_error = str(exc)
            self.snapshots = []
            self.opportunities = {}
            self.recommendations = []

    def execute(self, req: ExecuteRequest) -> Dict[str, Any]:
        opp = self.opportunities.get(req.id)
        if not opp:
            raise KeyError("Opportunity not found")

        if req.action == "queue":
            payload = {
                "action": "queue",
                "status": "queued",
                "opportunity": opp.to_dict(),
                "queued_at": time.time(),
            }
            self.execution_queue.append(payload)
            return payload

        if req.sportsbook in ("fanduel", "draftkings"):
            return {
                "action": "place",
                "status": "manual_required",
                "reason": "Manual mode only for this sportsbook in this app.",
                "opportunity": opp.to_dict(),
            }

        return {
            "action": "place",
            "status": "manual_required",
            "reason": "Opening the exact Kalshi market so the user can enter stake and submit.",
            "opportunity": opp.to_dict(),
            "deeplink": f"https://kalshi.com/markets/{opp.id}",
        }


bookie = BookieOrchestrator()

app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@app.on_event("startup")
async def startup() -> None:
    bookie.scan_once()


@app.on_event("shutdown")
async def shutdown() -> None:
    await bookie.stop()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "app_name": APP_NAME,
            "version": APP_VERSION,
            "port": os.getenv("APP_PORT", "55555"),
        },
    )


@app.get("/api/status")
async def status() -> Dict[str, Any]:
    return {
        "app": APP_NAME,
        "version": APP_VERSION,
        "running": bookie.running,
        "interval_seconds": bookie.interval_seconds,
        "last_scan_at": bookie.last_scan_at,
        "last_error": bookie.last_error,
        "data_source": "kalshi_live_only",
        "opportunity_count": len(bookie.opportunities),
        "recommendation_count": len(bookie.recommendations),
        "queue_count": len(bookie.execution_queue),
        "live_market_snapshot_count": len(bookie.snapshots),
        "kalshi_key_id_set": bool(os.getenv("KALSHI_API_KEY_ID", "").strip()),
        "kalshi_private_key_set": bool(load_kalshi_private_key_material()),
    }


@app.post("/api/swarm/start")
async def start_swarm(control: SwarmControl) -> Dict[str, Any]:
    bookie.start(interval_seconds=control.interval_seconds)
    return {"running": True, "interval_seconds": control.interval_seconds}


@app.post("/api/swarm/stop")
async def stop_swarm() -> Dict[str, Any]:
    await bookie.stop()
    return {"running": False}


@app.post("/api/swarm/scan")
async def scan_once() -> Dict[str, Any]:
    bookie.scan_once()
    return {
        "status": "ok" if not bookie.last_error else "error",
        "scanned_at": bookie.last_scan_at,
        "last_error": bookie.last_error,
        "opportunity_count": len(bookie.opportunities),
    }


@app.get("/api/opportunities")
async def opportunities(
    sport: Optional[str] = Query(default=None),
    sportsbook: Optional[str] = Query(default=None),
    market_type: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> Dict[str, Any]:
    items = list(bookie.opportunities.values())
    if sport:
        items = [i for i in items if i.sport.lower() == sport.lower()]
    if sportsbook:
        items = [i for i in items if i.sportsbook == sportsbook]
    if market_type:
        items = [i for i in items if i.market_type == market_type]

    ranked = sorted(items, key=lambda o: (o.edge, o.confidence), reverse=True)
    return {"items": [x.to_dict() for x in ranked[:limit]], "count": len(ranked), "live_only": True}


@app.get("/api/recommendations")
async def recommendations(limit: int = Query(default=25, ge=1, le=100)) -> Dict[str, Any]:
    return {
        "bookie": "bookie",
        "items": [x.to_dict() for x in bookie.recommendations[:limit]],
        "count": len(bookie.recommendations),
        "live_only": True,
    }


@app.post("/api/execute")
async def execute(req: ExecuteRequest):
    try:
        result = bookie.execute(req)
    except KeyError as exc:
        return JSONResponse({"error": str(exc)}, status_code=404)
    return result


@app.get("/api/integration-matrix")
async def integration_matrix() -> Dict[str, Any]:
    return {
        "as_of": "2026-02-07",
        "integrations": [
            {
                "sportsbook": "kalshi",
                "market_data": "live_official_api_only",
                "order_placement": "official_api_supported_not_implemented_here",
                "notes": "This app only ingests live Kalshi market data and fails closed on API errors.",
                "docs": "https://docs.kalshi.com/",
            },
            {
                "sportsbook": "fanduel",
                "market_data": "not_ingested",
                "order_placement": "manual_only",
                "notes": "No FanDuel market ingestion in this app.",
                "docs": "https://www.fanduel.com/terms",
            },
            {
                "sportsbook": "draftkings",
                "market_data": "not_ingested",
                "order_placement": "manual_only",
                "notes": "No DraftKings market ingestion in this app.",
                "docs": "https://sportsbook.draftkings.com/legal/us-terms-of-use",
            },
        ],
    }
