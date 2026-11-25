from contextlib import asynccontextmanager
import hashlib
import json
import os
import sqlite3
import struct
import threading
import time

from fastapi import FastAPI, HTTPException
import numpy as np
from pydantic import BaseModel, Field

# Must match ESP32 MLP: (6->12) + (3->6) + (18->12) + (12->3) = 375 floats
NUM_WEIGHTS = 375
QUORUM = int(os.getenv("FL_QUORUM", "2"))
DB_FILE = os.getenv("SQLITE_PATH", "fl_server.db")

db_lock = threading.Lock()


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def compute_sha256(weights: list[float]) -> str:
    """Compute SHA-256 over raw little-endian IEEE-754 float32 bytes."""
    raw_bytes = struct.pack(f"<{len(weights)}f", *weights)
    return hashlib.sha256(raw_bytes).hexdigest()


def init_db() -> None:
    with db_lock:
        conn = get_conn()
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS fl_meta (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    round INTEGER NOT NULL,
                    global_weights TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pending_updates (
                    node_id TEXT PRIMARY KEY,
                    round INTEGER NOT NULL,
                    num_samples INTEGER NOT NULL,
                    local_loss REAL NOT NULL,
                    weights TEXT NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS round_history (
                    round INTEGER PRIMARY KEY,
                    participants TEXT NOT NULL,
                    total_samples INTEGER NOT NULL,
                    weighted_loss REAL NOT NULL,
                    timestamp REAL NOT NULL
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS weather_telemetry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    node_id TEXT NOT NULL,
                    temperature REAL,
                    humidity REAL,
                    pressure REAL,
                    irradiance REAL,
                    v_bat REAL,
                    v_uc REAL,
                    v_pv REAL,
                    used_ultracap INTEGER,
                    timestamp REAL NOT NULL
                )
            """)
            row = conn.execute("SELECT round FROM fl_meta WHERE id = 1").fetchone()
            if row is None:
                rng = np.random.default_rng(seed=42)
                init_w = rng.normal(0.0, 0.1, size=NUM_WEIGHTS).astype(np.float32).tolist()
                conn.execute(
                    "INSERT INTO fl_meta (id, round, global_weights) VALUES (1, 0, ?)",
                    (json.dumps(init_w),),
                )
            conn.commit()
        finally:
            conn.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="WeatherWAN Master FL Server",
    version="2.0.0",
    lifespan=lifespan,
)


class NodeUpdate(BaseModel):
    node_id: str = Field(min_length=1)
    round: int = Field(ge=0)
    num_samples: int = Field(gt=0)
    local_loss: float
    weights: list[float]
    sha256: str = Field(min_length=64, max_length=64)


class TelemetryPacket(BaseModel):
    node_id: str = Field(min_length=1)
    temperature: float
    humidity: float
    pressure: float
    irradiance: float
    v_bat: float
    v_uc: float
    v_pv: float
    used_ultracap: int = Field(ge=0, le=1)


@app.get("/")
def health():
    with db_lock:
        conn = get_conn()
        try:
            meta = conn.execute("SELECT round FROM fl_meta WHERE id = 1").fetchone()
            pending = conn.execute("SELECT COUNT(*) AS c FROM pending_updates").fetchone()
            return {
                "status": "ok",
                "current_round": meta["round"],
                "pending_updates": pending["c"],
                "quorum": QUORUM,
            }
        finally:
            conn.close()


@app.get("/fl/version")
def get_version():
    with db_lock:
        conn = get_conn()
        try:
            meta = conn.execute("SELECT round FROM fl_meta WHERE id = 1").fetchone()
            pending = conn.execute("SELECT COUNT(*) AS c FROM pending_updates").fetchone()
            return {"round": meta["round"], "pending_count": pending["c"]}
        finally:
            conn.close()


@app.get("/fl/model")
def get_global_model():
    with db_lock:
        conn = get_conn()
        try:
            meta = conn.execute("SELECT round, global_weights FROM fl_meta WHERE id = 1").fetchone()
            weights = json.loads(meta["global_weights"])
            return {
                "round": meta["round"],
                "weights": weights,
                "sha256": compute_sha256(weights),
            }
        finally:
            conn.close()


@app.post("/fl/update")
def submit_update(update: NodeUpdate):
    if len(update.weights) != NUM_WEIGHTS:
        raise HTTPException(status_code=400, detail=f"Expected {NUM_WEIGHTS} weights")

    arr = np.array(update.weights, dtype=np.float32)
    if not np.isfinite(arr).all():
        raise HTTPException(status_code=400, detail="Weights contain NaN or Inf")

    weights_f32 = arr.tolist()
    expected_hash = compute_sha256(weights_f32)
    if update.sha256.lower() != expected_hash.lower():
        raise HTTPException(status_code=400, detail="SHA-256 checksum mismatch")

    with db_lock:
        conn = get_conn()
        try:
            meta = conn.execute("SELECT round FROM fl_meta WHERE id = 1").fetchone()
            current_round = meta["round"]

            conn.execute(
                """
                INSERT INTO pending_updates (node_id, round, num_samples, local_loss, weights, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    round = excluded.round,
                    num_samples = excluded.num_samples,
                    local_loss = excluded.local_loss,
                    weights = excluded.weights,
                    timestamp = excluded.timestamp
                """,
                (
                    update.node_id,
                    update.round,
                    update.num_samples,
                    float(update.local_loss),
                    json.dumps(weights_f32),
                    time.time(),
                ),
            )

            rows = conn.execute("SELECT * FROM pending_updates").fetchall()
            aggregated = False

            if len(rows) >= QUORUM:
                total_samples = sum(r["num_samples"] for r in rows)
                new_weights = np.zeros(NUM_WEIGHTS, dtype=np.float32)
                weighted_loss = 0.0
                participants = []

                for r in rows:
                    factor = r["num_samples"] / total_samples
                    new_weights += factor * np.array(json.loads(r["weights"]), dtype=np.float32)
                    weighted_loss += factor * r["local_loss"]
                    participants.append(r["node_id"])

                conn.execute(
                    """
                    INSERT OR REPLACE INTO round_history (round, participants, total_samples, weighted_loss, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (current_round, ",".join(participants), total_samples, float(weighted_loss), time.time()),
                )
                current_round += 1
                conn.execute(
                    "UPDATE fl_meta SET round = ?, global_weights = ? WHERE id = 1",
                    (current_round, json.dumps(new_weights.tolist())),
                )
                conn.execute("DELETE FROM pending_updates")
                aggregated = True

            conn.commit()
            return {
                "status": "accepted",
                "aggregated": aggregated,
                "current_round": current_round,
            }
        finally:
            conn.close()


@app.post("/telemetry")
def receive_weather_packet(pkt: TelemetryPacket):
    with db_lock:
        conn = get_conn()
        try:
            conn.execute(
                """
                INSERT INTO weather_telemetry
                (node_id, temperature, humidity, pressure, irradiance, v_bat, v_uc, v_pv, used_ultracap, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pkt.node_id, pkt.temperature, pkt.humidity, pkt.pressure,
                    pkt.irradiance, pkt.v_bat, pkt.v_uc, pkt.v_pv,
                    pkt.used_ultracap, time.time(),
                ),
            )
            conn.commit()
            return {"status": "logged"}
        finally:
            conn.close()


@app.get("/fl/metrics")
def get_metrics():
    with db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT round, participants, total_samples, weighted_loss, timestamp FROM round_history ORDER BY round"
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


@app.get("/telemetry")
def get_telemetry(limit: int = 200):
    with db_lock:
        conn = get_conn()
        try:
            rows = conn.execute(
                "SELECT * FROM weather_telemetry ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()