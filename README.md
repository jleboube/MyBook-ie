# Bookie Swarm

Real-data-only sports market scanner.

## Data policy

- The app ingests **live Kalshi API market data only**.
- No mock/random fallback exists.
- If live fetch fails, opportunities are cleared (fail-closed).

## Run

```bash
cp .env.example .env
docker compose up --build
```

Open: `http://localhost:55556`

## Port configuration

- Set `APP_PORT` in `.env` (default `55556`).
- Docker Compose and Uvicorn both respect `APP_PORT`.

## Kalshi private key setup (recommended)

Use a PEM file, not multiline `.env` text.

1. Create file:
   `/Users/joeleboube/Documents/New project/bookie-swarm-app/secrets/kalshi_private_key.pem`
2. Paste full private key PEM into that file.
3. Set in `.env`:
   `KALSHI_PRIVATE_KEY_PATH=/run/secrets/kalshi_private_key.pem`
4. Set in `.env`:
   `KALSHI_API_KEY_ID=...`

`KALSHI_API_PRIVATE_KEY` is supported, but multiline raw PEM in `.env` can break Docker env parsing.

## API endpoints

- `GET /api/status`
- `POST /api/swarm/start`
- `POST /api/swarm/stop`
- `POST /api/swarm/scan`
- `GET /api/opportunities`
- `GET /api/recommendations`
- `POST /api/execute`
- `GET /api/integration-matrix`
