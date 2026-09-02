# Production deployment

This application is deployed as one service: FastAPI serves both `/api/*` and
the built React application. Keep the browser and API on the same public domain
unless there is a deliberate need for cross-origin access.

## Local production check

1. Copy `.env.example` to `.env` and set only the values needed by your
   environment.
2. Build and start the service:

   ```bash
   docker compose up --build
   ```

3. Confirm `http://localhost:8000/api/health` returns `{"ok": true}` and load
   `http://localhost:8000`.

The `fund_data` volume persists ETF data plus any saved asset allocations and
backtest results. Back up this volume before redeploying or migrating hosts.

## Cloud target requirements

Choose a service that runs Docker containers and provides:

- a persistent disk mounted at `/app/data`;
- a public HTTPS domain routed to container port `8000`;
- a health check at `/api/health`;
- environment variables for `APP_ENV=production` and, only if the frontend is
  hosted on a different domain, `CORS_ALLOW_ORIGINS`.

The Dashboard accepts the Tushare Token only through `PUT /api/data/token` and
never returns it. Keep `DATA_REFRESH_ENABLED=false` on a public deployment
unless `/api/data/*` is protected by HTTPS and authentication. The credential
is persisted on the `/app/data` volume with owner-only file permissions.

## Release checks

Run the frontend test and build, then the backend route regression tests before
building a production image. Verify the health check after deployment and make
sure the persistent data disk contains the required Parquet files.
