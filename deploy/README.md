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

## Independent assistant candidate

The migration candidate adds a separate agent and a same-origin gateway. The
base Compose file remains the manual business application. The assistant
overlay requires Compose 2.24.4 or later:

```bash
docker compose -f docker-compose.yml -f deploy/portable-agent.compose.yml config --quiet
docker compose -f docker-compose.yml -f deploy/portable-agent.compose.yml up --build
```

Only the gateway publishes a port (loopback by default). An HTTPS ingress must
route to it. `/assistant/widget/` and `/assistant/v2/apps/` go to the agent;
SSE buffering is disabled. `/internal/` and other assistant paths are denied.
The two services keep distinct persistent volumes. The agent receives only
its own data, declarative configuration and the host-tool credential.
Regenerate the agent configuration with `scripts/export_portable_agent_config.py`
when the shared navigation/capability catalog changes, so the agent loads the
same stage, node and tool links exposed by the host capability API.

Before starting, supply these environment variables from deployment secrets:

| Variable | Meaning |
| --- | --- |
| `PORTABLE_AGENT_IMAGE` | Published agent image including `@sha256:…`; must match the release file |
| `PORTABLE_AGENT_GATEWAY_IMAGE` | Pinned nginx image including its digest |
| `PORTABLE_AGENT_RELEASE_FILE` | Absolute path to the reviewed release JSON |
| `PORTABLE_AGENT_CONFIG_FILE` | Absolute path to generated agent `apps.json` |
| `PORTABLE_AGENT_BROWSER_ORIGIN` | Exact public HTTPS origin |
| `PORTABLE_AGENT_IDENTITY_KEY`, `PORTABLE_AGENT_IDENTITY_ISSUER` | Trusted platform identity JWT signing key and issuer |
| `PORTABLE_AGENT_WORKSPACE` | Authorized business workspace |
| `PORTABLE_AGENT_AUTHORITY_URL`, `PORTABLE_AGENT_AUTHORITY_TOKEN` | HTTPS endpoint and credential for checking current user permissions |
| `PORTABLE_AGENT_ISSUER_KEY` | Application-specific key derived by the independent agent CLI from its data volume |
| `PORTABLE_AGENT_SERVICE_TOKEN` | Separate credential for internal host tools, at least 32 characters |

The identity issuer must supply the `research_identity` HttpOnly cookie or a
trusted Bearer token. The current-permissions endpoint receives
`{subject, workspace}` and returns matching `sub`, `workspace`, and `scopes`.
No public local-owner login is enabled by this overlay.

The platform archive importer uses this same authority to mint contexts for the
archive's assigned owner/workspace; the archive cannot grant wildcard permissions.
`--apply` therefore needs the configured authority URL/token in production and
fails before binding the dataset if that identity cannot be verified or lacks
`assistant:use`. Authorization/service failures during import leave it retryable;
restore access and rerun the same archive instead of accepting a partial import. `--apply` in explicit local mode accepts
only the configured local owner/workspace. Dry-run remains offline and creates no
business store. Retry preserves previously frozen grants; live bootstrap still
checks current permissions, including subsequent revocations.

Generate the non-secret application configuration with the platform's Python:

```bash
python3 scripts/export_portable_agent_config.py \
  --platform-url http://fund-research:8000 \
  --origin https://research.example.com \
  --model-endpoint https://provider.example.com/v1 \
  --output /absolute/private-config/apps.json
```

Replace the example origins with the actual approved endpoints. The release
JSON must contain `source_commit` (40 hexadecimal characters), `image_digest`
(full image reference), `protocol_major: 2`, `widget_version`,
`manifest_sha256`, and `required_capabilities`. Required capabilities are
`profiles`, `turn-context`, `draft-tools`, `history-pagination`, `memory`,
`compaction`, `handoff`, `host-admission`, `deferred-operations`,
`artifact-slots`, and `ag-ui`. Values come from the reviewed framework build;
there is deliberately no fabricated release lock in this working tree.
Production bootstrap fails closed if the lock is absent or invalid. The SDK
checks metadata and the loaded component manifest on every connection and
retains the original request identity when versions differ.

For development, run the independently installed agent on loopback and set
`VITE_PORTABLE_AGENT_TARGET` before `./start_services.sh start`. That script
owns only the platform processes; agent lifecycle remains independent.

The offline integration runner creates temporary business/agent data and a
local deterministic OpenAI wire endpoint. It never contacts a model provider:

```bash
INDICATOR_TEST_PYTHON=/absolute/platform/python \
PORTABLE_AGENT_PYTHON=/absolute/independent-agent/python \
npm run test:e2e --prefix frontend -- --config=playwright.portable.config.ts
```

The agent interpreter must contain a built wheel. `PORTABLE_AGENT_SOURCE` is
an explicit development-only import path override; production does not mount
or import the sibling repository. The runner starts tests only after both
services and the fixture model profile are ready, and removes its temporary
data on exit. Current evidence and remaining cutover gates are in the
[migration design](../docs/research/portable-agent-platform-integration.md).
