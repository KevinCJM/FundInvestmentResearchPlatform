# Production deployment

The base deployment packages the application as one service: FastAPI serves both `/api/*` and
the built React application. Keep the browser and API on the same public domain
unless there is a deliberate need for cross-origin access.

## Local production check

1. Copy `.env.example` to `.env` and set only the values needed by your
   environment.
2. Before building, provide the controlled `data/` build input required by `Dockerfile` (`COPY data ./data`). A clean checkout at the reviewed baseline has no `data/` directory; running the build unchanged in that state fails before runtime storage setup. Follow the [storage](../docs/data/storage.md) and [acquisition](../docs/data/acquisition-protocol.md) contracts to prepare only authorized inputs, and review the build context for private data and credentials. Do not assume the Docker volume supplies a missing build-time directory. This documentation review did not build an image or validate production.
3. Build and start the service:

   ```bash
   docker compose up --build
   ```

4. Confirm `http://localhost:8000/api/health` returns `{"ok": true}` and load
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

## Local development with the assistant

Use `./start_services.sh {start|status|stop|restart}` for the local platform and
the independently installed agent. All three processes bind to loopback. The
script refuses to adopt or kill an unrelated process on any of their ports.
The production Compose/authentication contract above remains separate.
Running `./start_services.sh` without an action defaults to `start`. Repeated
start reuses healthy managed services and starts a missing agent automatically.
The AI agent page is `/settings/ai-agent`; supported workbenches such as the
indicator center load the AI assistant automatically at the bottom right.

Install a reviewed v2 wheel and its matching hash-locked dependencies in a
separate Python 3.12 environment. The default interpreter is
`.venv/portable-agent/bin/python`; to use another independent installation, set
`PORTABLE_AGENT_PYTHON` on first initialization. A relative interpreter path is
anchored to the project root and saved as an absolute path before initialization;
the virtual-environment executable link is preserved rather than resolved to
system Python. A missing executable fails before creating config or agent data.
Do not install the agent into
the platform environment or point `PYTHONPATH` at its source tree. Installation
is explicit; service startup never downloads a moving source branch. Agent and
probe subprocesses receive only process, locale, temporary-directory and TLS
certificate environment settings. Platform/provider/cloud credentials, source
paths and unrelated proxies are not inherited; the running agent receives only
its explicitly paired host service authorization.

```bash
python3.12 -m venv .venv/portable-agent
.venv/portable-agent/bin/python -m pip install --require-hashes -r /absolute/agent-release/requirements.lock
.venv/portable-agent/bin/python -m pip install --no-deps /absolute/agent-release/portable_web_agent-0.2.0-py3-none-any.whl
./start_services.sh start
```

First startup creates owner-only `.run/portable-agent-local.json` with stable
local identity/service credentials, and generates `.run/portable-agent-apps.json`
from the platform's current tool/capability catalog. The application issuer key
is derived by the installed agent CLI from `.data/portable-agent`; the master
key stays there. Existing `data/llm_settings.json` endpoints are allowed and its
profiles are copied once over authenticated HTTP, encrypted by the agent. The
old file and conversation archives are untouched. Partial profile imports are
retryable; later startup does not overwrite user changes. The copy preserves
model timeout, reasoning effort and `context_window_tokens`; 0 retains the
agent's automatic context budget.

For a new installation without legacy model settings, set
`PORTABLE_AGENT_MODEL_ENDPOINTS=https://your-approved-provider/v1` on first start
(comma-separated for several endpoints), then configure the model in the web
settings. The UI remains available for this repair, but overall startup/status
does not report success until a model with a key is selected. Additional allowed
endpoints can be added with that variable and `restart`. Use the exact printed
browser origin; the local login does not permit other origins or public access.

Back up `.data/portable-agent/` and `.run/portable-agent-local.json` together;
they contain persistent conversations, encrypted profiles and paired credentials.
Never delete them as routine log cleanup. The virtual environment and generated
apps file are reproducible. For a completed initialization, missing or mismatched issuer data, or existing nonempty agent data without
its paired local config, fails closed rather than generating another owner or
credentials. Restore both from the same backup; an empty runtime directory alone
can be initialized. Every model endpoint from saved configuration, the environment
or legacy profiles is validated before creating any runtime directory or key.
Use explicit HTTP(S) endpoints with a valid host and optional port 1–65535
(omitted ports use the protocol default), without whitespace,
control characters, URL credentials, queries or fragments. Invalid input is reported without echoing its value;
correct it in the original source before retrying. Do not commit these private files.

First initialization flushes the paired owner, workspace and service/identity
credentials before issuing a master key. Both file contents and parent directory
entries are fsynced before moving to the next stage, including newly created
runtime/data directories and the linked master key. Its `initializing` record is only
accepted by `prepare`; service launch/status remain blocked until completion.
The installed CLI creates the new key in a private scratch directory; only a
successful, validated result is linked into place without overwriting a key.
If the key command is interrupted or a later config write fails, rerun `start`
to continue with the same recorded identity and any already published master.
Incomplete scratch keys cannot become the active key. This recovery does not
adopt a pre-existing data volume whose paired configuration is missing. A sync
failure stops initialization; a retry resyncs the pending identity and retains
the same published key. Offline fault-injection tests cover these boundaries;
they are not a physical power-loss or filesystem-hardware certification.

Start checks the managed SSD identity first and refuses local fallback when it
is offline. Restore the original mounted volume before retrying; the script
does not create a replacement directory under a missing mount. Backend readiness
still requires all NJIT/worker warmup. Assistant readiness exercises local login,
context/bootstrap authorization, v2 capabilities, the JavaScript component and
model selection, including the actual `/assistant` proxy through the frontend.
Before reusing a healthy backend, startup checks its local identity cookie,
application signature and service credentials against the current paired config.
An owned backend from before this integration is gracefully restarted if that
check fails, then warmup and host authorization are checked again. Stop failure
or a remaining listener blocks replacement; unknown services are never stopped.
After a port opens, startup waits for authentication and the component/proxy to
be ready, retrying transient failures for up to 60 seconds per service. Override
with `ASSISTANT_HEALTH_TIMEOUT_SECONDS` (`STARTUP_WAIT_SECONDS` remains accepted
as its fallback). A timeout or an exited/foreign process fails startup with a
reason; the repair settings page remains available if model configuration is
missing. The final success message includes both AI entry points.
It does not send a billable model request on every status check. After initial
setup or provider changes, open the AI agent page and send a short benign message
to verify provider access. Successful HTTP checks alone do not prove model quality.

`stop` closes the recorded frontend, agent and backend sessions in that order;
independent ETL executors remain governed by their existing task lifecycle.
Default ports are 5173/8787/8000; overrides are `FRONTEND_PORT`, `AGENT_PORT` and
`BACKEND_PORT`. Explicit local service ports must be 1–65535; invalid ports
and URLs containing whitespace/control characters are rejected before parsing
or creating any identity configuration or master key. Check copied environment
values for CR/LF or trailing spaces. Logs
and process identities are in `.run/`.

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
