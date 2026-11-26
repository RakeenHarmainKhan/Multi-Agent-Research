# MARA Frontend

MARA's web interface is built with React, TypeScript and Vite.

## Features

- **Chat:** ask in Automatic or Web Research mode, follow agent progress, and view cited answers.
- **Reports:** expand sources, copy or download answers, and regenerate a response.
- **Dashboard:** view backend health, query counts, latency and token usage.
- **History:** search and review runs stored by the current backend process.
- **Settings:** view backend connection and configuration status. Model and search credentials stay on the backend.

Run history resets when the backend restarts.

## Local setup

Requires Node.js and npm; CI uses Node.js 22. Start the [backend](../backend/README.md) first, then from the repository root:

```powershell
cd frontend
npm ci
npm run dev
```

Open the address printed by Vite, normally `http://localhost:5173`. The frontend defaults to the API at `http://localhost:8000`. To use another backend address, set `VITE_API_URL` in `frontend/.env` using [`.env.example`](.env.example) as a guide, then restart Vite.

If the frontend runs on a different port or hostname, add its exact browser origin to `BACKEND_CORS_ORIGINS` in the backend `.env` and restart the backend. Do not put backend API keys in the frontend environment file.

## API integration

Chat sends questions to `POST /api/v1/research/stream`. It displays `stage` events while work is in progress, then renders the returned `result` or an error. Dashboard, history and settings read the API's admin and health endpoints.

## Testing

From `frontend`:

```powershell
npm test
npm run build
npm run lint
```

The last recorded local run passed **89 frontend tests**, as well as build and lint checks.
