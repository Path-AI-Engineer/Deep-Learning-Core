# Frontend

PyTorch Tabular Studio is a React, TypeScript and Vite SPA. During development Vite proxies
`/api` to FastAPI on port 8008. Production output is served by the same FastAPI container.

```powershell
npm install
npm run dev
```

The interface never trains models. It reads task schemas, model cards and approved prediction
contracts from `/api/v1`.

`npm test` compiles the tested TypeScript modules and runs deterministic Node
runtime checks for CSV validation and accessible recoverable-error output.
`npm run test:vitest` keeps the richer jsdom suite available in unrestricted
development environments.
