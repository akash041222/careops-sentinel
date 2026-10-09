# CareOps Sentinel web
`npm install && npm run dev` (API URL via `VITE_API_URL`, default http://127.0.0.1:8000). `npm test` runs the chat tests, `npm run build` makes `dist/`.

Human escalation alarms are implemented in `src/App.jsx` and styled in `src/styles.css`. Rebuild with `npm run build` after changing the UI so the backend-served `dist/` is updated.
