# Food-Sure — Web App

A React + Tailwind + Framer Motion frontend for the Food-Sure backend.
Four pages (Home, Analyze, How it Works, Scan History), dark theme,
scroll/entry animations throughout.

## Requirements

- [Node.js](https://nodejs.org/) 18 or newer (this was built and tested on Node 22).
  Check with `node -v`. If you don't have it, download the LTS installer from
  nodejs.org — it's a normal Windows installer, nothing exotic.

## Run it (development)

```
npm install
npm run dev
```

This starts a dev server, normally at `http://127.0.0.1:5173`. Open that in your
browser. Hot-reloads as you edit.

## Run it (production build)

```
npm run build
npm run preview
```

`npm run build` outputs a static site into `dist/`. `npm run preview` serves that
build locally so you can check it behaves the same as it will once deployed.
You can also deploy the `dist/` folder as-is to any static host (Vercel, Netlify,
GitHub Pages, etc.) if you want a public link for your defense.

## Connecting to the backend

Make sure the FastAPI backend is running first:

```
cd ../backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

`--host 0.0.0.0` lets a phone on the same Wi-Fi reach it; open the "Network:"
address Vite prints (e.g. `http://192.168.x.x:5173`) on the phone. Windows
Firewall must allow inbound TCP 8000.

By default this app talks to port 8000 on the machine that served the page. To point it somewhere else
(e.g. a deployed backend), create a `.env` file in this folder:

```
VITE_API_BASE=https://your-backend-url.example.com
```

## Project structure

```
src/
  pages/            Home, Analyze, HowItWorks, History — one file per route
  components/        Reusable UI: NavBar, Dropzone, ResultPanel, ModelCompareChart, etc.
  context/            HistoryContext — in-memory scan history for the session
  lib/
    api.js            All backend calls (/health, /predict)
    format.js          Spice list, severity labels, class-label formatting
    modelData.js       Real 5-model accuracy comparison (from final_comparison_summary.csv)
    useHealthPing.js   Polls /health for the "backend online/offline" indicator
```

## Design notes

- Dark theme with two brand accents — lime (primary/CTA) and amber (secondary/data) —
  chosen specifically to avoid the generic purple-gradient "AI product" look.
- Severity colors (PURE/LOW/MEDIUM/HIGH) use the dataviz skill's pre-validated
  status palette (contrast + colorblind-safe), always paired with an icon and
  text label, never color alone.
- Scroll-reveal, page transitions, button press feedback, and the top-3
  prediction bars all follow a "does this animation earn its keep" rule —
  nothing animates purely for decoration, everything stays under ~300ms for
  UI feedback, and reduced-motion is respected.
- The model comparison chart on the How It Works page uses real accuracy
  figures from your training run (`final_comparison_summary.csv`) — update
  `src/lib/modelData.js` if you retrain.
- Routing uses `HashRouter` (URLs look like `/#/analyze`) specifically so the
  production build works from any static host or even a plain file server
  without needing server-side rewrite rules.

## If something looks off

- "backend offline" badge in the nav: start the backend (see above) — the app
  polls `/health` every 15s.
- Blank page after `npm run build` + opening `dist/index.html` directly: don't
  open the file directly, serve it (`npm run preview`, or `python -m http.server`
  from inside `dist/`) — browsers block ES module scripts from `file://` pages.
