# StreamSense console design system

## Direction

- Purpose: a high-density technical demonstration for multimodal inference, evidence, and routing.
- Tone: editorial observability console, informed by state timelines and trace-oriented dashboards.
- Constraint: a single static HTML asset, no Node build or external font/network dependency.
- Memory point: the evidence timeline uses route-colored nodes to make model escalation visible.

## Tokens

- Canvas `#080B0D`, surface `#101519`, raised surface `#171E23`.
- Primary text `#EEF5F4`, muted text `#9AABA9`.
- Grounded/lightweight accent `#51E5CE`; escalation accent `#FFBD59`; error `#FF6B6B`.
- Display/body: IBM Plex Sans or Noto Sans SC; telemetry: IBM Plex Mono or Cascadia Code.
- Spacing follows a 4 px base; panel radius is 10 px; controls use 5 px radii.

## Accessibility and audit

- Keyboard-visible focus, semantic forms, live status regions, labels, and non-color route text.
- Responsive layouts verified at 560/900 px breakpoints; motion disabled via reduced-motion media query.
- Mechanical frontend audit: 6/6 layout rules passed and AI-tell lint returned CLEAN.
- Chromium screenshot at 1440 x 1100 is committed as `docs/assets/dashboard.png`.

