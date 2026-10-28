# RIE Workbench — Design System

Tokens live in `design-tokens.css`. **Every value in `styles.css` must come from a
token — no raw hex, px, ms, or cubic-bezier values in rules.** This doc is the
component contract for the frontend rebuild; it maps tokens to the atom/component
layer and fixes the states and accessibility requirements.

## 1. Hierarchy (atomic, flattened for a 630-line app)

Atoms → Components → Sections. No organisms/templates — this is a single-page tool.

| Layer | Members |
|---|---|
| **Atoms** | `.button`, `.tag`, `.status-label`, `.status-dot`, `.check-box`, `kbd`, `.skip-link` |
| **Components** | `.job-form` (+textarea/footer), `.metric-card`, `.panel`, `.requirement`, `.profile-card`, `.generation-card`, `.job-run`, `.result-card`, `.code-preview`, `.empty-state`, `.health-pill`, `.message-region`, `.loading-line` |
| **Sections** | sidebar (nav/brand/foot), topbar, `#workbench`, `#match-report`, `#profile`, `#runs`, footer |

## 2. Token map

| Token group | Used by |
|---|---|
| `--accent-*` | primary buttons, links, active nav, focus outlines, progress bar, brand em, evidence paths |
| `--ink-*` | surfaces: body `--ink-950`, cards `--ink-850-soft`, raised `--ink-850-strong`, recessed `--ink-800` |
| `--cream-*` | text: `100` strongest → `500` labels → `600/700` faintest. **Headings/cards use cream-200 minimum for contrast.** |
| `--ok/--warn/--err` | status semantics only: matched/partial/missing, offline, failed runs. Never decorative. |
| `--line` | all hairlines; `--line-strong` for primary container borders |
| `--space-*` | all padding/gap/margins. `--space-section` between page sections. |
| `--text-*` / `--font-*` | display serif (`--font-display`) for h1/h2/brand/metric values; `--font-body` everywhere else; `--font-mono` in `.code-preview` |
| `--radius-*` | `--radius-xl` containers, `--radius-lg` panels, `--radius-md` controls, `--radius-badge` badges, `--radius-pill` health pill |

## 3. Component contracts

### Button (atom, 3 variants × 2 sizes × 4 states)
- Sizes: `--control-h` default; wide variant min-width 161px or full-width (<570px)
- Variants: `--primary` (accent fill + `--accent-contrast` text), `--quiet` (transparent + `--line` border)
- States: hover/accent-400, disabled opacity .58 + wait cursor, focus-visible 2px accent outline offset 3px
- Busy pattern (JS): text swaps to busy label, stays disabled until promise settles

### Status labels + dots
- `--matched` → ok; `--partial/--review` → warn; `--missing` → err
- Dot variants: default ok, `--quiet` accent, `--loading` warn+pulse, `--offline/--error` err
- Dots always pair a 3px soft ring (`--shadow-focus`-style tint)

### Panels (`metric-card`, `.panel`, `.requirement`, `.job-run`, `.result-card`)
- Padding `--space-4/‑5`; border `--line`; raise with `--ink-850*` fills, NOT extra borders
- One warm tint allowed per view (`.metric-card--warm`, `.generation-card` gradient)

### Code preview
- `--font-mono` `--text-xs`/1.6, `--ink-850` fill, max-height 360px, error state → `--err-500` text

## 4. States & accessibility (non-negotiable)

- Focus: `:focus-visible` 2px accent outline, offset 3px — every interactive element
- Text: labels ≥ `--text-xs` in cream-500 minimum; body data in cream-400/200 on cream bg never < 4.5:1
- `prefers-reduced-motion`: kill pulse/shimmer/scroll-smooth (existing block stays)
- Targets: nav links 42px, buttons 37px+; `--poll-interval` 1600ms only via JS timers
- Live regions: `#global-message` assertive; health/status polite

## 5. Responsiveness

| Query | Change |
|---|---|
| ≤ 850px (`--bp-narrow`) | sidebar collapses to 72px rail (icons only); panels stack 1-col; generation card 2-col |
| ≤ 570px (`--bp-phone`) | sidebar → fixed bottom bar; h1 scales 13vw; action stacks; result cards 1-col |

## 6. Rebuild rules for `styles.css` / `app.js`

1. `index.html` loads `design-tokens.css` **before** `styles.css`
2. `styles.css` re-expressed against tokens only — same DOM, same class names, same behavior
3. `app.js` keeps: fetch API layer, polling, draft persistence, DOM-safe rendering
   (`create()`/`textContent` — XSS-safe), keyboard shortcuts, busy-state pattern
4. One `innerHTML` allowance: the fixed "Reading the role…" placeholder at `app.js:299`
   (constant string, no interpolation proven safe in audit) — convert to `create()`
   calls anyway so zero `innerHTML` remains
5. Bump `?v=` cache-busters on both CSS files and `app.js` in `index.html`
