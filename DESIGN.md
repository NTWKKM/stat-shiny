# 🎨 Design System & Guidelines (DESIGN.md)

This document outlines the **Minimal Slate-Monochrome Design System** for the Medical Statistical Tool. It details design tokens, component states, layouts, and accessibility requirements.

---

## 1. Design Tokens

### Colors (Slate-Monochrome Palette)

| Name | Hex | Role | Usage |
| :--- | :--- | :--- | :--- |
| `primary` | `#0F172A` | Slate 900 | Main titles, primary action buttons, active navigation markers |
| `primary_dark` | `#020617` | Slate 950 | Darkest slate tone for highlights |
| `primary_light`| `#F8FAFC` | Slate 50 | Light gray/slate backgrounds, card/table row hover highlights |
| `secondary` | `#64748B` | Slate 500 | Secondary text, captions, outline button borders |
| `smoke_white` | `#F8FAFC` | Slate 50 | Empty states, secondary panels background |
| `text` | `#0F172A` | Slate 900 | Primary body text, labels, button text |
| `text_secondary`| `#64748B` | Slate 500 | Muted help copy, secondary metrics, subtitles |
| `border` | `#E2E8F0` | Slate 200 | Clean, thin borders for inputs, cards, and tables |
| `background` | `#FAFAFA` | Off-White | Page background, dashboard shell |
| `surface` | `#FFFFFF` | White | Cards, modal content, results container background |

#### Soft Semantic States

- **Success**: `#059669` (Emerald 600) — Validation passed, successful calculation, check badges.
- **Danger**: `#DC2626` (Red 600) — Error alerts, excluded data indicators, risk flags.
- **Warning**: `#D97706` (Amber 600) — Assumptions violated, caveats, warning badges.
- **Info**: `#475569` (Slate 600) — Informational panels, helpful annotations.

---

### Typography

- **Base Font Family**: `'Inter', -apple-system, BlinkMacSystemFont, sans-serif` — Refined, professional sans-serif.
- **Monospace Font Family**: `'Courier New', monospace` — For CI tables, P-values, statistical outputs.
- **Weights**:
  - `300` (Light) — Sub-headers, display headings.
  - `400` (Regular) — Body copy, field labels, paragraph text.
  - `500` (Medium) — Active states, table headers, buttons, key metrics.
  - `600` (Semi-bold) — Main component headers, strong emphasis.

#### Scale

- **Display**: `30px` (line-height: `1.2`, letter-spacing: `-0.02em`)
- **Heading (H2)**: `22px` (line-height: `1.3`, letter-spacing: `-0.01em`)
- **Subheading (H3/H4)**: `16px` (line-height: `1.4`)
- **Body**: `14px` (line-height: `1.6`)
- **Caption / Code**: `12px` / `13px` (line-height: `1.4/1.5`)

---

### Spacing & Grid System

Based on an 8px modular scale:
- `var(--spacing-2xs)`: `2px`
- `var(--spacing-xs)`: `4px`
- `var(--spacing-sm)`: `8px`
- `var(--spacing-1-5sm)`: `12px`
- `var(--spacing-md)`: `16px`
- `var(--spacing-lg)`: `20px`
- `var(--spacing-xl)`: `32px`
- `var(--spacing-2xl)`: `48px`

---

## 2. UI Components & States

### Cards

- **Role**: Groups input fields, filters, and statistical outputs.
- **Styling**: `border: 1px solid var(--color-border); border-radius: 8px; box-shadow: none;`
- **States**:
  - *Idle*: Flat card layout.
  - *Hover* (Standard): Very faint border change to Slate 300, no shadows.
  - *Hover* (Feature Cards): Border turns to Slate 900 (`#0F172A`).
  - *Card Header*: Transparent background with thin bottom border (`1px solid var(--color-border)`). Lighter weight (`500`) titles.

### Buttons

- **Primary**: Flat Slate 900 (`#0F172A`) background with white text. Hover state darkens background.
- **Secondary**: Transparent background with Slate 500 (`#64748B`) border and Slate 900 text. Hover gets faint slate light background.
- **States**:
  - *Default*: Flat presentation, no active translates.
  - *Hover*: Soft transition in background opacity.
  - *Loading*: Spin animation replaces text (btn-loading class).
  - *Disabled*: `opacity: 0.6`, cursor not allowed.

### Form Inputs & Dropdowns

- **Role**: Selectize fields, number inputs, variables dropdown.
- **Styling**: Apple/Stripe-like. Background is Slate 50 (`#F8FAFC`), thin borders (`#E2E8F0`), border-radius `6px`.
- **States**:
  - *Idle*: Off-white input, secondary gray placeholder text.
  - *Focus*: Fades background to white, border to Slate 900, thin `2px` focus outline `rgba(15, 23, 42, 0.15)`.
  - *Disabled*: Gray background (`#F8FAFC`), muted secondary text, cursor blocked.

### Navigation Header

- **Role**: Fluid page-level navbar links.
- **Styling**: Transparent header background, thin bottom border.
- **Active State**: Inactive text is Slate 500. Active text turns to Slate 900 with a clean `2px` bottom border underline indicator (no dark navy pills).

### Tables

- **Role**: Displays model metrics, coefficients, hazard ratios.
- **Styling**: Generous padding (`10px 16px`), cell-only bottom borders, clean transparent header background, font-weight `500` headers.
- **Hover**: Table row hover highlights with `var(--color-primary-light)` (`#F8FAFC`).

---

## 3. Accessibility & Constraints

1. **Contrast Ratios**: All text elements must achieve a minimum contrast ratio of `4.5:1` against their respective backgrounds (Slate 900 text on off-white or white backgrounds ensures this).
2. **Motion Preference**: Layout animations (like fade-in entries) must respect `@media (prefers-reduced-motion: reduce)` by disabling transitions.
3. **Keyboard Focus Indicators**: Every interactive control must show a standard high-visibility outline state on `:focus-visible` (3px outline offset by 2px in Slate 900).
4. **Skip Links**: Retain screen-reader landmarks and page skip-links for keyboard-only navigation.
