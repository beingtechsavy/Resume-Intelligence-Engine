# Cover Letter Intelligence Engine

AI-powered cover letter generator built for mass-producing personalized, human-sounding cover letters for scholarships and job applications.

## Architecture

Mirrors the Resume Maker system with cover-letter-specific prompt engineering and anti-AI-detection strategy.

```
Cover Letter Maker/
├── .env / .env.example     → API config (or falls back to Resume Maker's .env)
├── motivation_data.json    → Your personal voice, values, stories (anti-AI-detection)
├── cl_prompt_builder.py    → Cover Letter PCB + output contract
├── cover_letter_generator.py → Main orchestrator
├── output.tex              → Last generated LaTeX (runtime)
└── outputs/                → Compiled PDFs: Org_Position.pdf
```

**Profile data** (`data_pool.json`) is read from `../Resume Maker/` — single source of truth, zero duplication.

## Quick Start

```powershell
# 1. Set up API key (or copy from Resume Maker)
copy "..\Resume Maker\.env" .env

# 2. Install deps (same as Resume Maker)
pip install -r requirements.txt

# 3. Run
python cover_letter_generator.py
```

## How It Works

1. Paste the JD / scholarship description → type `END`
2. (First time or if no `motivation_data.json`) Type 2-3 personal sentences → `END` or `SKIP`
3. Engine generates a tailored LaTeX cover letter
4. Auto-compiles to PDF if pdflatex is available
5. Output → `outputs/Org_Position.pdf`

## Anti-AI-Detection (Built-In)

The system uses a 5-layer strategy:

1. **Prompt-level** — banned phrases, varied sentence structure, mandatory contractions
2. **Structural** — non-standard paragraph layout, rhetorical questions, sentence fragments
3. **Seed sentences** — your real words woven into the output (strongest single technique)
4. **motivation_data.json** — your authentic voice, stories, values
5. **Human pass** — spend 5 min reading aloud and tweaking (always recommended)

## Files You Should Edit

| File | When |
|------|------|
| `motivation_data.json` | Once — write your personal stories and values in YOUR voice |
| `../Resume Maker/data_pool.json` | When your profile changes (new project, new skill, etc.) |
| `.env` | Once — your API key |

## Cost

Pricing is model-dependent; check the selected OpenRouter model and account usage.
