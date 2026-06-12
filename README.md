# Resume Intelligence Engine

> A personal AI system that reads a job description and generates a tailored, ATS-optimized resume — as a PDF — in under 60 seconds.

No generic templates. No manual rewriting. Just structured personal data + a language model doing the heavy lifting.

---

## How It Works

The engine uses a **Profile Context Block (PCB)** architecture:

1. **You define yourself once** — in `data_pool.json`, you describe your skills, experience, projects, and career positioning in a structured format.
2. **You paste a job description** — the engine reads it at runtime.
3. **A prompt is constructed** — `prompt_builder.py` assembles your profile data + the JD into a precise instruction for the LLM.
4. **The LLM generates a resume** — targeted to the exact role, using only facts from your data pool.
5. **LaTeX → PDF** — the output is clean LaTeX that compiles to a professional PDF.

The cover letter variant (`cl_prompt_builder.py`) adds `motivation_data.json` — a second data source for personal narrative, tone, and origin stories. This keeps factual data and personal voice cleanly separated.

---

## File Structure

```
.
├── resume_generator.py        # Main orchestrator — run this for resumes
├── generate.py                # Core generation logic
├── prompt_builder.py          # PCB + resume prompt construction
├── cl_prompt_builder.py       # PCB + cover letter prompt construction
├── Cover Letter Maker/
│   └── cover_letter_generator.py  # Cover letter variant
├── data_pool.json             # YOUR candidate data (not in repo — use example)
├── data_pool.example.json     # Template — copy and fill this
├── motivation_data.json       # YOUR personal narrative (not in repo — use example)
├── motivation_data.example.json   # Template — copy and fill this
├── .env                       # YOUR API key (not in repo — use example)
├── .env.example               # Template — copy and fill this
└── requirements.txt           # Python dependencies
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure your API key

```bash
# Windows
copy .env.example .env

# Mac/Linux
cp .env.example .env
```

Open `.env` and paste your [DeepSeek API key](https://platform.deepseek.com).

### 3. Fill in your profile data

```bash
# Windows
copy data_pool.example.json data_pool.json

# Mac/Linux
cp data_pool.example.json data_pool.json
```

Edit `data_pool.json` — replace every placeholder with your real information. This is the single source of truth the engine draws from.

### 4. (Optional) Fill in motivation data for cover letters

```bash
cp motivation_data.example.json motivation_data.json
```

Edit `motivation_data.json` with your personal narrative, origin stories, and tone preferences.

---

## Run

### Resume Generator

```bash
python resume_generator.py
```

Paste the job description, type `END` on a new line, press Enter.

**Output:**
- `output.tex` — raw LaTeX resume
- `outputs/Company_Role.pdf` — compiled PDF (if LaTeX is installed)

### Cover Letter Generator

```bash
cd "Cover Letter Maker"
python cover_letter_generator.py
```

---

## PDF Compilation

Requires a local LaTeX distribution:

| OS | Tool |
|---|---|
| Windows | [MiKTeX](https://miktex.org) or [TeX Live](https://tug.org/texlive/) |
| Mac | `brew install --cask mactex` |
| Linux | `sudo apt install texlive-full` |

Without LaTeX installed locally, upload the generated `output.tex` to [Overleaf](https://overleaf.com) for free compilation.

---

## Design Philosophy

Most resume tools ask you to fill a form. This one asks you to define yourself.

The `data_pool.json` format is intentionally richer than a traditional resume — it captures *positioning*, *strengths*, *business relevance*, and *role tags* alongside raw facts. This gives the LLM enough context to make intelligent trade-offs: which projects to surface, which skills to emphasize, how to frame your impact — based on what the job description actually asks for.

The `motivation_data.json` is even more opinionated: it's a structure for your *voice*, not just your facts. It captures origin stories, values, what you're not, and how to adapt your narrative for different role types. The goal is a cover letter that sounds like you wrote it — not like you filled a template.

---

## Model

Default: `deepseek-chat` (DeepSeek-V3 / V4-Flash).  
Update `DEEPSEEK_MODEL` in your `.env` to switch models.

```env
# Fast and cheap — good for resume generation
DEEPSEEK_MODEL=deepseek-chat

# Slower, reasoning mode — useful for complex narrative tasks
DEEPSEEK_MODEL=deepseek-reasoner
```

---

## Requirements

```
python-dotenv
openai        # DeepSeek uses an OpenAI-compatible API
```
