# Resume Intelligence Engine

> A personal AI system that reads a job description and generates a tailored, evidence-backed resume — as editable LaTeX and optionally a PDF.

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
├── ai_config.py               # Shared local OpenRouter configuration
├── job_matcher.py             # Local explainable JD-to-evidence triage
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

Open `.env` and paste your [OpenRouter API key](https://openrouter.ai/keys), then set an available OpenRouter model slug.

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

### Local job triage

Analyze a saved job-description text file before making an AI generation call:

```bash
python job_matcher.py --jd-file path/to/job-description.txt --output match-report.json
```

The report identifies role-family signals, explicit skills, evidence status,
experience gates, compensation mentions, eligibility questions, and the
role-specific resume evidence plan. The local selector suppresses unrelated
experience (for example, a podcast internship for a technical JD) before the
drafting model sees the profile. It is not an ATS score and does not submit an
application.

### Local ATS-oriented validation

Validate a generated LaTeX resume without uploading private profile data to a
third-party scanner:

```bash
python ats_validator.py --resume output.tex --jd-file path/to/job-description.txt
```

The validator checks text-safe structure, standard headings, one-column layout,
parser-hostile elements, package compatibility and overlap with the local
matcher’s detected job skills. It is an explainable parser-safety check, not a
universal ATS score or hiring prediction.

### Resume Generator

```bash
python resume_generator.py
```

Paste the job description, type `END` on a new line, press Enter.

**Output:**
- `output.tex` — raw LaTeX resume
- `outputs/Company_Role.pdf` — compiled PDF (if LaTeX is installed)

### Parallel application packet generation

Use `generate.py` when you want the resume and cover letter generated
concurrently from the same job description:

```bash
python generate.py
```

`resume_generator.py` remains the standalone resume-only path. Both paths use
the shared OpenRouter model configuration. The recommended DeepSeek setup is:

```env
OPENROUTER_MODEL=deepseek/deepseek-v4.1-flash-1017
OPENROUTER_PROVIDER_SORT=throughput
OPENROUTER_PROVIDER_ALLOW_FALLBACKS=true
OPENROUTER_PROVIDER_MAX_PROMPT_PRICE=0.10
OPENROUTER_PROVIDER_MAX_COMPLETION_PRICE=0.60
OPENROUTER_REASONING_EFFORT=none
```

The provider policy asks OpenRouter for a high-throughput endpoint under the
configured price ceiling and permits fallback when an endpoint is unavailable.
The selected routing policy is printed at startup, and the combined summary
still reports separate CV/cover-letter timing and token usage.

Resume drafting is constrained to one page: at most two experience entries,
three projects, two certifications, three skill rows, and two bullets per
experience/project entry. Lower-scoring evidence is omitted rather than pushed
onto a second page.

### Cover Letter Generator

```bash
cd "Cover Letter Maker"
python cover_letter_generator.py
```

### Local web workbench

Run the dependency-light local frontend and API from the project root:

```bash
python app.py --host 127.0.0.1 --port 8000
```

Then open `http://127.0.0.1:8000`. The workbench provides local job triage,
minimal profile context, bounded generation jobs and links to versioned local
run artifacts. AI generation is always user-triggered; the server does not call
OpenRouter during startup. Completed manifests are recovered from `runs/` when
the server restarts, while private job-description payloads are not restored.
If all retained jobs are still active, new generation requests receive a clear
queue-full response instead of silently evicting an active job. Completed
resume and cover-letter LaTeX can be previewed and copied directly inside the
workbench; downloading is optional. The raw `.tex` links are served as
`text/plain` with an inline disposition so browsers open them as readable text
instead of treating them as downloads.

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

## Provider and model

The local generators use OpenRouter's OpenAI-compatible API. This keeps the
workflow local while allowing the selected model to be changed without changing
the generation code. The actual model and current pricing are controlled by
`OPENROUTER_MODEL` and the OpenRouter account/catalog, not hard-coded in this
repository.

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=deepseek/deepseek-v4.1-flash-1017
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

---

## Requirements

```
python-dotenv
openai        # OpenRouter exposes an OpenAI-compatible API
```
