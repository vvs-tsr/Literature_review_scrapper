# Literature Review Scraper

An automated pipeline for harvesting, deduplicating, scoring, and exporting academic papers from the [OpenAlex](https://openalex.org/) API. Designed to produce a ranked list of the most relevant papers for a given research topic.

## How It Works

1. **Harvest** - queries OpenAlex for papers matching your keywords across a configurable time window
2. **Deduplicate** - merges duplicate results by DOI or title similarity
3. **Enrich** - attaches journal impact factors and SJR scores from a venue metrics database
4. **Score** - ranks each paper using a weighted multi-factor scoring model
5. **Export** - writes the top N papers to `.csv`, `.json`, and `.bib` files

## Tunable Parameters

All key parameters are defined at the top of `Scrapper.py`:

### Search Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `KEYWORDS` | `["neural architecture search", "AutoML", "automated machine learning"]` | List of search terms sent to OpenAlex. Add or remove terms to broaden/narrow results. |
| `YEARS_BACK` | `65` | How many years back to search (from current year). Set to `10` for recent work only. |
| `TOP_N` | `100` | Number of top-ranked papers to export. |

### Scoring Weights

The final score is a weighted sum of five sub-scores. Weights are in `WEIGHTS` and must sum to 1.0:

| Weight Key | Default | What It Measures |
|------------|---------|-----------------|
| `relevance` | `0.30` | Keyword presence in title and abstract; frequency and title-start bonus. |
| `impact` | `0.25` | Log-scaled citation count × venue prestige. |
| `recency` | `0.15` | Exponential decay with paper age. Foundational papers (>1000 citations, >20 years old) get a recency floor of `0.3`. |
| `pub_strength` | `0.20` | Normalised combination of impact factor, SJR score, venue h-index, and open-access status. |
| `velocity` | `0.10` | Citations per year (normalised). Favours fast-rising papers. |

**Example - prioritise recent work:**
```python
WEIGHTS = {
    'relevance':    0.30,
    'impact':       0.15,
    'recency':      0.30,   # increased
    'pub_strength': 0.15,
    'velocity':     0.10,
}
```

### API Politeness

In `harvest_openalex()`, set the `mailto` field to your email address so OpenAlex can contact you if your usage is unusual:
```python
"mailto": "your@email.edu"
```

## Setup

### Option 1 - Single click (Windows)
Double-click `install.bat`. It will create a virtual environment and install all dependencies automatically.

### Option 2 - Manual
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

## Running

```bash
python Scrapper.py
```

Output files (`top_100_papers.csv`, `top_100_papers.json`, `top_100_papers.bib`) are written to the project directory.

## Dependencies

See [requirements.txt](requirements.txt) for the full list. Key packages:

- `pyalex` - OpenAlex API client
- `pandas` / `polars` - data manipulation
- `scikit-learn` - TF-IDF vectorisation
- `rapidfuzz` - fuzzy title deduplication
- `bibtexparser` - BibTeX export
- `scholarly` / `semanticscholar` / `habanero` - supplementary data sources
