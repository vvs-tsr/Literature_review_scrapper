import requests
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# ============ CONFIGURATION ============
KEYWORDS = ["neural architecture search", "AutoML", "automated machine learning"]
YEARS_BACK = 65  # From 1961 to 2026
TOP_N = 100
WEIGHTS = {
    'relevance': 0.30,
    'impact': 0.25,
    'recency': 0.15,
    'pub_strength': 0.20,
    'velocity': 0.10
}

# ============ 1. DATA HARVESTING ============
def harvest_openalex(keywords, years_back):
    """OpenAlex is free and has excellent coverage."""
    base_url = "https://api.openalex.org/works"
    papers = []
    
    for keyword in keywords:
        cursor = "*"
        while cursor:
            params = {
                "search": keyword,
                "filter": f"publication_year:>{datetime.now().year - years_back}",
                "per_page": 200,
                "cursor": cursor,
                "mailto": "your@email.edu"  # Be polite to their servers
            }
            resp = requests.get(base_url, params=params, timeout=30)
            data = resp.json()
            
            for work in data.get("results", []):
                papers.append({
                    "id": work.get("id"),
                    "title": work.get("display_name"),
                    "authors": [a["author"]["display_name"] for a in work.get("authorships", [])],
                    "abstract": work.get("abstract", ""),
                    "publication_date": work.get("publication_date"),
                    "venue": work.get("host_venue", {}).get("display_name", ""),
                    "venue_type": work.get("type", ""),
                    "cited_by_count": work.get("cited_by_count", 0),
                    "concepts": [c["display_name"] for c in work.get("concepts", [])],
                    "doi": work.get("doi"),
                    "open_access": work.get("open_access", {}).get("is_oa", False),
                    "biblio": work.get("biblio", {}),
                    "keywords_matched": [keyword]
                })
            
            cursor = data.get("meta", {}).get("next_cursor")
            if not cursor or len(papers) > 50000:  # Safety limit
                break
    
    return papers

# ============ 2. DEDUPLICATION ============
def deduplicate(papers):
    """Merge duplicates by DOI or title similarity."""
    seen = {}
    for paper in papers:
        key = paper.get("doi") or paper["title"].lower().strip()
        if key in seen:
            # Merge keyword matches
            seen[key]["keywords_matched"] = list(set(
                seen[key]["keywords_matched"] + paper["keywords_matched"]
            ))
            # Keep higher citation count
            seen[key]["cited_by_count"] = max(
                seen[key]["cited_by_count"], 
                paper["cited_by_count"]
            )
        else:
            seen[key] = paper
    return list(seen.values())

# ============ 3. ENRICHMENT ============
def enrich_with_impact_factors(papers):
    """Map venues to impact factors using a local database or CrossRef."""
    # You'd load a mapping: venue_name -> {impact_factor, sjr, h_index}
    # Sources: Scimago, Journal Citation Reports, or CrossRef
    for paper in papers:
        venue = paper.get("venue", "")
        # Lookup in your venue metrics database
        metrics = VENUE_METRICS_DB.get(venue, {})
        paper["impact_factor"] = metrics.get("if", 0)
        paper["sjr_score"] = metrics.get("sjr", 0)
        paper["h_index_venue"] = metrics.get("h_index", 0)
    return papers

# ============ 4. SCORING ============
def calculate_scores(papers, keywords, weights):
    current_year = datetime.now().year
    
    for paper in papers:
        # --- Relevance Score ---
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        
        relevance = 0
        for kw in paper.get("keywords_matched", []):
            kw_lower = kw.lower()
            if kw_lower in title:
                relevance += 0.8
                if title.startswith(kw_lower):
                    relevance += 0.2  # Title-start bonus
            if kw_lower in abstract:
                relevance += 0.4
                # Count frequency
                relevance += 0.1 * abstract.count(kw_lower)
        
        # --- Impact Score ---
        citations = paper.get("cited_by_count", 0)
        venue_prestige = get_venue_prestige(paper.get("venue", ""), paper.get("venue_type", ""))
        impact = np.log1p(citations) * venue_prestige
        
        # --- Recency Score ---
        pub_year = int(str(paper.get("publication_date", ""))[:4]) if paper.get("publication_date") else 2000
        years_old = max(0, current_year - pub_year)
        recency = np.exp(-0.1 * years_old)
        
        # Foundational paper protection
        if citations > 1000 and years_old > 20:
            recency = max(recency, 0.3)
        
        # --- Publication Strength ---
        if_max = max(p.get("impact_factor", 0) for p in papers) or 1
        sjr_max = max(p.get("sjr_score", 0) for p in papers) or 1
        h_max = max(p.get("h_index_venue", 0) for p in papers) or 1
        
        pub_strength = (
            0.4 * (paper.get("impact_factor", 0) / if_max) +
            0.3 * (paper.get("sjr_score", 0) / sjr_max) +
            0.2 * (paper.get("h_index_venue", 0) / h_max) +
            0.1 * (1 if paper.get("open_access") else 0)
        )
        
        # --- Citation Velocity ---
        velocity = citations / max(1, years_old)
        vel_max = max(p.get("cited_by_count", 0) / max(1, current_year - int(str(p.get("publication_date", ""))[:4])) for p in papers) or 1
        velocity = velocity / vel_max
        
        # --- Final Weighted Score ---
        paper["score_relevance"] = relevance
        paper["score_impact"] = impact
        paper["score_recency"] = recency
        paper["score_pub_strength"] = pub_strength
        paper["score_velocity"] = velocity
        
        paper["total_score"] = (
            weights["relevance"] * relevance +
            weights["impact"] * impact +
            weights["recency"] * recency +
            weights["pub_strength"] * pub_strength +
            weights["velocity"] * velocity
        )
    
    return papers

# ============ 5. EXPORT ============
def export_top_papers(papers, n=100):
    df = pd.DataFrame(papers)
    df = df.sort_values("total_score", ascending=False).head(n)
    
    # Export to BibTeX, CSV, JSON
    df.to_csv("top_100_papers.csv", index=False)
    df.to_json("top_100_papers.json", orient="records", indent=2)
    
    # Generate BibTeX
    bibtex_entries = []
    for _, row in df.iterrows():
        entry = f"""@article{{{row.get('doi', 'unknown').replace('/', '_')},
    title = {{{row['title']}}},
    author = {{{' and '.join(row.get('authors', []))}}},
    journal = {{{row.get('venue', '')}}},
    year = {{{str(row.get('publication_date', ''))[:4]}}},
    doi = {{{row.get('doi', '')}}}
}}"""
        bibtex_entries.append(entry)
    
    with open("top_100_papers.bib", "w") as f:
        f.write("\n\n".join(bibtex_entries))
    
    return df

# ============ MAIN PIPELINE ============
if __name__ == "__main__":
    # 1. Harvest
    raw_papers = harvest_openalex(KEYWORDS, YEARS_BACK)
    
    # 2. Deduplicate
    unique_papers = deduplicate(raw_papers)
    
    # 3. Enrich
    enriched_papers = enrich_with_impact_factors(unique_papers)
    
    # 4. Score
    scored_papers = calculate_scores(enriched_papers, KEYWORDS, WEIGHTS)
    
    # 5. Export top 100
    top_100 = export_top_papers(scored_papers, TOP_N)
    print(f"Exported top {len(top_100)} papers")