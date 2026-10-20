# Automated Paper Ingestion Pipeline Plan

**Date:** 2026-07-13  
**Scope:** GPU training performance engineering  
**Target:** two or three papers per digest, every two days, within a three-hour weekly reading budget

## Decision

Start with a **thin local scheduled digest** that reuses official metadata APIs and keeps Zotero as the durable paper library. The local code should fetch, normalize, deduplicate, filter, rank, and render a short digest. It should not crawl websites, download PDFs, build a vector database, or replace ResearchRabbit.

Run a four-week pilot before adding learned local ranking or a larger web application.

## What the Two Reports Contribute

The earlier Codex response provides the cleaner operating model:

- arXiv is the regular source.
- Conference proceedings are event-driven sources.
- ResearchRabbit is for citation-graph expansion after a paper passes triage.
- Project notes hold the conclusions that matter for DINOv3 work.

The supplied deep-research report adds a useful venue taxonomy and a concrete weekly reading routine. Its calendar should not become pipeline configuration. At least two calendar claims are stale or inconsistent with official venue pages, so venue dates need separate, replaceable configuration rather than application logic.

Neither report fully specifies automated deduplication, feedback ownership, source failures, or false-negative checks. Those are the main additions in this plan.

## Reuse Boundary

| Need | Reuse | Local code required |
|---|---|---|
| Fresh broad metadata | [arXiv API](https://info.arxiv.org/help/api/user-manual.html) | Query configuration and Atom normalization |
| Curated daily signal | [Hugging Face Hub paper methods](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api) | Adapter and version probe |
| Feedback-based discovery | [Semantic Scholar Recommendations API](https://api.semanticscholar.org/api-docs/recommendations) | ID resolution and score blending |
| Durable personal library | [Zotero feeds, collections, tags, and saved searches](https://www.zotero.org/support/quick_start_guide) | Import file generation only |
| Citation-graph expansion | [ResearchRabbit Zotero importer](https://learn.researchrabbit.ai/en/articles/12796541-using-the-zotero-importer) | None in version 1 |
| Optional local learned UI | [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite) | Deployment and integration work |

ResearchRabbit currently imports Zotero collections and exports BibTeX, RIS, or CSV. Its current Zotero workflow is not documented as two-way synchronization. The pipeline should therefore treat Zotero as the system of record and ResearchRabbit as a downstream discovery tool.

## Proposal A: Zotero-First Feed Hub

### Shape

```text
arXiv RSS or custom Atom feeds
            |
            v
      Zotero feed inbox
            |
      manual tags/searches
            |
            v
   Zotero kept collection
            |
            v
      ResearchRabbit
```

### Workflow

1. Subscribe Zotero to narrow arXiv feeds.
2. Review feed items using `paper-inbox`, `paper-keep`, `paper-defer`, and `paper-reject` tags.
3. Save selected items to a `GPU Performance` collection.
4. Import that collection into ResearchRabbit once per week.
5. Create a project reading note only for deep-read papers.

### Tradeoffs

- **Setup:** one or two hours.
- **Maintenance:** very low.
- **Ranking:** feed order and Zotero saved searches only.
- **Digest cap:** manual. It will not reliably produce exactly two or three strong papers.
- **Best fit:** choose this if any local service is more work than the expected time savings.

## Proposal B: Thin Scheduled Digest

**Recommendation for the four-week pilot.**

### Shape

```text
 arXiv API       HF papers       Semantic Scholar
     |               |                  |
     +---------------+------------------+
                     |
                     v
          normalize + canonical IDs
                     |
                     v
        deduplicate + hard filters
                     |
                     v
          transparent static score
                     |
          optional top-10 rerank
                     |
                     v
       0 to 3 paper Markdown/HTML digest
                     |
             keep / defer / reject
                     |
                     v
          Zotero tags and collection
                     |
                     v
              ResearchRabbit
```

### Components

| Component | Version 1 choice | Reason |
|---|---|---|
| Scheduler | `systemd --user` timer, every two days | Persistent logs and missed-run handling; cron is an acceptable fallback |
| Runtime | Small Python package or script | All source SDKs and XML parsing are available in Python |
| Configuration | One YAML file | Queries, categories, negative terms, source weights, score threshold |
| Fetch/cache state | SQLite | Enough for a personal metadata pipeline; no server required |
| Durable decisions | Zotero tags and collections | Prevents SQLite and Zotero from becoming competing libraries |
| Output | Dated Markdown plus self-contained HTML | Fast skim, durable local record, no web server |
| ResearchRabbit handoff | Zotero collection import | Uses the supported product workflow |

### Source policy

1. **arXiv:** primary broad source. Query `cs.DC`, `cs.PF`, and filtered `cs.LG`; add conditional `cs.AR`, `cs.PL`, and `cs.CV` terms when they concern training systems or performance.
2. **Hugging Face:** secondary popularity and code-community signal. At startup, probe the installed `huggingface_hub` version. Current official docs expose `list_daily_papers` with date, week, month, and sort controls plus `list_papers` for query search. Fall back to `list_papers` when the installed version lacks the daily method.
3. **Semantic Scholar:** recommendation signal from a small positive and negative seed set. Resolve arXiv IDs or DOIs to Semantic Scholar IDs, cache successful mappings, and skip uncertain title-only mappings.
4. **Proceedings:** event-driven adapter or manual import. Keep venue URLs and active dates in configuration, not code.

The arXiv adapter should cache responses, make at most one equivalent query per day, use small result pages, and respect the documented delay for repeated calls.

### Identity and deduplication

Store these fields for every candidate:

- `canonical_arxiv_id` without the version suffix
- `arxiv_version`, `published_at`, `updated_at`, and `first_seen_at`
- DOI when present
- Semantic Scholar paper ID when confidently resolved
- normalized title hash as a last-resort duplicate key
- source list and source-specific timestamps

Default update rule: a new arXiv version does not re-enter the digest unless its title or abstract changes materially and the previous decision was `defer`. This avoids resurfacing formatting-only revisions.

### Filtering and scoring

Use two stages.

**Stage 1: hard rules**

- Accept configured categories and date window.
- Require at least one positive topic family: distributed training, communication, compiler/runtime, input pipeline, memory, kernels, profiling, measurement, or training-system architecture.
- Exclude application-only papers with no systems or performance contribution.
- Exclude already decided papers and exact duplicates.

**Stage 2: transparent score**

```text
score = topic_match
      + source_agreement
      + semantic_scholar_recommendation
      + hugging_face_signal
      + recency
      - negative_topic_penalty
      - near_duplicate_penalty
```

Do not score code or artifacts in version 1 unless a source adapter supplies structured artifact metadata. Do not infer it from abstract prose.

An LLM reranker is optional and restricted to the top 10 to 20 abstracts after static filtering. Its output must include a short reason tied to configured interests. The digest remains usable when the reranker is unavailable.

### Digest contract

Produce zero to three papers. Do not fill a quota with low-scoring items.

Each card contains:

- title, authors, date, and canonical links
- source provenance
- three-sentence maximum abstract compression
- why it may matter for DINOv3 performance work
- one likely limitation or reason to skip
- `keep`, `defer`, and `reject` instructions

Friday's output is a weekly rollup of kept and deferred papers, not another independent candidate batch.

### Failure semantics

- Run under a lock so timer overlap cannot create duplicate digests.
- Use bounded retries with backoff for each source.
- Preserve the last successful response and timestamp per source.
- Continue when one source fails; mark the digest with stale or missing provenance.
- Generate a digest idempotently from run date and candidate IDs.
- Log source counts, filter counts, selected IDs, errors, and runtime.
- Alert only after two consecutive failed runs or a source remains stale for more than its expected update interval.
- Keep API keys outside the repository; prefer unauthenticated endpoints where their limits are sufficient.

### Weekly human loop

| Activity | Time |
|---|---:|
| Four short digest reviews | 40 minutes |
| One or two deeper paper reads | 100 minutes |
| ResearchRabbit expansion from kept papers | 20 minutes |
| Project note and pipeline feedback | 20 minutes |
| **Total** | **180 minutes** |

## Proposal C: Self-Hosted arxiv-sanity-lite

### Shape

Reuse arxiv-sanity-lite's arXiv poller, abstract TF-IDF features, per-tag SVM ranking, tagging UI, and recommendation email process.

### Tradeoffs

- **Setup:** half a day to several days, depending on deployment expectations.
- **Maintenance:** highest of the three proposals.
- **Ranking:** learned local ranking from tags.
- **Coverage:** arXiv-centric without custom adapters.
- **Integration:** still needs Zotero export, ResearchRabbit handoff, and project-note conventions.
- **Best fit:** only after the pilot shows that static ranking is inadequate and a persistent browser UI is worth operating.

This proposal reuses an existing project, but it still creates more operational work than the target workflow requires. It should remain an exit option, not the starting point.

## Comparison

| Criterion | A. Zotero-first | B. Thin digest | C. arxiv-sanity-lite |
|---|---:|---:|---:|
| Initial work | Low | Medium | High |
| Ongoing maintenance | Low | Low to medium | Medium to high |
| Automatic two-to-three-paper cap | No | Yes | Yes |
| Multi-source ranking | No | Yes | No, without extensions |
| Feedback learning | Manual saved searches | S2 feedback, optional rerank | Per-tag SVM |
| False-negative audit | Manual | Built into pilot | Must be added |
| Zotero fit | Native | System of record | Export integration needed |
| Recommendation | Fallback | **Start here** | Defer |

## Four-Week Pilot

### Week 0: configuration

- Define category queries, positive topic families, and negative terms.
- Select three to five positive seed papers and three to five clear negatives.
- Create Zotero tags and the `GPU Performance` collection.
- Choose the digest output directory and timer schedule.

### Week 1: minimum useful loop

- Implement arXiv fetch, canonical identity, deduplication, hard filters, static score, SQLite cache, and Markdown/HTML rendering.
- Record decisions in Zotero. A small local decision export may mirror those tags for scoring, but Zotero remains authoritative.

### Week 2: secondary signals

- Add Hugging Face with dependency probing.
- Add Semantic Scholar ID resolution and recommendations.
- Add independent retries, source freshness, and provenance.

### Weeks 3 and 4: tune and audit

- Tune rules from keep, defer, and reject decisions.
- Once per week, inspect a random sample of five hard-rejected papers plus the five candidates immediately below the selection threshold.
- Record missed relevant papers and the rule that hid each one.
- Add an LLM reranker only if static ranking repeatedly puts relevant papers below irrelevant ones.

### Pilot measures

- **Precision at three:** fraction of surfaced papers marked keep or defer.
- **Deep-read conversion:** surfaced papers that become a full reading note.
- **Miss rate audit:** relevant papers found in rejected or below-threshold samples.
- **Duplicate rate:** resurfaced papers that should have been suppressed.
- **Source yield:** selected papers attributed to each source.
- **Weekly triage time:** time spent deciding, excluding deep reading.
- **Empty-digest rate:** tracked but not treated as failure by itself.

At the end of week four, choose one outcome:

1. Keep Proposal B and tune configuration.
2. Reduce to Proposal A if automation saves too little time.
3. Adopt Proposal C if learned ranking and a browser UI justify the extra operation.

## Suggested Version 1 Layout

```text
tools/paper_digest/
  pyproject.toml
  config.example.yaml
  paper_digest/
    cli.py
    models.py
    normalize.py
    dedupe.py
    score.py
    render.py
    sources/
      arxiv.py
      huggingface.py
      semantic_scholar.py
  templates/
    digest.md.j2
    digest.html.j2
  systemd/
    paper-digest.service
    paper-digest.timer
  tests/
    test_identity.py
    test_dedupe.py
    test_scoring.py
```

This is a planning layout, not a commitment to create a new top-level service before the pilot is approved.

## Non-Goals for Version 1

- PDF download, parsing, OCR, or full-text embeddings
- a vector database
- a hosted multi-user service
- scraping alphaXiv, conference websites, or ResearchRabbit
- browser automation against Zotero or ResearchRabbit
- automatic project conclusions from paper abstracts
- hard-coded yearly conference calendars
- daily LLM processing of the full candidate pool

## Sources

- [arXiv API User's Manual](https://info.arxiv.org/help/api/user-manual.html)
- [Hugging Face `HfApi` paper methods](https://huggingface.co/docs/huggingface_hub/en/package_reference/hf_api)
- [Semantic Scholar Recommendations API](https://api.semanticscholar.org/api-docs/recommendations)
- [Zotero Quick Start Guide](https://www.zotero.org/support/quick_start_guide)
- [Zotero advanced feed preferences](https://www.zotero.org/support/preferences/advanced)
- [ResearchRabbit Zotero importer](https://learn.researchrabbit.ai/en/articles/12796541-using-the-zotero-importer)
- [ResearchRabbit collection export](https://learn.researchrabbit.ai/en/articles/13192798-how-to-share-papers-and-collections)
- [arxiv-sanity-lite](https://github.com/karpathy/arxiv-sanity-lite)
