---
pretty_name: PreReviewBench
language:
  - en
task_categories:
  - text-classification
tags:
  - benchmark
  - scientific-publishing
  - editorial-desk-screening
  - journal-recommendation
  - robustness
configs:
  - config_name: main
    default: true
    data_files:
      - split: test
        path: benchmark2025.json
  - config_name: keyword_noise
    data_files:
      - split: test
        path: benchmark2025_keyword_noise_mixed.json
  - config_name: synthetic_degradation
    data_files:
      - split: test
        path: benchmark2025_synthetic_degradation.json
---

# Dataset Card for PreReviewBench

**PreReviewBench: Benchmarking LLMs for Editorial Desk Screening in Scholarly Publishing**

[Code and experiment instructions](https://github.com/kg4sci/PRB_db) · [Dataset](https://huggingface.co/datasets/kg4sci/PreReviewBench)

PreReviewBench evaluates whether language models can support journal editors at the desk-screening stage, before external peer review. It supports three connected tasks: **desk triage**, **branch-specific editorial assessment**, and **transfer-journal recommendation**.

The main benchmark contains **7,292 paper–journal instances**, representing **5,498 distinct papers**, across **58 journals, eight subfields, and four broad areas**. It uses publication-time metadata for papers published in 2025 and indexed in Web of Science. Its labels are **constructed editorial proxies**, not confidential records of actual desk decisions.

## Dataset Details

### Dataset Description

A model receives a manuscript's title, abstract, and keywords together with a target-journal profile. It predicts whether to send the manuscript for external review or desk reject it, supplies the appropriate branch-specific judgment, and ranks transfer venues for rejected cases.

- **Curated by:** The PreReviewBench research team: Ming Chen, Zhiyuan Ning, and Yi Du.
- **Shared by:** `kg4sci`.
- **Language:** English.
- **Source publication year:** 2025.
- **Broad areas:** Biology, Chemistry, Materials Science, and Medicine.
- **Label origin:** Controlled paper–journal pairing, with a separate expert validation study.
- **License:** No dataset license has been specified for this release. Contact the maintainers for clarification; this card does not assign a license to the dataset or its source materials.

### Dataset Sources

- **Code repository:** https://github.com/kg4sci/PRB_db
- **Dataset repository:** https://huggingface.co/datasets/kg4sci/PreReviewBench
- **Accompanying manuscript:** *PreReviewBench: Benchmarking LLMs for Editorial Desk Screening in Scholarly Publishing*, by Ming Chen, Zhiyuan Ning, and Yi Du.
- **Source resources:** Web of Science article metadata and journal indicators, Google Scholar Top Publications/h5-index information, and official journal aims & scope descriptions.

## Uses

### Direct Use

The dataset is intended for evaluating and auditing editorial decision-support systems:

| Task | Prediction | Evaluation population |
| --- | --- | --- |
| Desk triage | `Send for Review` or `Desk Reject`, with a confidence score from 1 to 10 | All main-benchmark instances |
| Branch-specific assessment | Send: `Matches Journal Caliber` / `Exceeds Journal Caliber`; reject: `Insufficient Novelty/Impact` / `Out of Scope` | Corresponding gold send or reject branch |
| Transfer recommendation | Ranking of candidate journals | Gold reject instances |

Applications include studying scope-versus-caliber judgments, journal-context sensitivity, robustness to input perturbations, transfer-ranking quality, and model decision tendencies. The paper's task protocol places triage before access to transfer candidates. The released experiment code uses single-stage requests with an instruction to consult candidates after triage; it does not enforce separate-call information isolation.

### Out-of-Scope Use

The benchmark is not intended to automate editorial acceptance or rejection, replace expert peer review, or establish the intrinsic quality of a paper or journal. It does not evaluate full manuscripts, verify scientific validity, or observe all factors involved in real editorial decisions. Benchmark labels should not be presented as historical decisions made by the named journals.

## Dataset Structure

### Configurations and Files

| Configuration | File | Split | Instances |
| --- | --- | --- | ---: |
| `main` | `benchmark2025.json` | `test` | 7,292 |
| `keyword_noise` | `benchmark2025_keyword_noise_mixed.json` | `test` | 1,823 |
| `synthetic_degradation` | `benchmark2025_synthetic_degradation.json` | `test` | 1,823 |

All files are JSON arrays. The `test` designation identifies evaluation data; no training or validation split is provided. The robustness sets derive from the Published-at-Target cases and do not represent additional independent papers. The three configurations should not be pooled as independent observations.

### Scenarios

The main benchmark is exactly balanced across four scenarios, with **1,823 instances per scenario** and **3,646 instances per triage class**.

| Scenario in the paper | Serialized `data_type` | Pairing | Triage label | Branch label |
| --- | --- | --- | --- | --- |
| Published-at-Target | `Real_Submission` | Paper and its realized publication venue | Send for Review | Matches Journal Caliber |
| Caliber-Surplus | `High_to_Low` | High-SSI source to lower-SSI target within a subfield | Send for Review | Exceeds Journal Caliber |
| Caliber-Deficit | `Low_to_High` | Low-SSI source to higher-SSI target within a subfield | Desk Reject | Insufficient Novelty/Impact |
| Scope-Mismatch | `Out_of_Scope` | Paper to the other subfield in the same broad area | Desk Reject | Out of Scope |

`Real_Submission` is a scenario identifier, not an indication that confidential submission records were obtained. The JSON serializes the scope label as `Out of scope`; the paper uses `Out of Scope`, and the evaluator normalizes the capitalization.

### Subfield Coverage

| Broad area | Source subfield | Per scenario | Total instances | Within-subfield candidate pool | Paired-subfield union |
| --- | --- | ---: | ---: | ---: | ---: |
| Biology | Cell Biology | 468 | 1,872 | 10 | 15 |
| Biology | Mycology | 69 | 276 | 5 | 15 |
| Chemistry | Inorganic Chemistry | 234 | 936 | 7 | 14 |
| Chemistry | Medicinal Chemistry | 237 | 948 | 7 | 14 |
| Materials Science | Ceramics | 189 | 756 | 9 | 16 |
| Materials Science | Composites | 297 | 1,188 | 7 | 16 |
| Medicine | Dermatology | 121 | 484 | 7 | 13 |
| Medicine | Ophthalmology | 208 | 832 | 6 | 13 |
| **Total / range** | **8 subfields** | **1,823** | **7,292** | **5–10** | **13–16** |

Caliber-Deficit transfer candidates come from the source subfield. Scope-Mismatch candidates come from the union of the two paired subfields. The realized publication venue is the primary gold destination. The paper reports an average of 203.8 abstract words and 8.8 keywords per paper.

### Data Fields

| Field | Description |
| --- | --- |
| `uid`, `doi` | Paper identifiers; a paper may occur in multiple scenarios |
| `Category` | Broad source area |
| `subject` | Source Web of Science subfield |
| `data_type` | Constructed scenario identifier |
| `paper_content.title` | Published paper title |
| `paper_content.abstract_text` | Indexed abstract; see the degradation-specific rule below |
| `paper_content.keywords` | Paper keywords, including injected terms in the noise configuration |
| `simulation_setting` | Target-journal metadata, including `target_journal_name`, `aim_scope`, `JIF`, `JIF_Quartile`, and `h5-index` |
| `ground_truth.actual_published_journal` | Realized publication venue, for evaluation rather than model input |
| `ground_truth.expected_decision` | Constructed desk-triage label |
| `ground_truth.expected_reason_label` | Constructed branch-specific label |
| `transfer_recommendation_task.candidate_count` | Number of candidate journals |
| `transfer_recommendation_task.candidates` | Option letters and candidate profiles: `option`, `journal_name`, `JIF`, `JIF_Quartile`, `h5-index`, and `aim_scope` |
| `transfer_recommendation_task.correct_option` | Option letter for the realized publication venue |

The two robustness files omit `transfer_recommendation_task` and add perturbation-specific fields:

- **Keyword noise:** `added_keywords` records injected terms. `paper_content.keywords` already contains the modified input. The gold send label is unchanged.
- **Synthetic degradation:** `degraded_abstract_text` is the abstract to present to the model. `paper_content.abstract_text` retains the original abstract. The gold label changes to `Desk Reject` / `Insufficient Novelty/Impact`.

**Instance identity:** neither `uid` nor DOI is unique across the main benchmark. Use the paper identifier, `data_type`, `subject`, and target-journal name together. Any training/evaluation partition created for downstream research should group by paper to avoid placing different instances of the same paper in different splits.

**Input separation:** do not pass `ground_truth`, the source/realized publication venue, or constructed scenario labels to the model. Candidate lists can be stored for send cases, but transfer ranking is evaluated only on gold reject cases.

### Loading the Dataset

After this card is saved as the dataset repository's root `README.md`, its YAML configuration enables:

```python
from datasets import load_dataset

main = load_dataset("kg4sci/PreReviewBench", "main", split="test")
noise = load_dataset("kg4sci/PreReviewBench", "keyword_noise", split="test")
degraded = load_dataset("kg4sci/PreReviewBench", "synthetic_degradation", split="test")

# The frozen degraded abstract must replace the original model input.
example = degraded[0]
paper_for_model = dict(example["paper_content"])
paper_for_model["abstract_text"] = example["degraded_abstract_text"]
```

To obtain the original JSON without changing its nested structure:

```python
import json
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="kg4sci/PreReviewBench",
    filename="benchmark2025.json",
    repo_type="dataset",
)
with open(path, encoding="utf-8") as handle:
    records = json.load(handle)
```

## Dataset Creation

### Curation Rationale

Real desk-screening decisions are generally confidential, making direct evaluation difficult. PreReviewBench constructs controlled comparisons to test two distinct judgments: whether a paper fits a journal's topical scope, and whether an in-scope paper meets its caliber expectations. Three construction tracks produce four scenarios, allowing both send and reject decisions to be assessed under balanced conditions.

### Source Data

#### Data Collection and Processing

1. Select eight WoS subfields, two per broad area, with corresponding Google Scholar Top Publications listings. Retain journals represented in both systems.
2. Collect title, indexed abstract, and keyword metadata for papers published in 2025. Assemble journal profiles from JIF, JCR quartile, h5-index, and official aims & scope.
3. Compute the Scholar Synergy Index (SSI) from within-subfield JIF and h5-index percentile ranks; retain the top and bottom 25% journal strata.
4. Use `qwen3-embedding-8b` paper embeddings to form normalized journal centroids. Filter topically isolated journals using median cosine distances and a MAD-based criterion; the paper uses a cutoff of 3.0 in subfields with at least four journals.
5. Preserve published-at-target pairs. For within-subfield high/low journal pairs, narrow citation-selection bands until every retained high-side paper has more citations than every retained low-side paper, then construct surplus and deficit directions.
6. Construct scope mismatches with the other subfield in the same broad area, choosing a target journal with the closest SSI. Balance the four scenarios within each source subfield.

SSI is defined as

$$
\mathrm{SSI}=100\left(\frac{\sqrt{R_{\mathrm{JIF}}R_{h5}}}{100}\right)^{1.5},
$$

where the ranks are within-subfield percentiles. SSI is a venue-level proxy. Citation separation is pair-specific; pooled journal-level citation distributions may overlap across different pairs.

The keyword-noise stress test injects five hot-topic and cross-domain terms while preserving send labels. The synthetic-degradation test uses generated weaker abstracts that preserve broad topic and target fit, with reject labels reflecting the intended perturbation. These stress-test labels are constructed, not observed editorial decisions.

#### Source Data Producers

Manuscript text originates from authors of published scholarly articles. Journal profiles originate from the journals and the bibliographic/metric resources described above. The PreReviewBench team curated the records and constructed the paper–journal instances and stress tests. The benchmark does not provide a demographic characterization of source authors.

### Annotations

#### Annotation Process

The full benchmark labels are assigned by the construction procedure, rather than independently assigned by human reviewers for every instance.

A separate blind, stratified expert-validation study reviewed **200 cases**: 40 Published-at-Target, 60 Caliber-Surplus, 60 Caliber-Deficit, and 40 Scope-Mismatch. Four broad-area teams each reviewed 50 cases, covering all eight subfields. Annotators saw only title, abstract, keywords, and target-journal profiles, without source venues, constructed labels, or model outputs.

| Agreement with constructed labels | Result |
| --- | ---: |
| Desk triage | 98.00% (196/200) |
| Branch label, conditional on triage agreement | 98.47% (193/196) |
| Joint triage and branch label | 96.50% (193/200) |

All four triage disagreements were Caliber-Surplus cases judged out of scope. Each case was reviewed by one annotator, so these figures measure agreement with constructed labels, not inter-annotator agreement. The validation study does not imply that all benchmark instances or synthetic perturbations received expert review. Its annotation files are not included among the three JSON files in this Hugging Face release.

#### Annotators

The study was led by four domain experts outside the author team, one per broad area. Each expert annotated cases and coordinated graduate students from the same field who also participated in labeling.

### Personal and Sensitive Information

Inputs are publication-time records rather than confidential submitted manuscripts or editorial correspondence. DOI, title, and publication-venue information make the source papers identifiable; the release is not an anonymized corpus. No demographic annotations of authors are provided. Public scholarly abstracts may discuss human participants, health conditions, or other sensitive research topics; publication-time provenance should not be interpreted as a guarantee that every text is free of sensitive content.

## Bias, Risks, and Limitations

- **Proxy labels:** publication venue, citation contrast, and SSI do not establish intrinsic manuscript quality or recover an actual editor's threshold.
- **Restricted coverage:** only 2025 papers from four broad areas, eight subfields, and 58 journals are represented. Results need not generalize to other fields, years, languages, or editorial cultures.
- **Published rather than submitted text:** final indexed abstracts can differ from original submission versions. Full-text methods, reviewer availability, journal portfolios, and other decision factors are unobserved.
- **Balanced design:** the 50/50 send–reject balance is an evaluation construction, not an estimate of real submission prevalence.
- **Correlated instances:** the same paper can occur in multiple scenarios, and robustness examples derive from main-benchmark papers.
- **Limited human validation:** the 200-case study uses one annotator per case and does not make all labels objective or noise-free.
- **Transfer ambiguity:** the realized venue is a single primary target, although multiple destinations may be reasonable. The paper's metadata multi-gold analysis uses weak compatibility labels rather than editor-confirmed alternatives.
- **Potential memorization:** publication-time records may overlap with model training data; closed-book probes cannot rule out contamination.
- **Editorial bias:** bibliometric stratification and automated recommendations may reinforce disciplinary, linguistic, or prestige-related biases.

### Recommendations

Use the dataset for controlled evaluation and auditing, with human accountability for real editorial decisions. Report scenario-specific results alongside overall scores. Keep model inputs separate from ground truth and use paper-grouped splits if constructing downstream training experiments. Distinguish exact-venue, candidate-projected, strict-permutation, and metadata multi-gold ranking results. Do not interpret confidence scores as calibrated probabilities or benchmark predictions as authoritative judgments of the named papers or journals.

## Citation

The following entry identifies the accompanying manuscript without asserting an unverified venue, DOI, or publication date:

```bibtex
@misc{chen_prereviewbench,
  title = {{PreReviewBench}: Benchmarking LLMs for Editorial Desk Screening in Scholarly Publishing},
  author = {Chen, Ming and Ning, Zhiyuan and Du, Yi},
  howpublished = {Accompanying manuscript},
  url = {https://github.com/kg4sci/PRB_db}
}
```

**APA-style reference:** Chen, M., Ning, Z., & Du, Y. (n.d.). *PreReviewBench: Benchmarking LLMs for editorial desk screening in scholarly publishing* [Manuscript]. https://github.com/kg4sci/PRB_db

## Glossary

- **Desk triage:** the initial editorial decision about whether to allocate external reviewer attention.
- **Journal caliber:** the journal-relative selectivity/impact dimension operationalized by benchmark proxies.
- **SSI:** Scholar Synergy Index, combining within-subfield JIF and h5-index percentile ranks.
- **Gold branch:** the send/reject branch specified by the constructed label, used to select the population for branch-specific evaluation.
- **Joint correctness:** both the triage decision and its branch-specific label are correct.

## Dataset Card Contact

Questions and corrections: [GitHub issues](https://github.com/kg4sci/PRB_db/issues).

Research contacts: Ming Chen (`chenming@cnic.cn`), Zhiyuan Ning (`ningzhiyuan@westlake.edu.cn`), and Yi Du (`duyi@cnic.cn`).
