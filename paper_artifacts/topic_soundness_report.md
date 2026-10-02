# Six-corpus topic coherence / label soundness (LLM judge, 1--5)

Both baselines were run on the full shared cohort of every corpus (same documents as the EviMap demo artifact, same LLM backend).
EviMap rows use 96 deterministic stratified leaf topics per corpus; baseline rows use all topic units with evidence.
One DeepSeek-V4-Flash judgment per unit. Diagnostic of semantic soundness, not topic accuracy.

## EviMap leaves excluding Other bins

| Corpus | EviMap coh/sound (N) | TnT-LLM-style coh/sound (N) | TopicGPT-style coh/sound (N) |
|---|---:|---:|---:|
| ICLR peer reviews | 4.49/4.34 (96) | 3.58/3.46 (24) | 3.50/3.46 (24) |
| TripAdvisor hotel reviews | 4.50/4.32 (96) | 4.32/4.11 (28) | 4.67/4.62 (24) |
| Pitchfork music reviews | 4.50/4.26 (96) | 4.13/4.00 (31) | 4.46/4.29 (24) |
| LinkedIn job postings | 4.43/4.35 (96) | 3.59/3.53 (32) | 3.04/3.04 (24) |
| Recipes | 4.72/4.44 (96) | 4.12/3.69 (32) | 2.62/2.33 (24) |
| EU debates | 4.52/4.28 (96) | 4.00/3.94 (32) | 3.29/3.25 (24) |
| **Macro average** | **4.53/4.33** | **3.96/3.79** | **3.60/3.50** |

EviMap strictly leads both baselines on coherence in 5/6 corpora and on label soundness in 4/6.

## EviMap leaves including Other bins

| Corpus | EviMap coh/sound (N) | TnT-LLM-style coh/sound (N) | TopicGPT-style coh/sound (N) |
|---|---:|---:|---:|
| ICLR peer reviews | 4.51/4.38 (96) | 3.54/3.46 (24) | 3.50/3.38 (24) |
| TripAdvisor hotel reviews | 4.52/4.31 (96) | 4.36/4.04 (28) | 4.62/4.62 (24) |
| Pitchfork music reviews | 4.43/4.24 (96) | 4.19/4.07 (31) | 4.42/4.33 (24) |
| LinkedIn job postings | 3.16/2.52 (96) | 3.56/3.53 (32) | 2.96/2.96 (24) |
| Recipes | 4.71/4.45 (96) | 4.12/3.66 (32) | 2.62/2.25 (24) |
| EU debates | 4.50/4.31 (96) | 4.03/3.91 (32) | 3.38/3.42 (24) |
| **Macro average** | **4.30/4.04** | **3.97/3.77** | **3.58/3.49** |

EviMap strictly leads both baselines on coherence in 4/6 corpora and on label soundness in 3/6.

Other-bin share of the 96 sampled EviMap leaves: ICLR peer reviews 1, TripAdvisor hotel reviews 2, Pitchfork music reviews 1, LinkedIn job postings 72, Recipes 1, EU debates 1.

## Sources

- `experiments/evaluation/topic_soundness_peer_review_full_shared_nonother`
- `experiments/evaluation/topic_soundness_peer_review_full_shared`
- `experiments/evaluation/topic_soundness_city_hotel_full_shared_nonother`
- `experiments/evaluation/topic_soundness_city_hotel_full_shared`
- `experiments/evaluation/topic_soundness_music_reviews_full_shared_nonother`
- `experiments/evaluation/topic_soundness_music_reviews_full_shared`
- `experiments/evaluation/topic_soundness_career_jobs_full_shared_nonother`
- `experiments/evaluation/topic_soundness_career_jobs_full_shared`
- `experiments/evaluation/topic_soundness_recipes_full_shared_nonother`
- `experiments/evaluation/topic_soundness_recipes_full_shared`
- `experiments/evaluation/topic_soundness_eu_debates_full_shared_nonother`
- `experiments/evaluation/topic_soundness_eu_debates_full_shared`
