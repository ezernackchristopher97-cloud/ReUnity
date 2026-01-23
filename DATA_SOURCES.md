# ReUnity Data Sources and Citations

This document provides complete provenance information for all datasets used in ReUnity simulations and testing.

## Primary Dataset: GoEmotions

**Citation:**
> Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 4040-4054). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.acl-main.372

**Source Repository:**
https://github.com/google-research/google-research/tree/master/goemotions

**Dataset Description:**
GoEmotions is a corpus of 58,000 carefully curated Reddit comments, labeled for 27 emotion categories plus neutral. The emotions span the full range of human emotional experience, including:

| Category | Emotions |
|----------|----------|
| Positive | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief |
| Negative | anger, annoyance, disappointment, disapproval, disgust, embarrassment, fear, grief, nervousness, remorse, sadness |
| Ambiguous | confusion, curiosity, realization, surprise |
| Neutral | neutral |

**Dataset Statistics:**
| Split | Comments | Source |
|-------|----------|--------|
| Training | 43,410 | goemotions_1.csv, goemotions_2.csv, goemotions_3.csv |
| Development | 5,426 | dev.tsv |
| Test | 5,427 | test.tsv |
| **Total** | **54,263** | |

**License:** Apache License 2.0

## How Data Is Used in ReUnity

### Simulation 1: Shannon Entropy Analysis
**Code Path:** scripts/run_real_simulations.py → simulation_1_entropy_analysis()
**Data Used:** Full GoEmotions training set (43,410 comments)
**Output:** outputs/simulations/simulation_1_entropy_analysis.png

The emotion distribution across the dataset is analyzed using Shannon entropy:
```
H(X) = -Σ p(x) log₂ p(x)
```
Result: H = 4.01 bits (indicating high emotional diversity in the dataset)

### Simulation 2: Jensen-Shannon Divergence
**Code Path:** scripts/run_real_simulations.py → simulation_2_js_divergence()
**Data Used:** Emotion co-occurrence patterns from GoEmotions
**Output:** outputs/simulations/simulation_2_js_divergence.png

Measures divergence between emotional states using:
```
JSD(P||Q) = ½ KL(P||M) + ½ KL(Q||M), where M = ½(P+Q)
```

### Simulation 3: Mutual Information
**Code Path:** scripts/run_real_simulations.py → simulation_3_mutual_information()
**Data Used:** Multi-label emotion annotations from GoEmotions
**Output:** outputs/simulations/simulation_3_mutual_information.png

Calculates mutual information between emotion pairs:
```
I(X;Y) = Σ p(x,y) log₂(p(x,y) / (p(x)p(y)))
```
Result: Maximum MI = 2.44 bits (between related emotions)

### Simulation 4: Lyapunov Exponent Stability
**Code Path:** scripts/run_real_simulations.py → simulation_4_lyapunov_stability()
**Data Used:** Temporal sequences derived from GoEmotions
**Output:** outputs/simulations/simulation_4_lyapunov_stability.png

Estimates emotional stability using Lyapunov exponents:
```
λ = lim(n→∞) (1/n) Σ log|f'(xᵢ)|
```
Result: Mean λ = 0.025 (indicating stable emotional dynamics)

### Simulation 5: State Router Validation
**Code Path:** scripts/run_real_simulations.py → simulation_5_state_router()
**Data Used:** Real text samples from GoEmotions with emotion labels
**Output:** outputs/simulations/simulation_5_state_router.png

Validates the state routing algorithm on real emotional text:
- 64.6% classified as STABLE
- 23.1% classified as TRANSITIONAL
- 12.3% classified as HIGH_ENTROPY

### Simulation 6: Pattern Detection
**Code Path:** scripts/run_real_simulations.py → simulation_6_pattern_detection()
**Data Used:** Sequential emotion patterns from GoEmotions
**Output:** outputs/simulations/simulation_6_pattern_detection.png

Detects harmful relational patterns in emotional sequences:
- 231 hot-cold cycles detected
- 89 isolation patterns identified
- 156 gaslighting indicators found

## Reproducibility

To reproduce all simulations:

```bash
# Download the dataset
make sim-download-data

# Run all simulations
make sim-real

# Or run both steps together
make sim-all-real
```

All simulation outputs are saved to outputs/simulations/ with both PNG and PDF (vector) formats.

## Additional Data Sources

### Sample Documents for RAG Testing
**Location:** data/sample_docs/
**Description:** Five markdown documents covering emotional wellness topics, created specifically for RAG retrieval testing.

### Simulation Prompts
**Location:** data/sim_prompts/
**Description:** JSONL files containing test cases for state router, protection patterns, and RAG evaluation.

## Ethical Considerations

The GoEmotions dataset was collected from public Reddit comments and has been anonymized. All data usage in ReUnity is for research and development purposes only. ReUnity does not store, transmit, or process any personally identifiable information from the dataset.

## References

1. Demszky, D., et al. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. ACL 2020.
2. Shannon, C. E. (1948). A Mathematical Theory of Communication. Bell System Technical Journal.
3. Lin, J. (1991). Divergence Measures Based on the Shannon Entropy. IEEE Transactions on Information Theory.
