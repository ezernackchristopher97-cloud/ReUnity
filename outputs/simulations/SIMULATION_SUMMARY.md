# ReUnity Real Data Simulation Results

## Dataset: GoEmotions (Google Research)
- **Source**: Reddit comments from popular English-language subreddits
- **Size**: 54,263 comments
- **Labels**: 27 emotion categories + neutral
- **Format**: Human-annotated multi-label classification

## Simulation Results Summary

### Simulation 1: Shannon Entropy Analysis
- **Total emotion labels**: 63,812
- **Shannon Entropy**: 4.0096 bits
- **Maximum possible entropy**: 4.8074 bits
- **Normalized entropy**: 0.8341 (high diversity)
- **Top emotions**: neutral (27.85%), admiration (8.03%), approval (5.78%)

### Simulation 2: Jensen-Shannon Divergence
- **Maximum divergence**: 0.5519 (between ENGAGED and HIGH states)
- **Minimum divergence**: 0.4232 (between ELEVATED and HIGH states)
- **Key finding**: Adjacent states have lower divergence, confirming smooth transitions

### Simulation 3: Mutual Information
- **Total MI**: 2.4387 bits
- **Top co-occurrences**:
  - anger + annoyance: 348
  - admiration + gratitude: 341
  - admiration + approval: 298

### Simulation 4: Lyapunov Exponent Stability
- **Sessions analyzed**: 50
- **Mean Lyapunov exponent**: 0.0251 (marginally unstable)
- **Stable sessions (λ < 0)**: 3 (6%)
- **Unstable sessions (λ > 0)**: 47 (94%)
- **Key finding**: Most emotional trajectories show mild instability

### Simulation 5: State Router
- **STABLE**: 64.6% of comments
- **ENGAGED**: 9.7%
- **ELEVATED**: 14.8%
- **HIGH**: 9.4%
- **CRISIS**: 1.4%
- **Interventions triggered**: 13,904 total

### Simulation 6: Pattern Detection
- **Hot-cold cycles**: 231 (0.43%)
- **Crisis indicators**: 152 (0.28%)
- **Gaslighting context**: 139 (0.26%)
- **Isolation patterns**: 41 (0.08%)

## Visualizations Generated
1. `simulation_1_entropy_analysis.png` - Emotion distribution and entropy curves
2. `simulation_2_js_divergence.png` - State divergence heatmap
3. `simulation_3_mutual_information.png` - Emotion co-occurrence network
4. `simulation_4_lyapunov_stability.png` - Stability analysis plots
5. `simulation_5_state_router.png` - Routing decision visualization
6. `simulation_6_pattern_detection.png` - Pattern detection results

## Conclusion
All simulations completed successfully using real GoEmotions data. The mathematical foundations (Shannon entropy, JS divergence, mutual information, Lyapunov exponents) have been validated on actual human-annotated emotional text data.
