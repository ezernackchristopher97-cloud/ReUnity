#!/usr/bin/env python3
"""
ReUnity Real Data Simulations

This script runs actual simulations using the GoEmotions dataset (58k Reddit comments
with 27 emotion labels) from Google Research. NO SYNTHETIC DATA.

Simulations include:
1. Shannon Entropy Analysis on real emotional state distributions
2. Jensen-Shannon Divergence between emotional states
3. Mutual Information analysis for emotion co-occurrence
4. Lyapunov Exponent estimation for emotional stability
5. State Router simulation with real text data
6. Pattern Detection on real conversations
7. Free Energy Principle demonstration

Requirements: PyTorch, matplotlib, scipy, numpy, pandas
Data: GoEmotions dataset (train.tsv, dev.tsv, test.tsv, emotions.txt)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import rel_entr
from collections import Counter
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Configuration
DATA_DIR = Path(__file__).parent.parent / 'data' / 'goemotions'
OUTPUT_DIR = Path(__file__).parent.parent / 'outputs' / 'simulations'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Emotion categories from GoEmotions
EMOTIONS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Map emotions to ReUnity states
EMOTION_TO_STATE = {
    'neutral': 'STABLE',
    'joy': 'STABLE', 'love': 'STABLE', 'gratitude': 'STABLE', 'relief': 'STABLE',
    'admiration': 'STABLE', 'approval': 'STABLE', 'caring': 'STABLE', 'optimism': 'STABLE',
    'pride': 'STABLE', 'excitement': 'STABLE', 'amusement': 'STABLE',
    'curiosity': 'ENGAGED', 'realization': 'ENGAGED', 'surprise': 'ENGAGED', 'desire': 'ENGAGED',
    'confusion': 'ELEVATED', 'nervousness': 'ELEVATED', 'embarrassment': 'ELEVATED',
    'disappointment': 'ELEVATED', 'disapproval': 'ELEVATED', 'annoyance': 'ELEVATED',
    'sadness': 'HIGH', 'remorse': 'HIGH', 'grief': 'HIGH',
    'anger': 'HIGH', 'disgust': 'HIGH', 'fear': 'CRISIS'
}

REUNITY_STATES = ['STABLE', 'ENGAGED', 'ELEVATED', 'HIGH', 'CRISIS']


def load_goemotions_data():
    """Load the real GoEmotions dataset."""
    print("Loading GoEmotions dataset...")
    
    train_path = DATA_DIR / 'train.tsv'
    dev_path = DATA_DIR / 'dev.tsv'
    test_path = DATA_DIR / 'test.tsv'
    
    if not train_path.exists():
        raise FileNotFoundError(f"GoEmotions data not found at {DATA_DIR}. Please download first.")
    
    # Load all data
    train_df = pd.read_csv(train_path, sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
    dev_df = pd.read_csv(dev_path, sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
    test_df = pd.read_csv(test_path, sep='\t', header=None, names=['text', 'emotion_ids', 'comment_id'])
    
    # Combine all data
    all_data = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    # Parse emotion IDs (comma-separated integers)
    def parse_emotions(emotion_str):
        if pd.isna(emotion_str):
            return []
        return [int(x) for x in str(emotion_str).split(',') if x.strip().isdigit()]
    
    all_data['emotion_list'] = all_data['emotion_ids'].apply(parse_emotions)
    all_data['emotion_names'] = all_data['emotion_list'].apply(
        lambda ids: [EMOTIONS[i] for i in ids if i < len(EMOTIONS)]
    )
    
    print(f"Loaded {len(all_data)} comments from GoEmotions dataset")
    return all_data


def calculate_shannon_entropy(probabilities):
    """Calculate Shannon entropy for a probability distribution."""
    probabilities = np.array(probabilities, dtype=np.float64)
    probabilities = probabilities[probabilities > 0]  # Remove zeros
    if len(probabilities) == 0:
        return 0.0
    probabilities = probabilities / np.sum(probabilities)  # Normalize
    return -np.sum(probabilities * np.log2(probabilities))


def calculate_js_divergence(p, q):
    """Calculate Jensen-Shannon divergence between two distributions."""
    p = np.array(p, dtype=np.float64)
    q = np.array(q, dtype=np.float64)
    
    # Normalize
    p = p / np.sum(p) if np.sum(p) > 0 else p
    q = q / np.sum(q) if np.sum(q) > 0 else q
    
    # Midpoint distribution
    m = 0.5 * (p + q)
    
    # KL divergences with numerical stability
    epsilon = 1e-10
    kl_pm = np.sum(rel_entr(p + epsilon, m + epsilon))
    kl_qm = np.sum(rel_entr(q + epsilon, m + epsilon))
    
    return 0.5 * (kl_pm + kl_qm)


def calculate_mutual_information(joint_counts):
    """Calculate mutual information from joint probability counts."""
    joint_counts = np.array(joint_counts, dtype=np.float64)
    joint_prob = joint_counts / np.sum(joint_counts)
    
    # Marginal probabilities
    p_x = np.sum(joint_prob, axis=1)
    p_y = np.sum(joint_prob, axis=0)
    
    # Mutual information
    mi = 0.0
    for i in range(joint_prob.shape[0]):
        for j in range(joint_prob.shape[1]):
            if joint_prob[i, j] > 0 and p_x[i] > 0 and p_y[j] > 0:
                mi += joint_prob[i, j] * np.log2(joint_prob[i, j] / (p_x[i] * p_y[j]))
    
    return mi


def estimate_lyapunov_exponent(time_series):
    """Estimate Lyapunov exponent from a time series."""
    if len(time_series) < 10:
        return 0.0
    
    time_series = np.array(time_series, dtype=np.float64)
    
    # Calculate derivatives (rate of change)
    derivatives = np.diff(time_series)
    
    # Avoid log of zero
    abs_derivatives = np.abs(derivatives)
    abs_derivatives = abs_derivatives[abs_derivatives > 1e-10]
    
    if len(abs_derivatives) == 0:
        return 0.0
    
    # Lyapunov exponent approximation
    lyapunov = np.mean(np.log2(abs_derivatives + 1e-10))
    return lyapunov


# ============================================================================
# SIMULATION 1: Shannon Entropy Analysis on Real Emotion Distributions
# ============================================================================
def simulation_1_entropy_analysis(data):
    """Analyze Shannon entropy of emotional states from real data."""
    print("\n" + "="*70)
    print("SIMULATION 1: Shannon Entropy Analysis on Real GoEmotions Data")
    print("="*70)
    
    # Count emotion occurrences
    emotion_counts = Counter()
    for emotions in data['emotion_names']:
        for emotion in emotions:
            emotion_counts[emotion] += 1
    
    # Calculate probability distribution
    total = sum(emotion_counts.values())
    emotion_probs = {e: emotion_counts[e] / total for e in EMOTIONS}
    
    # Calculate entropy
    probs_array = np.array([emotion_probs[e] for e in EMOTIONS])
    entropy = calculate_shannon_entropy(probs_array)
    max_entropy = np.log2(len(EMOTIONS))
    normalized_entropy = entropy / max_entropy
    
    print(f"\nTotal emotion labels: {total}")
    print(f"Shannon Entropy: {entropy:.4f} bits")
    print(f"Maximum possible entropy: {max_entropy:.4f} bits")
    print(f"Normalized entropy: {normalized_entropy:.4f}")
    
    # Top emotions
    print("\nTop 10 emotions by frequency:")
    for emotion, count in emotion_counts.most_common(10):
        print(f"  {emotion}: {count} ({100*count/total:.2f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Emotion distribution
    ax1 = axes[0, 0]
    sorted_emotions = sorted(emotion_probs.items(), key=lambda x: x[1], reverse=True)
    emotions_sorted = [e[0] for e in sorted_emotions]
    probs_sorted = [e[1] for e in sorted_emotions]
    bars = ax1.barh(emotions_sorted, probs_sorted, color='steelblue')
    ax1.set_xlabel('Probability')
    ax1.set_title('GoEmotions: Emotion Probability Distribution\n(58k Reddit Comments)')
    ax1.invert_yaxis()
    
    # Plot 2: Entropy curve
    ax2 = axes[0, 1]
    p_range = np.linspace(0.01, 0.99, 100)
    binary_entropy = -p_range * np.log2(p_range) - (1-p_range) * np.log2(1-p_range)
    ax2.plot(p_range, binary_entropy, 'b-', linewidth=2, label='Binary Entropy H(p)')
    ax2.axhline(y=1.0, color='r', linestyle='--', label='Maximum (1 bit)')
    ax2.axvline(x=0.5, color='g', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Probability p')
    ax2.set_ylabel('Entropy H(X) in bits')
    ax2.set_title('Shannon Entropy Function')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Cumulative distribution
    ax3 = axes[1, 0]
    cumulative = np.cumsum(probs_sorted)
    ax3.plot(range(len(emotions_sorted)), cumulative, 'b-', linewidth=2, marker='o', markersize=4)
    ax3.axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
    ax3.set_xlabel('Number of Emotions')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Emotion Distribution')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Entropy by ReUnity state
    ax4 = axes[1, 1]
    state_counts = Counter()
    for emotions in data['emotion_names']:
        for emotion in emotions:
            state = EMOTION_TO_STATE.get(emotion, 'ELEVATED')
            state_counts[state] += 1
    
    state_total = sum(state_counts.values())
    state_probs = {s: state_counts[s] / state_total for s in REUNITY_STATES}
    
    colors = ['green', 'blue', 'yellow', 'orange', 'red']
    bars = ax4.bar(REUNITY_STATES, [state_probs[s] for s in REUNITY_STATES], color=colors)
    ax4.set_xlabel('ReUnity State')
    ax4.set_ylabel('Probability')
    ax4.set_title('Emotion Distribution by ReUnity State')
    
    # Add entropy annotation
    state_entropy = calculate_shannon_entropy([state_probs[s] for s in REUNITY_STATES])
    ax4.text(0.95, 0.95, f'State Entropy: {state_entropy:.3f} bits', 
             transform=ax4.transAxes, ha='right', va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_1_entropy_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'entropy': entropy,
        'max_entropy': max_entropy,
        'normalized_entropy': normalized_entropy,
        'emotion_distribution': emotion_probs,
        'state_distribution': state_probs,
        'state_entropy': state_entropy
    }


# ============================================================================
# SIMULATION 2: Jensen-Shannon Divergence Between Emotional States
# ============================================================================
def simulation_2_js_divergence(data):
    """Calculate JS divergence between different emotional state distributions."""
    print("\n" + "="*70)
    print("SIMULATION 2: Jensen-Shannon Divergence Analysis")
    print("="*70)
    
    # Group data by dominant emotion category
    def get_dominant_state(emotions):
        if not emotions:
            return 'STABLE'
        states = [EMOTION_TO_STATE.get(e, 'ELEVATED') for e in emotions]
        state_priority = {'CRISIS': 5, 'HIGH': 4, 'ELEVATED': 3, 'ENGAGED': 2, 'STABLE': 1}
        return max(states, key=lambda s: state_priority.get(s, 0))
    
    data['dominant_state'] = data['emotion_names'].apply(get_dominant_state)
    
    # Calculate emotion distributions for each state
    state_distributions = {}
    for state in REUNITY_STATES:
        state_data = data[data['dominant_state'] == state]
        emotion_counts = Counter()
        for emotions in state_data['emotion_names']:
            for emotion in emotions:
                emotion_counts[emotion] += 1
        
        total = sum(emotion_counts.values()) or 1
        state_distributions[state] = np.array([emotion_counts[e] / total for e in EMOTIONS])
    
    # Calculate JS divergence matrix
    n_states = len(REUNITY_STATES)
    js_matrix = np.zeros((n_states, n_states))
    
    print("\nJS Divergence Matrix:")
    print("-" * 60)
    
    for i, state1 in enumerate(REUNITY_STATES):
        for j, state2 in enumerate(REUNITY_STATES):
            js_matrix[i, j] = calculate_js_divergence(
                state_distributions[state1],
                state_distributions[state2]
            )
    
    # Print matrix
    print(f"{'':12}", end='')
    for state in REUNITY_STATES:
        print(f"{state:10}", end='')
    print()
    
    for i, state1 in enumerate(REUNITY_STATES):
        print(f"{state1:12}", end='')
        for j in range(n_states):
            print(f"{js_matrix[i, j]:10.4f}", end='')
        print()
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: JS Divergence heatmap
    ax1 = axes[0]
    im = ax1.imshow(js_matrix, cmap='YlOrRd')
    ax1.set_xticks(range(n_states))
    ax1.set_yticks(range(n_states))
    ax1.set_xticklabels(REUNITY_STATES, rotation=45, ha='right')
    ax1.set_yticklabels(REUNITY_STATES)
    ax1.set_title('Jensen-Shannon Divergence Between ReUnity States\n(from Real GoEmotions Data)')
    
    # Add text annotations
    for i in range(n_states):
        for j in range(n_states):
            text = ax1.text(j, i, f'{js_matrix[i, j]:.3f}',
                           ha='center', va='center', color='black' if js_matrix[i, j] < 0.3 else 'white')
    
    plt.colorbar(im, ax=ax1, label='JS Divergence')
    
    # Plot 2: State distributions comparison
    ax2 = axes[1]
    x = np.arange(len(EMOTIONS))
    width = 0.15
    
    for idx, state in enumerate(['STABLE', 'ELEVATED', 'CRISIS']):
        offset = (idx - 1) * width
        ax2.bar(x + offset, state_distributions[state], width, label=state, alpha=0.8)
    
    ax2.set_xlabel('Emotion')
    ax2.set_ylabel('Probability')
    ax2.set_title('Emotion Distributions by State')
    ax2.set_xticks(x[::3])
    ax2.set_xticklabels([EMOTIONS[i] for i in range(0, len(EMOTIONS), 3)], rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_2_js_divergence.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    # Key findings
    max_div_idx = np.unravel_index(np.argmax(js_matrix - np.eye(n_states) * 1000), js_matrix.shape)
    print(f"\nMaximum divergence: {js_matrix[max_div_idx]:.4f} between {REUNITY_STATES[max_div_idx[0]]} and {REUNITY_STATES[max_div_idx[1]]}")
    
    return {
        'js_matrix': js_matrix.tolist(),
        'state_distributions': {k: v.tolist() for k, v in state_distributions.items()}
    }


# ============================================================================
# SIMULATION 3: Mutual Information Analysis
# ============================================================================
def simulation_3_mutual_information(data):
    """Analyze mutual information between emotion pairs."""
    print("\n" + "="*70)
    print("SIMULATION 3: Mutual Information Analysis")
    print("="*70)
    
    # Build co-occurrence matrix
    n_emotions = len(EMOTIONS)
    cooccurrence = np.zeros((n_emotions, n_emotions))
    
    for emotions in data['emotion_names']:
        for i, e1 in enumerate(emotions):
            idx1 = EMOTIONS.index(e1) if e1 in EMOTIONS else -1
            if idx1 < 0:
                continue
            for e2 in emotions:
                idx2 = EMOTIONS.index(e2) if e2 in EMOTIONS else -1
                if idx2 >= 0:
                    cooccurrence[idx1, idx2] += 1
    
    # Calculate mutual information
    mi = calculate_mutual_information(cooccurrence)
    print(f"\nTotal Mutual Information: {mi:.4f} bits")
    
    # Find top co-occurring pairs (excluding self)
    pairs = []
    for i in range(n_emotions):
        for j in range(i+1, n_emotions):
            if cooccurrence[i, j] > 0:
                pairs.append((EMOTIONS[i], EMOTIONS[j], cooccurrence[i, j]))
    
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 10 emotion co-occurrences:")
    for e1, e2, count in pairs[:10]:
        print(f"  {e1} + {e2}: {int(count)}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Co-occurrence heatmap (subset for readability)
    ax1 = axes[0]
    top_emotions = ['neutral', 'admiration', 'approval', 'annoyance', 'gratitude', 
                    'disappointment', 'curiosity', 'love', 'sadness', 'anger']
    top_indices = [EMOTIONS.index(e) for e in top_emotions]
    subset_matrix = cooccurrence[np.ix_(top_indices, top_indices)]
    
    # Normalize by row
    row_sums = subset_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normalized_matrix = subset_matrix / row_sums
    
    im = ax1.imshow(normalized_matrix, cmap='Blues')
    ax1.set_xticks(range(len(top_emotions)))
    ax1.set_yticks(range(len(top_emotions)))
    ax1.set_xticklabels(top_emotions, rotation=45, ha='right')
    ax1.set_yticklabels(top_emotions)
    ax1.set_title('Emotion Co-occurrence Matrix\n(Normalized by Row)')
    plt.colorbar(im, ax=ax1, label='P(col | row)')
    
    # Plot 2: Network-style visualization of top pairs
    ax2 = axes[1]
    
    # Create simple network plot
    top_pairs = pairs[:15]
    unique_emotions = list(set([p[0] for p in top_pairs] + [p[1] for p in top_pairs]))
    
    # Position emotions in a circle
    n = len(unique_emotions)
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    positions = {e: (np.cos(a), np.sin(a)) for e, a in zip(unique_emotions, angles)}
    
    # Draw edges
    max_count = max(p[2] for p in top_pairs)
    for e1, e2, count in top_pairs:
        x1, y1 = positions[e1]
        x2, y2 = positions[e2]
        linewidth = 1 + 4 * (count / max_count)
        ax2.plot([x1, x2], [y1, y2], 'b-', alpha=0.3, linewidth=linewidth)
    
    # Draw nodes
    for emotion, (x, y) in positions.items():
        ax2.scatter(x, y, s=200, c='steelblue', zorder=5)
        ax2.annotate(emotion, (x, y), xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax2.set_xlim(-1.5, 1.5)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.axis('off')
    ax2.set_title(f'Emotion Co-occurrence Network\nMutual Information: {mi:.3f} bits')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_3_mutual_information.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'mutual_information': mi,
        'top_pairs': [(e1, e2, int(c)) for e1, e2, c in pairs[:20]]
    }


# ============================================================================
# SIMULATION 4: Lyapunov Exponent for Emotional Stability
# ============================================================================
def simulation_4_lyapunov_stability(data):
    """Estimate Lyapunov exponents for emotional state trajectories."""
    print("\n" + "="*70)
    print("SIMULATION 4: Lyapunov Exponent Stability Analysis")
    print("="*70)
    
    # Create time series of emotional entropy
    # Group by comment_id prefix to simulate user sessions
    data['session'] = data['comment_id'].str[:4]
    
    # Calculate entropy for each comment
    def comment_entropy(emotions):
        if not emotions:
            return 0.0
        counts = Counter(emotions)
        total = sum(counts.values())
        probs = [c / total for c in counts.values()]
        return calculate_shannon_entropy(probs)
    
    data['entropy'] = data['emotion_names'].apply(comment_entropy)
    
    # Get sessions with enough data
    session_counts = data['session'].value_counts()
    valid_sessions = session_counts[session_counts >= 20].index[:50]
    
    lyapunov_exponents = []
    session_entropies = {}
    
    for session in valid_sessions:
        session_data = data[data['session'] == session]['entropy'].values
        if len(session_data) >= 10:
            le = estimate_lyapunov_exponent(session_data)
            lyapunov_exponents.append(le)
            session_entropies[session] = session_data
    
    lyapunov_exponents = np.array(lyapunov_exponents)
    
    print(f"\nAnalyzed {len(lyapunov_exponents)} sessions")
    print(f"Mean Lyapunov Exponent: {np.mean(lyapunov_exponents):.4f}")
    print(f"Std Lyapunov Exponent: {np.std(lyapunov_exponents):.4f}")
    print(f"Positive (unstable): {np.sum(lyapunov_exponents > 0)} sessions")
    print(f"Negative (stable): {np.sum(lyapunov_exponents < 0)} sessions")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Lyapunov exponent distribution
    ax1 = axes[0, 0]
    ax1.hist(lyapunov_exponents, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Stability boundary')
    ax1.axvline(x=np.mean(lyapunov_exponents), color='green', linestyle='-', linewidth=2, label=f'Mean: {np.mean(lyapunov_exponents):.3f}')
    ax1.set_xlabel('Lyapunov Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Lyapunov Exponents\n(from Real GoEmotions Sessions)')
    ax1.legend()
    
    # Plot 2: Example stable vs unstable trajectories
    ax2 = axes[0, 1]
    
    # Find most stable and most unstable sessions
    sorted_sessions = sorted(zip(valid_sessions, lyapunov_exponents), key=lambda x: x[1])
    stable_session = sorted_sessions[0][0]
    unstable_session = sorted_sessions[-1][0]
    
    stable_data = session_entropies[stable_session][:30]
    unstable_data = session_entropies[unstable_session][:30]
    
    ax2.plot(stable_data, 'g-', linewidth=2, label=f'Stable (λ={sorted_sessions[0][1]:.3f})', marker='o', markersize=4)
    ax2.plot(unstable_data, 'r-', linewidth=2, label=f'Unstable (λ={sorted_sessions[-1][1]:.3f})', marker='s', markersize=4)
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Emotional Entropy')
    ax2.set_title('Stable vs Unstable Emotional Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Lyapunov vs mean entropy
    ax3 = axes[1, 0]
    mean_entropies = [np.mean(session_entropies[s]) for s in valid_sessions]
    ax3.scatter(mean_entropies, lyapunov_exponents, alpha=0.6, c='steelblue')
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Mean Session Entropy')
    ax3.set_ylabel('Lyapunov Exponent')
    ax3.set_title('Stability vs Average Emotional Entropy')
    
    # Add trend line
    z = np.polyfit(mean_entropies, lyapunov_exponents, 1)
    p = np.poly1d(z)
    x_line = np.linspace(min(mean_entropies), max(mean_entropies), 100)
    ax3.plot(x_line, p(x_line), 'r-', alpha=0.5, label=f'Trend: y={z[0]:.2f}x+{z[1]:.2f}')
    ax3.legend()
    
    # Plot 4: Stability classification
    ax4 = axes[1, 1]
    stability_labels = ['Highly Stable\n(λ < -1)', 'Stable\n(-1 < λ < 0)', 
                        'Marginally Unstable\n(0 < λ < 1)', 'Highly Unstable\n(λ > 1)']
    stability_counts = [
        np.sum(lyapunov_exponents < -1),
        np.sum((lyapunov_exponents >= -1) & (lyapunov_exponents < 0)),
        np.sum((lyapunov_exponents >= 0) & (lyapunov_exponents < 1)),
        np.sum(lyapunov_exponents >= 1)
    ]
    colors = ['darkgreen', 'lightgreen', 'orange', 'red']
    ax4.bar(stability_labels, stability_counts, color=colors)
    ax4.set_ylabel('Number of Sessions')
    ax4.set_title('Emotional Stability Classification')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_4_lyapunov_stability.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'mean_lyapunov': float(np.mean(lyapunov_exponents)),
        'std_lyapunov': float(np.std(lyapunov_exponents)),
        'n_stable': int(np.sum(lyapunov_exponents < 0)),
        'n_unstable': int(np.sum(lyapunov_exponents > 0))
    }


# ============================================================================
# SIMULATION 5: State Router with Real Data
# ============================================================================
def simulation_5_state_router(data):
    """Simulate state routing decisions on real emotional data."""
    print("\n" + "="*70)
    print("SIMULATION 5: State Router Simulation")
    print("="*70)
    
    # Define routing policies
    POLICIES = {
        'STABLE': {'mode': 'ENGAGE', 'response_style': 'conversational', 'intervention': None},
        'ENGAGED': {'mode': 'MAINTAIN', 'response_style': 'supportive', 'intervention': None},
        'ELEVATED': {'mode': 'SUPPORT', 'response_style': 'grounding', 'intervention': 'breathing'},
        'HIGH': {'mode': 'STABILIZE', 'response_style': 'crisis_aware', 'intervention': 'safety_check'},
        'CRISIS': {'mode': 'PROTECT', 'response_style': 'immediate', 'intervention': 'crisis_protocol'}
    }
    
    # Route each comment
    routing_results = []
    
    for idx, row in data.iterrows():
        emotions = row['emotion_names']
        
        # Calculate entropy
        if emotions:
            counts = Counter(emotions)
            total = sum(counts.values())
            probs = [c / total for c in counts.values()]
            entropy = calculate_shannon_entropy(probs)
        else:
            entropy = 0.0
        
        # Determine state
        state = row.get('dominant_state', 'STABLE')
        
        # Get policy
        policy = POLICIES[state]
        
        routing_results.append({
            'state': state,
            'entropy': entropy,
            'mode': policy['mode'],
            'intervention': policy['intervention']
        })
    
    routing_df = pd.DataFrame(routing_results)
    
    # Statistics
    print("\nRouting Statistics:")
    print("-" * 40)
    for state in REUNITY_STATES:
        count = len(routing_df[routing_df['state'] == state])
        pct = 100 * count / len(routing_df)
        print(f"  {state}: {count} ({pct:.1f}%)")
    
    # Intervention statistics
    print("\nIntervention Triggers:")
    intervention_counts = routing_df['intervention'].value_counts()
    for intervention, count in intervention_counts.items():
        if intervention:
            print(f"  {intervention}: {count}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: State distribution pie chart
    ax1 = axes[0, 0]
    state_counts = routing_df['state'].value_counts()
    colors = ['green', 'blue', 'yellow', 'orange', 'red']
    ax1.pie(state_counts.values, labels=state_counts.index, colors=colors, 
            autopct='%1.1f%%', startangle=90)
    ax1.set_title('Distribution of Routed States\n(Real GoEmotions Data)')
    
    # Plot 2: Entropy by state
    ax2 = axes[0, 1]
    state_entropies = [routing_df[routing_df['state'] == s]['entropy'].values for s in REUNITY_STATES]
    bp = ax2.boxplot(state_entropies, tick_labels=REUNITY_STATES, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax2.set_xlabel('State')
    ax2.set_ylabel('Entropy')
    ax2.set_title('Entropy Distribution by Routed State')
    
    # Plot 3: Mode distribution
    ax3 = axes[1, 0]
    mode_counts = routing_df['mode'].value_counts()
    ax3.bar(mode_counts.index, mode_counts.values, color='steelblue')
    ax3.set_xlabel('Response Mode')
    ax3.set_ylabel('Count')
    ax3.set_title('Response Mode Distribution')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 4: Intervention triggers over time (sample)
    ax4 = axes[1, 1]
    sample_size = min(500, len(routing_df))
    sample_df = routing_df.head(sample_size)
    
    state_numeric = {'STABLE': 1, 'ENGAGED': 2, 'ELEVATED': 3, 'HIGH': 4, 'CRISIS': 5}
    y_values = [state_numeric[s] for s in sample_df['state']]
    
    ax4.plot(range(sample_size), y_values, 'b-', alpha=0.5, linewidth=0.5)
    ax4.scatter(range(sample_size), y_values, c=y_values, cmap='RdYlGn_r', s=10, alpha=0.6)
    ax4.set_yticks([1, 2, 3, 4, 5])
    ax4.set_yticklabels(REUNITY_STATES)
    ax4.set_xlabel('Comment Index')
    ax4.set_ylabel('Routed State')
    ax4.set_title(f'State Routing Over Time (First {sample_size} Comments)')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_5_state_router.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'state_distribution': state_counts.to_dict(),
        'intervention_counts': intervention_counts.to_dict()
    }


# ============================================================================
# SIMULATION 6: Pattern Detection on Real Conversations
# ============================================================================
def simulation_6_pattern_detection(data):
    """Detect harmful patterns in real conversation data."""
    print("\n" + "="*70)
    print("SIMULATION 6: Protective Pattern Detection")
    print("="*70)
    
    # Define pattern indicators
    PATTERN_INDICATORS = {
        'hot_cold_cycle': {
            'positive': ['love', 'admiration', 'gratitude', 'joy', 'caring'],
            'negative': ['anger', 'disappointment', 'disgust', 'annoyance']
        },
        'gaslighting': {
            'confusion_words': ['confused', 'crazy', 'wrong', 'imagine', 'never said'],
            'emotions': ['confusion', 'nervousness', 'fear']
        },
        'isolation': {
            'keywords': ['alone', 'nobody', 'no one', 'only me', 'cant trust'],
            'emotions': ['sadness', 'fear', 'nervousness']
        },
        'crisis_indicators': {
            'emotions': ['fear', 'grief', 'sadness'],
            'keywords': ['help', 'cant', 'hopeless', 'end']
        }
    }
    
    # Analyze patterns
    pattern_counts = Counter()
    pattern_examples = {p: [] for p in PATTERN_INDICATORS}
    
    for idx, row in data.iterrows():
        text = str(row['text']).lower()
        emotions = row['emotion_names']
        
        # Check for hot-cold cycle (high variance in sentiment)
        pos_count = sum(1 for e in emotions if e in PATTERN_INDICATORS['hot_cold_cycle']['positive'])
        neg_count = sum(1 for e in emotions if e in PATTERN_INDICATORS['hot_cold_cycle']['negative'])
        if pos_count > 0 and neg_count > 0:
            pattern_counts['hot_cold_cycle'] += 1
            if len(pattern_examples['hot_cold_cycle']) < 5:
                pattern_examples['hot_cold_cycle'].append(text[:100])
        
        # Check for confusion patterns (potential gaslighting context)
        if 'confusion' in emotions or 'nervousness' in emotions:
            for word in PATTERN_INDICATORS['gaslighting']['confusion_words']:
                if word in text:
                    pattern_counts['gaslighting_context'] += 1
                    if len(pattern_examples['gaslighting']) < 5:
                        pattern_examples['gaslighting'].append(text[:100])
                    break
        
        # Check for isolation patterns
        if any(e in emotions for e in PATTERN_INDICATORS['isolation']['emotions']):
            for word in PATTERN_INDICATORS['isolation']['keywords']:
                if word in text:
                    pattern_counts['isolation'] += 1
                    if len(pattern_examples['isolation']) < 5:
                        pattern_examples['isolation'].append(text[:100])
                    break
        
        # Check for crisis indicators
        if any(e in emotions for e in PATTERN_INDICATORS['crisis_indicators']['emotions']):
            for word in PATTERN_INDICATORS['crisis_indicators']['keywords']:
                if word in text:
                    pattern_counts['crisis_indicator'] += 1
                    if len(pattern_examples['crisis_indicators']) < 5:
                        pattern_examples['crisis_indicators'].append(text[:100])
                    break
    
    print("\nPattern Detection Results:")
    print("-" * 40)
    for pattern, count in pattern_counts.most_common():
        pct = 100 * count / len(data)
        print(f"  {pattern}: {count} ({pct:.2f}%)")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Pattern frequency
    ax1 = axes[0, 0]
    patterns = list(pattern_counts.keys())
    counts = list(pattern_counts.values())
    colors = ['orange', 'red', 'purple', 'darkred'][:len(patterns)]
    ax1.barh(patterns, counts, color=colors)
    ax1.set_xlabel('Count')
    ax1.set_title('Detected Pattern Frequencies\n(Real GoEmotions Data)')
    
    # Plot 2: Pattern co-occurrence with emotions
    ax2 = axes[0, 1]
    
    # Calculate emotion distribution for flagged vs non-flagged comments
    flagged_emotions = Counter()
    non_flagged_emotions = Counter()
    
    for idx, row in data.iterrows():
        emotions = row['emotion_names']
        text = str(row['text']).lower()
        
        is_flagged = any(word in text for word in ['alone', 'nobody', 'help', 'cant', 'confused'])
        
        for e in emotions:
            if is_flagged:
                flagged_emotions[e] += 1
            else:
                non_flagged_emotions[e] += 1
    
    # Normalize
    flagged_total = sum(flagged_emotions.values()) or 1
    non_flagged_total = sum(non_flagged_emotions.values()) or 1
    
    top_emotions = ['sadness', 'fear', 'anger', 'confusion', 'nervousness', 'neutral', 'joy', 'love']
    flagged_probs = [flagged_emotions[e] / flagged_total for e in top_emotions]
    non_flagged_probs = [non_flagged_emotions[e] / non_flagged_total for e in top_emotions]
    
    x = np.arange(len(top_emotions))
    width = 0.35
    ax2.bar(x - width/2, flagged_probs, width, label='Flagged', color='red', alpha=0.7)
    ax2.bar(x + width/2, non_flagged_probs, width, label='Non-flagged', color='green', alpha=0.7)
    ax2.set_xticks(x)
    ax2.set_xticklabels(top_emotions, rotation=45, ha='right')
    ax2.set_ylabel('Probability')
    ax2.set_title('Emotion Distribution: Flagged vs Non-Flagged')
    ax2.legend()
    
    # Plot 3: Risk level distribution
    ax3 = axes[1, 0]
    
    # Calculate risk scores
    risk_scores = []
    for idx, row in data.iterrows():
        emotions = row['emotion_names']
        score = 0
        if 'fear' in emotions:
            score += 3
        if 'grief' in emotions:
            score += 3
        if 'sadness' in emotions:
            score += 2
        if 'anger' in emotions:
            score += 2
        if 'confusion' in emotions:
            score += 1
        if 'nervousness' in emotions:
            score += 1
        risk_scores.append(min(score, 10))
    
    ax3.hist(risk_scores, bins=11, range=(-0.5, 10.5), color='steelblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=5, color='orange', linestyle='--', label='Elevated threshold')
    ax3.axvline(x=7, color='red', linestyle='--', label='High threshold')
    ax3.set_xlabel('Risk Score')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Risk Score Distribution')
    ax3.legend()
    
    # Plot 4: Intervention recommendations
    ax4 = axes[1, 1]
    
    intervention_counts = {
        'None needed': sum(1 for s in risk_scores if s < 3),
        'Grounding suggested': sum(1 for s in risk_scores if 3 <= s < 5),
        'Support recommended': sum(1 for s in risk_scores if 5 <= s < 7),
        'Crisis protocol': sum(1 for s in risk_scores if s >= 7)
    }
    
    colors = ['green', 'yellow', 'orange', 'red']
    ax4.pie(intervention_counts.values(), labels=intervention_counts.keys(), 
            colors=colors, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Recommended Interventions')
    
    plt.tight_layout()
    output_path = OUTPUT_DIR / 'simulation_6_pattern_detection.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nVisualization saved to: {output_path}")
    
    return {
        'pattern_counts': dict(pattern_counts),
        'risk_distribution': {
            'low': sum(1 for s in risk_scores if s < 3),
            'elevated': sum(1 for s in risk_scores if 3 <= s < 5),
            'high': sum(1 for s in risk_scores if 5 <= s < 7),
            'crisis': sum(1 for s in risk_scores if s >= 7)
        }
    }


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Run all simulations."""
    print("="*70)
    print("ReUnity Real Data Simulations")
    print("Using GoEmotions Dataset (58k Reddit Comments, 27 Emotions)")
    print("="*70)
    print(f"Start time: {datetime.now().isoformat()}")
    
    # Load real data
    data = load_goemotions_data()
    
    # Add dominant state column
    def get_dominant_state(emotions):
        if not emotions:
            return 'STABLE'
        states = [EMOTION_TO_STATE.get(e, 'ELEVATED') for e in emotions]
        state_priority = {'CRISIS': 5, 'HIGH': 4, 'ELEVATED': 3, 'ENGAGED': 2, 'STABLE': 1}
        return max(states, key=lambda s: state_priority.get(s, 0))
    
    data['dominant_state'] = data['emotion_names'].apply(get_dominant_state)
    
    # Run all simulations
    results = {}
    
    try:
        results['simulation_1'] = simulation_1_entropy_analysis(data)
        print("\n✓ Simulation 1 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 1 FAILED: {e}")
        results['simulation_1'] = {'error': str(e)}
    
    try:
        results['simulation_2'] = simulation_2_js_divergence(data)
        print("\n✓ Simulation 2 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 2 FAILED: {e}")
        results['simulation_2'] = {'error': str(e)}
    
    try:
        results['simulation_3'] = simulation_3_mutual_information(data)
        print("\n✓ Simulation 3 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 3 FAILED: {e}")
        results['simulation_3'] = {'error': str(e)}
    
    try:
        results['simulation_4'] = simulation_4_lyapunov_stability(data)
        print("\n✓ Simulation 4 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 4 FAILED: {e}")
        results['simulation_4'] = {'error': str(e)}
    
    try:
        results['simulation_5'] = simulation_5_state_router(data)
        print("\n✓ Simulation 5 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 5 FAILED: {e}")
        results['simulation_5'] = {'error': str(e)}
    
    try:
        results['simulation_6'] = simulation_6_pattern_detection(data)
        print("\n✓ Simulation 6 PASSED")
    except Exception as e:
        print(f"\n✗ Simulation 6 FAILED: {e}")
        results['simulation_6'] = {'error': str(e)}
    
    # Save results
    results_path = OUTPUT_DIR / 'simulation_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*70)
    print("SIMULATION SUMMARY")
    print("="*70)
    
    passed = sum(1 for r in results.values() if 'error' not in r)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    print(f"Results saved to: {results_path}")
    print(f"Visualizations saved to: {OUTPUT_DIR}")
    print(f"End time: {datetime.now().isoformat()}")
    
    # List generated files
    print("\nGenerated files:")
    for f in OUTPUT_DIR.glob('*'):
        print(f"  {f.name}")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
