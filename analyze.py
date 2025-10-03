import json
import re
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr

def extract_decision(text: str) -> Optional[str]:
    """Extract decision using regex, returns 'guilty' or 'not_guilty' or None"""
    decision_match = re.search(r'<decision>\s*(.*?)\s*</decision>', text, re.IGNORECASE)
    if not decision_match:
        return None

    decision = decision_match.group(1).strip()
    if re.search(r'\bnot\s+guilty\b', decision, re.IGNORECASE):
        return 'not_guilty'
    elif re.search(r'\bguilty\b', decision, re.IGNORECASE):
        return 'guilty'
    return None

def extract_belief(text: str) -> Optional[float]:
    """Extract belief and convert to float between 0 and 1"""
    belief_match = re.search(r'<belief>\s*(.*?)\s*</belief>', text, re.IGNORECASE)
    if not belief_match:
        return None

    belief_str = belief_match.group(1).strip()
    try:
        belief = float(belief_str)
        # Ensure it's between 0 and 1
        if 0 <= belief <= 1:
            return belief
        # If it's a percentage, convert
        elif 0 < belief <= 100:
            return belief / 100
        return None
    except ValueError:
        return None

def load_simulation(filepath: str) -> List[Dict]:
    """Load simulation results from JSON file"""
    print(f"Loading simulation results from {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)

def is_valid_simulation(sim: Dict) -> bool:
    """Check if simulation is valid by verifying <arguments> tag exists in response"""
    response = sim.get('response', '')
    return bool(re.search(r'<arguments>.*?</arguments>', response, re.IGNORECASE | re.DOTALL))

def compare_simulations(bayesian_path: str, prospect_path: str):
    """Compare Bayesian and Prospect Theory simulation results"""
    bayesian = load_simulation(bayesian_path)
    prospect = load_simulation(prospect_path)

    print(f"Loaded {len(bayesian)} Bayesian simulations and {len(prospect)} Prospect Theory simulations\n")

    # Statistics
    stats = {
        'total': 0,
        'valid': 0,
        'invalid_bayesian': 0,
        'invalid_prospect': 0,
        'both_extracted': 0,
        'decision_match': 0,
        'decision_differ': 0,
        'belief_differences': [],
        'bayesian_guilty': 0,
        'prospect_guilty': 0,
        'bayesian_not_guilty': 0,
        'prospect_not_guilty': 0,
        'bayesian_beliefs': [],
        'prospect_beliefs': [],
    }

    # Compare one-by-one
    min_len = min(len(bayesian), len(prospect))

    for i in range(min_len):
        b_sim = bayesian[i]
        p_sim = prospect[i]

        # Check if both simulations are valid
        b_valid = is_valid_simulation(b_sim)
        p_valid = is_valid_simulation(p_sim)

        if not b_valid:
            stats['invalid_bayesian'] += 1
        if not p_valid:
            stats['invalid_prospect'] += 1

        # Skip if either simulation is invalid
        if not (b_valid and p_valid):
            continue

        stats['valid'] += 1

        b_response = b_sim.get('simulation_response', '')
        p_response = p_sim.get('simulation_response', '')

        b_decision = extract_decision(b_response)
        p_decision = extract_decision(p_response)

        b_belief = extract_belief(b_response)
        p_belief = extract_belief(p_response)

        stats['total'] += 1

        if b_decision and p_decision:
            stats['both_extracted'] += 1

            if b_decision == 'guilty':
                stats['bayesian_guilty'] += 1
            else:
                stats['bayesian_not_guilty'] += 1

            if p_decision == 'guilty':
                stats['prospect_guilty'] += 1
            else:
                stats['prospect_not_guilty'] += 1

            if b_decision == p_decision:
                stats['decision_match'] += 1
            else:
                stats['decision_differ'] += 1
                print(f"\n[Case {i}] DECISION DIFFERS:")
                print(f"  Bayesian: {b_decision}")
                print(f"  Prospect: {p_decision}")
                print(f"  Bayesian belief: {b_belief}")
                print(f"  Prospect belief: {p_belief}")

        if b_belief is not None and p_belief is not None:
            diff = abs(b_belief - p_belief)
            stats['belief_differences'].append(diff)
            stats['bayesian_beliefs'].append(b_belief)
            stats['prospect_beliefs'].append(p_belief)

            if diff > 0.1:  # Significant difference threshold
                print(f"\n[Case {i}] SIGNIFICANT BELIEF DIFFERENCE:")
                print(f"  Bayesian belief: {b_belief}")
                print(f"  Prospect belief: {p_belief}")
                print(f"  Difference: {diff:.3f}")

    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total cases: {min_len}")
    print(f"Valid simulations (both have <arguments>): {stats['valid']}")
    print(f"Invalid Bayesian simulations: {stats['invalid_bayesian']}")
    print(f"Invalid Prospect simulations: {stats['invalid_prospect']}")
    print(f"\nCases with both decisions extracted: {stats['both_extracted']}")
    print(f"\nDecision Comparison:")
    print(f"  Matching decisions: {stats['decision_match']}")
    print(f"  Differing decisions: {stats['decision_differ']}")
    print(f"\nBayesian Results:")
    print(f"  Guilty: {stats['bayesian_guilty']}")
    print(f"  Not Guilty: {stats['bayesian_not_guilty']}")
    print(f"\nProspect Theory Results:")
    print(f"  Guilty: {stats['prospect_guilty']}")
    print(f"  Not Guilty: {stats['prospect_not_guilty']}")

    if stats['belief_differences']:
        avg_diff = sum(stats['belief_differences']) / len(stats['belief_differences'])
        max_diff = max(stats['belief_differences'])
        print(f"\nBelief Differences:")
        print(f"  Average: {avg_diff:.3f}")
        print(f"  Maximum: {max_diff:.3f}")
        print(f"  Cases with beliefs extracted: {len(stats['belief_differences'])}")

    return stats

def plot_correlation(bayesian_beliefs: List[float], prospect_beliefs: List[float], output_file: str = "belief_correlation.png"):
    """Plot correlation between Bayesian and Prospect Theory beliefs"""
    if not bayesian_beliefs or not prospect_beliefs:
        print("No beliefs to plot")
        return

    # Calculate correlation
    correlation, p_value = pearsonr(bayesian_beliefs, prospect_beliefs)

    # Create figure
    plt.figure(figsize=(10, 8))

    # Scatter plot
    plt.scatter(bayesian_beliefs, prospect_beliefs, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)

    # Add diagonal line (y=x)
    min_val = min(min(bayesian_beliefs), min(prospect_beliefs))
    max_val = max(max(bayesian_beliefs), max(prospect_beliefs))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (perfect agreement)')

    # Add regression line
    z = np.polyfit(bayesian_beliefs, prospect_beliefs, 1)
    p = np.poly1d(z)
    plt.plot(bayesian_beliefs, p(bayesian_beliefs), 'b-', linewidth=2,
             label=f'Linear fit: y={z[0]:.3f}x+{z[1]:.3f}')

    # Labels and title
    plt.xlabel('Bayesian Posterior Belief', fontsize=12)
    plt.ylabel('Prospect Theory Posterior Belief', fontsize=12)
    plt.title(f'Correlation between Bayesian and Prospect Theory Beliefs\nPearson r = {correlation:.3f}, p = {p_value:.4f}',
              fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set equal aspect ratio and limits
    plt.xlim(min_val - 0.05, max_val + 0.05)
    plt.ylim(min_val - 0.05, max_val + 0.05)
    plt.gca().set_aspect('equal', adjustable='box')

    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\nCorrelation plot saved to {output_file}")
    print(f"Pearson correlation: r = {correlation:.3f}, p-value = {p_value:.4f}")

    plt.close()

if __name__ == "__main__":
    bayesian_path = "simulations/controlled_bayesian_simulation.json"
    prospect_path = "simulations/controlled_prospect_simulation.json"

    stats = compare_simulations(bayesian_path, prospect_path)

    # Plot correlation
    if stats['bayesian_beliefs'] and stats['prospect_beliefs']:
        plot_correlation(stats['bayesian_beliefs'], stats['prospect_beliefs'])
