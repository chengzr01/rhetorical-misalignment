#!/usr/bin/env python3
"""
Demographics Analysis Script

This script analyzes annotator demographics from the annotation results,
using DeepSeek-V3.1 via OpenRouter to provide insights and patterns.
"""

import os
import json
import glob
from collections import Counter
from typing import Dict, List, Any
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec


def load_annotation_files(results_dir: str = "../results/usmle_sample") -> List[Dict[str, Any]]:
    """Load all annotation JSON files from the results directory."""
    # Get the directory relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(script_dir, results_dir)

    pattern = os.path.join(full_path, "*.json")
    files = glob.glob(pattern)

    annotations = []
    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                if 'demographics' in data:
                    annotations.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {file_path}: {e}")

    print(f"Loaded {len(annotations)} annotation files with demographics data")
    return annotations


def extract_demographics(annotations: List[Dict[str, Any]]) -> Dict[str, List]:
    """Extract demographics fields from annotations."""
    demographics = {
        'age': [],
        'expertise': [],
        'practice_location': [],
        'race': [],
        'sex': [],
        'years_of_practice': []
    }

    for ann in annotations:
        demo = ann.get('demographics', {})
        for key in demographics.keys():
            value = demo.get(key)
            if value:
                demographics[key].append(value)

    return demographics


def get_demographics_summary(demographics: Dict[str, List]) -> Dict[str, Any]:
    """Generate summary statistics for demographics."""
    summary = {}

    for key, values in demographics.items():
        counter = Counter(values)
        total = len(values)

        # Get top 10 most common values
        top_values = counter.most_common(10)

        summary[key] = {
            'total_count': total,
            'unique_values': len(counter),
            'top_10': [
                {
                    'value': value,
                    'count': count,
                    'percentage': round(count / total * 100, 2)
                }
                for value, count in top_values
            ],
            'all_values': dict(counter)
        }

    return summary


def call_deepseek_api(prompt: str, api_key: str = None) -> str:
    """Call DeepSeek-V3.1 API via OpenRouter."""
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY environment variable not set. "
            "Please set it with: export OPENROUTER_API_KEY='your-key-here'"
        )

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/persuasive-misalignment",
        "X-Title": "Demographics Analysis",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-chat",  # DeepSeek-V3
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "max_tokens": 4000
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response: {e.response.text}")
        raise


def analyze_age_distribution(ages: List[str]) -> str:
    """Analyze age distribution with LLM."""
    age_counter = Counter(ages)
    age_data = json.dumps(dict(age_counter), indent=2)

    prompt = f"""Analyze the following age distribution data from {len(ages)} medical annotators:

{age_data}

Please provide:
1. Key patterns in the age distribution (note that some values are exact ages like "32", while others are ranges like "25-34" or "45-54")
2. The predominant age groups
3. Any potential biases or concerns about age representation
4. How the age distribution might affect the annotation quality or perspectives

Be concise but insightful."""

    return call_deepseek_api(prompt)


def analyze_expertise(expertise_list: List[str]) -> str:
    """Analyze expertise distribution with LLM."""
    expertise_counter = Counter(expertise_list)
    # Get top 30 for analysis
    top_expertise = dict(expertise_counter.most_common(30))
    expertise_data = json.dumps(top_expertise, indent=2)

    prompt = f"""Analyze the following expertise/specialty distribution from {len(expertise_list)} medical annotators (showing top 30):

{expertise_data}

Note: The data contains various formats and cases (e.g., "internal medicine", "Internal Medicine", "internal medicine ").

Please provide:
1. Major categories of medical expertise represented
2. Issues with data standardization and consistency
3. The balance between physicians, nurses, and other healthcare workers
4. Any gaps in medical expertise coverage
5. How this expertise distribution might affect annotation quality

Be analytical and specific."""

    return call_deepseek_api(prompt)


def analyze_practice_location(locations: List[str]) -> str:
    """Analyze practice location distribution with LLM."""
    location_counter = Counter(locations)
    top_locations = dict(location_counter.most_common(30))
    location_data = json.dumps(top_locations, indent=2)

    prompt = f"""Analyze the following practice location distribution from {len(locations)} medical annotators (showing top 30):

{location_data}

Please provide:
1. Geographic distribution patterns (regions, states, international)
2. Urban vs. rural representation (if discernible)
3. Standardization issues (e.g., "USA" vs "United States" vs specific cities)
4. Any geographic biases in the data
5. How location diversity might affect annotation perspectives

Be specific about geographic patterns."""

    return call_deepseek_api(prompt)


def analyze_race_and_sex(race_list: List[str], sex_list: List[str]) -> str:
    """Analyze race and sex distribution with LLM."""
    race_counter = Counter(race_list)
    sex_counter = Counter(sex_list)

    race_data = json.dumps(dict(race_counter), indent=2)
    sex_data = json.dumps(dict(sex_counter), indent=2)

    prompt = f"""Analyze the demographic diversity of {len(race_list)} medical annotators:

Race/Ethnicity Distribution:
{race_data}

Sex/Gender Distribution:
{sex_data}

Please provide:
1. Assessment of racial/ethnic diversity
2. Assessment of gender balance
3. Comparison to actual demographics of US healthcare workers (if you know)
4. Potential biases or limitations due to demographic composition
5. How demographic diversity might affect annotation quality and perspectives

Be thoughtful about diversity and representation."""

    return call_deepseek_api(prompt)


def analyze_years_of_practice(years_list: List[str]) -> str:
    """Analyze years of practice distribution with LLM."""
    years_counter = Counter(years_list)
    years_data = json.dumps(dict(years_counter.most_common(20)), indent=2)

    prompt = f"""Analyze the years of practice distribution from {len(years_list)} medical annotators (showing top 20):

{years_data}

Please provide:
1. Distribution of experience levels (early career, mid-career, experienced)
2. The typical or median experience level
3. Balance between newer and more experienced practitioners
4. How experience distribution might affect annotation quality
5. Any concerns about experience representation

Consider that medical knowledge and practices evolve over time."""

    return call_deepseek_api(prompt)


def create_visualizations(demographics: Dict[str, List], summary: Dict[str, Any], output_dir: str = "."):
    """Create comprehensive visualizations for demographics data."""
    # Set style for presentation slides
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 18
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14
    plt.rcParams['font.weight'] = 'normal'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    # Create output directory for plots
    plots_dir = os.path.join(output_dir, "demographics_plots")
    os.makedirs(plots_dir, exist_ok=True)

    print(f"\nGenerating visualizations in {plots_dir}/...")

    # 1. Age Distribution
    create_age_visualization(demographics['age'], summary['age'], plots_dir)

    # 2. Expertise Distribution
    create_expertise_visualization(demographics['expertise'], summary['expertise'], plots_dir)

    # 3. Race Distribution
    create_race_visualization(demographics['race'], summary['race'], plots_dir)

    # 4. Sex Distribution
    create_sex_visualization(demographics['sex'], summary['sex'], plots_dir)

    # 5. Years of Practice Distribution
    create_years_visualization(demographics['years_of_practice'], summary['years_of_practice'], plots_dir)

    print(f"✓ All visualizations saved to {plots_dir}/")
    return plots_dir


def create_age_visualization(ages: List[str], summary: Dict, output_dir: str):
    """Create age distribution visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Parse ages - only numeric ages
    numeric_ages = []
    for age in ages:
        if '-' not in age and '+' not in age:
            try:
                numeric_ages.append(int(age))
            except ValueError:
                pass

    # Plot: Numeric ages histogram with percentage
    if numeric_ages:
        counts, bins, patches = ax.hist(numeric_ages, bins=20, color='steelblue',
                                        edgecolor='black', alpha=0.7, linewidth=1.5,
                                        weights=np.ones(len(numeric_ages)) / len(numeric_ages))
        ax.axvline(np.mean(numeric_ages), color='red', linestyle='--', linewidth=3,
                   label=f'Mean: {np.mean(numeric_ages):.1f} years')
        ax.axvline(np.median(numeric_ages), color='orange', linestyle='--', linewidth=3,
                   label=f'Median: {np.median(numeric_ages):.1f} years')
        ax.set_xlabel('Age (years)', fontsize=18, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=18, fontweight='bold')
        ax.set_title('Age Distribution', fontsize=22, fontweight='bold', pad=20)
        ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.tick_params(labelsize=16)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y*100)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ age_distribution.png")


def create_expertise_visualization(expertise_list: List[str], summary: Dict, output_dir: str):
    """Create expertise distribution visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Grouped categories (manual categorization)
    categorized = categorize_expertise(expertise_list)
    categories = list(categorized.keys())
    category_counts = list(categorized.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
    wedges, texts, autotexts = ax.pie(category_counts, labels=categories, autopct='%1.1f%%',
                                        colors=colors, startangle=90, textprops={'fontsize': 16})
    ax.set_title('Expertise Categories', fontsize=22, fontweight='bold', pad=20)

    # Make percentage text more readable
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    # Make labels more readable
    for text in texts:
        text.set_fontsize(16)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'expertise_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ expertise_distribution.png")


def categorize_expertise(expertise_list: List[str]) -> Dict[str, int]:
    """Categorize expertise into broader groups."""
    categories = {
        'Physicians (Specialists)': 0,
        'Physicians (Primary Care)': 0,
        'Nursing': 0,
        'Mental Health': 0,
        'Allied Health': 0,
        'Dental/Oral': 0,
        'Other Healthcare': 0,
        'Non-Healthcare': 0
    }

    for expertise in expertise_list:
        exp_lower = expertise.lower()

        # Physicians - Specialists
        if any(x in exp_lower for x in ['cardio', 'surgery', 'oncology', 'radiology', 'anesthesia',
                                         'pathology', 'dermatology', 'urology', 'neurology', 'endocrin',
                                         'ob', 'gyn', 'ortho', 'icu', 'emergency']):
            categories['Physicians (Specialists)'] += 1
        # Physicians - Primary Care
        elif any(x in exp_lower for x in ['internal medicine', 'family medicine', 'primary care',
                                           'general practice', 'family practice', 'general medicine']):
            categories['Physicians (Primary Care)'] += 1
        # Nursing
        elif any(x in exp_lower for x in ['nurse', 'nursing', 'rn', 'lpn', 'midwife']):
            categories['Nursing'] += 1
        # Mental Health
        elif any(x in exp_lower for x in ['mental health', 'psychology', 'psycho', 'social work',
                                           'counseling', 'psychiatr']):
            categories['Mental Health'] += 1
        # Dental
        elif any(x in exp_lower for x in ['dent', 'oral', 'rdh']):
            categories['Dental/Oral'] += 1
        # Allied Health
        elif any(x in exp_lower for x in ['therapy', 'therapist', 'emt', 'pharmacy', 'pharm',
                                           'optom', 'optical', 'aide', 'home health', 'caregiv']):
            categories['Allied Health'] += 1
        # Healthcare general
        elif any(x in exp_lower for x in ['health', 'medical', 'medicine', 'clinical']):
            categories['Other Healthcare'] += 1
        else:
            categories['Non-Healthcare'] += 1

    # Remove empty categories
    return {k: v for k, v in categories.items() if v > 0}


def create_race_visualization(race_list: List[str], summary: Dict, output_dir: str):
    """Create race/ethnicity distribution visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get all race data and map to formal labels
    race_data = summary['all_values']

    # Mapping to formal labels
    race_label_map = {
        'white': 'White',
        'black_african_american': 'Black',
        'asian': 'Asian',
        'hispanic_latino': 'Hispanic',
        'two_or_more': 'Mixed or Others'
    }

    formal_labels = []
    counts = []
    for race, count in race_data.items():
        formal_labels.append(race_label_map.get(race, race.replace('_', ' ').title()))
        counts.append(count)

    # Pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, len(formal_labels)))
    wedges, texts, autotexts = ax.pie(counts, labels=formal_labels,
                                        autopct='%1.1f%%', colors=colors, startangle=90,
                                        textprops={'fontsize': 16})
    ax.set_title('Race/Ethnicity Distribution', fontsize=22, fontweight='bold', pad=20)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    for text in texts:
        text.set_fontsize(16)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'race_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ race_distribution.png")


def create_sex_visualization(sex_list: List[str], summary: Dict, output_dir: str):
    """Create sex/gender distribution visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))

    # Get sex data
    sex_data = summary['all_values']
    sexes = list(sex_data.keys())
    counts = list(sex_data.values())

    # Pie chart with better styling
    colors = ['#FF69B4', '#4169E1'] if 'female' in sexes else plt.cm.Pastel1(range(len(sexes)))
    wedges, texts, autotexts = ax.pie(counts, labels=[s.title() for s in sexes],
                                        autopct='%1.1f%%', colors=colors, startangle=90,
                                        explode=[0.05] * len(sexes), textprops={'fontsize': 16})
    ax.set_title('Sex/Gender Distribution', fontsize=22, fontweight='bold', pad=20)

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(16)

    for text in texts:
        text.set_fontsize(16)
        text.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'sex_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ sex_distribution.png")


def create_years_visualization(years_list: List[str], summary: Dict, output_dir: str):
    """Create years of practice distribution visualization."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Parse years - convert to numeric where possible
    numeric_years = []
    for year in years_list:
        try:
            # Handle cases like "2.5", "20+", etc.
            year_clean = year.replace('+', '').strip()
            numeric_years.append(float(year_clean))
        except ValueError:
            pass

    # Histogram of numeric years with percentage
    if numeric_years:
        counts, bins, patches = ax.hist(numeric_years, bins=25, color='forestgreen',
                                        edgecolor='black', alpha=0.7, linewidth=1.5,
                                        weights=np.ones(len(numeric_years)) / len(numeric_years))
        ax.axvline(np.mean(numeric_years), color='red', linestyle='--', linewidth=3,
                   label=f'Mean: {np.mean(numeric_years):.1f} years')
        ax.axvline(np.median(numeric_years), color='orange', linestyle='--', linewidth=3,
                   label=f'Median: {np.median(numeric_years):.1f} years')
        ax.set_xlabel('Years of Practice', fontsize=18, fontweight='bold')
        ax.set_ylabel('Percentage (%)', fontsize=18, fontweight='bold')
        ax.set_title('Years of Practice Distribution', fontsize=22, fontweight='bold', pad=20)
        ax.legend(fontsize=16, loc='upper right', frameon=True, shadow=True)
        ax.grid(True, alpha=0.3, linewidth=1)
        ax.tick_params(labelsize=16)

        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1f}'.format(y*100)))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'years_of_practice_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ years_of_practice_distribution.png")


def generate_overall_analysis(summary: Dict[str, Any]) -> str:
    """Generate overall demographic analysis with LLM."""
    # Create a concise summary for the LLM
    summary_text = f"""Total Annotations: {summary['age']['total_count']}

Age: {summary['age']['unique_values']} unique values
Top 3 Ages: {', '.join([f"{item['value']} ({item['percentage']}%)" for item in summary['age']['top_10'][:3]])}

Expertise: {summary['expertise']['unique_values']} unique values
Top 3: {', '.join([f"{item['value']} ({item['percentage']}%)" for item in summary['expertise']['top_10'][:3]])}

Race Distribution:
{json.dumps(summary['race']['all_values'], indent=2)}

Sex Distribution:
{json.dumps(summary['sex']['all_values'], indent=2)}

Years of Practice: {summary['years_of_practice']['unique_values']} unique values
Top 3: {', '.join([f"{item['value']} years ({item['percentage']}%)" for item in summary['years_of_practice']['top_10'][:3]])}
"""

    prompt = f"""Provide an overall assessment of the annotator pool based on these demographics:

{summary_text}

Please provide:
1. Overall quality and representativeness of the annotator pool
2. Key strengths of this annotator demographic composition
3. Key limitations or potential biases
4. Recommendations for improving annotator recruitment
5. Expected impact on annotation quality and generalizability

Be comprehensive but concise."""

    return call_deepseek_api(prompt)


def main():
    """Main function to run demographics analysis."""
    print("=" * 80)
    print("DEMOGRAPHICS ANALYSIS")
    print("=" * 80)
    print()

    # Load annotations
    print("Loading annotation files...")
    annotations = load_annotation_files()

    if not annotations:
        print("No annotation files found!")
        return

    # Extract demographics
    print("Extracting demographics data...")
    demographics = extract_demographics(annotations)

    # Generate summary
    print("Generating summary statistics...")
    summary = get_demographics_summary(demographics)

    # Print basic statistics
    print("\n" + "=" * 80)
    print("BASIC STATISTICS")
    print("=" * 80)
    for field, data in summary.items():
        print(f"\n{field.upper().replace('_', ' ')}:")
        print(f"  Total responses: {data['total_count']}")
        print(f"  Unique values: {data['unique_values']}")
        print(f"  Top 5:")
        for item in data['top_10'][:5]:
            print(f"    - {item['value']}: {item['count']} ({item['percentage']}%)")

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)
    try:
        plots_dir = create_visualizations(demographics, summary)
        print(f"\n✓ All visualizations saved successfully!")
    except Exception as e:
        print(f"\nWarning: Could not generate visualizations: {e}")
        import traceback
        traceback.print_exc()

    # LLM Analysis
    print("\n" + "=" * 80)
    print("LLM-BASED ANALYSIS (Using DeepSeek-V3)")
    print("=" * 80)

    try:
        # Check API key
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            print("\nWARNING: OPENROUTER_API_KEY not set!")
            print("Set it with: export OPENROUTER_API_KEY='your-key-here'")
            print("Skipping LLM analysis...")
            return

        # Age analysis
        print("\n" + "-" * 80)
        print("AGE DISTRIBUTION ANALYSIS")
        print("-" * 80)
        age_analysis = analyze_age_distribution(demographics['age'])
        print(age_analysis)

        # Expertise analysis
        print("\n" + "-" * 80)
        print("EXPERTISE ANALYSIS")
        print("-" * 80)
        expertise_analysis = analyze_expertise(demographics['expertise'])
        print(expertise_analysis)

        # Location analysis
        print("\n" + "-" * 80)
        print("PRACTICE LOCATION ANALYSIS")
        print("-" * 80)
        location_analysis = analyze_practice_location(demographics['practice_location'])
        print(location_analysis)

        # Race and sex analysis
        print("\n" + "-" * 80)
        print("RACE AND SEX ANALYSIS")
        print("-" * 80)
        race_sex_analysis = analyze_race_and_sex(demographics['race'], demographics['sex'])
        print(race_sex_analysis)

        # Years of practice analysis
        print("\n" + "-" * 80)
        print("YEARS OF PRACTICE ANALYSIS")
        print("-" * 80)
        years_analysis = analyze_years_of_practice(demographics['years_of_practice'])
        print(years_analysis)

        # Overall analysis
        print("\n" + "-" * 80)
        print("OVERALL ASSESSMENT")
        print("-" * 80)
        overall_analysis = generate_overall_analysis(summary)
        print(overall_analysis)

        # Save results to file
        output_file = "demographics_analysis_results.txt"
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEMOGRAPHICS ANALYSIS RESULTS\n")
            f.write("=" * 80 + "\n\n")

            f.write("BASIC STATISTICS\n")
            f.write("=" * 80 + "\n")
            for field, data in summary.items():
                f.write(f"\n{field.upper().replace('_', ' ')}:\n")
                f.write(f"  Total responses: {data['total_count']}\n")
                f.write(f"  Unique values: {data['unique_values']}\n")
                f.write(f"  Top 10:\n")
                for item in data['top_10']:
                    f.write(f"    - {item['value']}: {item['count']} ({item['percentage']}%)\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("LLM-BASED ANALYSIS\n")
            f.write("=" * 80 + "\n")

            f.write("\nAGE DISTRIBUTION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(age_analysis + "\n")

            f.write("\nEXPERTISE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(expertise_analysis + "\n")

            f.write("\nPRACTICE LOCATION ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(location_analysis + "\n")

            f.write("\nRACE AND SEX ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(race_sex_analysis + "\n")

            f.write("\nYEARS OF PRACTICE ANALYSIS\n")
            f.write("-" * 80 + "\n")
            f.write(years_analysis + "\n")

            f.write("\nOVERALL ASSESSMENT\n")
            f.write("-" * 80 + "\n")
            f.write(overall_analysis + "\n")

        print(f"\n\nResults saved to: {output_file}")

        # Also save raw summary as JSON
        json_output = "demographics_summary.json"
        with open(json_output, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary statistics saved to: {json_output}")

        print(f"\n{'=' * 80}")
        print("ANALYSIS COMPLETE!")
        print(f"{'=' * 80}")
        print(f"\nGenerated files:")
        print(f"  1. {output_file} - Full text analysis with LLM insights")
        print(f"  2. {json_output} - Raw statistics in JSON format")
        print(f"  3. demographics_plots/ - Directory with all visualizations:")
        print(f"     - age_distribution.png")
        print(f"     - expertise_distribution.png")
        print(f"     - race_distribution.png")
        print(f"     - sex_distribution.png")
        print(f"     - years_of_practice_distribution.png")

    except Exception as e:
        print(f"\nError during LLM analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
