#!/bin/bash
# Run all USMLE Sample annotation analyses

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="../outputs"
mkdir -p "$OUTPUT_DIR"

echo "=================================="
echo "Running All Annotation Analyses"
echo "USMLE Sample Dataset"
echo "=================================="
echo ""

# Run main analysis
echo "1. Comprehensive Results Analysis"
echo "----------------------------------"
python analyze_annotations.py --mode stats > "$OUTPUT_DIR/analysis_report.txt"
echo "✓ Results saved to: $OUTPUT_DIR/analysis_report.txt"
echo ""

# Compare models
echo "2. Model Comparison"
echo "----------------------------------"
python analyze_annotations.py --mode compare > "$OUTPUT_DIR/model_comparison.txt"
echo "✓ Results saved to: $OUTPUT_DIR/model_comparison.txt"
echo ""

# Export to CSV
echo "3. Exporting to CSV"
echo "----------------------------------"
python analyze_annotations.py --mode export --output "$OUTPUT_DIR/usmle_sample_annotations.csv"
echo ""

# Collect persuasion examples
echo "4. Collecting Persuasion Examples"
echo "----------------------------------"
python collect_persuasion_examples.py
echo "✓ Results saved to: $OUTPUT_DIR/persuasion_examples.{txt,json}"
echo ""

# Analyze by case type (Bayesian vs Behavioral decisions)
echo "5. Analysis by Case Type"
echo "----------------------------------"
python coverage_report.py --mode by-case-type > "$OUTPUT_DIR/case_type_analysis.txt"
echo "✓ Results saved to: $OUTPUT_DIR/case_type_analysis.txt"
echo ""

# Case annotation counts
echo "6. Case Annotation Counts"
echo "----------------------------------"
python coverage_report.py --mode case-counts > "$OUTPUT_DIR/case_annotation_counts.txt"
echo "✓ Results saved to: $OUTPUT_DIR/case_annotation_counts.txt"
echo ""

# Full case annotation report (including unannotated cases)
echo "7. Full Case Annotation Report"
echo "----------------------------------"
python coverage_report.py --mode full-report > "$OUTPUT_DIR/full_case_annotation_report.txt"
echo "✓ Results saved to: $OUTPUT_DIR/full_case_annotation_report.txt"
echo ""

# Model accuracy analysis
echo "8. Model Accuracy Analysis"
echo "----------------------------------"
python analyze_model_accuracy.py > "$OUTPUT_DIR/model_accuracy_report.txt"
echo "✓ Results saved to: $OUTPUT_DIR/model_accuracy_report.txt"
echo ""

echo "=================================="
echo "All analyses complete!"
echo "=================================="
echo ""
echo "Generated files (in $OUTPUT_DIR/):"
echo "  - analysis_report.txt              (Comprehensive analysis)"
echo "  - model_comparison.txt             (Model-by-model comparison)"
echo "  - usmle_sample_annotations.csv     (Raw data export)"
echo "  - case_type_analysis.txt           (Analysis by Bayesian/Behavioral decisions)"
echo "  - case_annotation_counts.txt       (Annotation counts per case)"
echo "  - full_case_annotation_report.txt  (Complete annotation status report)"
echo "  - model_accuracy_report.txt        (Model accuracy on test questions)"
echo ""
