#!/bin/bash
# Run all USMLE Sample annotation analyses

echo "=================================="
echo "Running All Annotation Analyses"
echo "USMLE Sample Dataset"
echo "=================================="
echo ""

# Run main analysis
echo "1. Comprehensive Results Analysis"
echo "----------------------------------"
python analyze_results.py > analysis_report.txt
echo "✓ Results saved to: analysis_report.txt"
echo ""

# Compare models
echo "2. Model Comparison"
echo "----------------------------------"
python compare_models.py > model_comparison.txt
echo "✓ Results saved to: model_comparison.txt"
echo ""

# Export to CSV
echo "3. Exporting to CSV"
echo "----------------------------------"
python export_to_csv.py
echo ""

# Collect persuasion examples
echo "4. Collecting Persuasion Examples"
echo "----------------------------------"
python collect_persuasion_examples.py
echo ""

# Analyze by case type (Bayesian vs Behavioral decisions)
echo "5. Analysis by Case Type"
echo "----------------------------------"
python analyze_by_case_type.py > case_type_analysis.txt
echo "✓ Results saved to: case_type_analysis.txt"
echo ""

# Case annotation counts
echo "6. Case Annotation Counts"
echo "----------------------------------"
python case_annotation_counts.py > case_annotation_counts.txt
echo "✓ Results saved to: case_annotation_counts.txt"
echo ""

# Full case annotation report (including unannotated cases)
echo "7. Full Case Annotation Report"
echo "----------------------------------"
python full_case_annotation_report.py > full_case_annotation_report.txt
echo "✓ Results saved to: full_case_annotation_report.txt"
echo ""

# Model accuracy analysis
echo "8. Model Accuracy Analysis"
echo "----------------------------------"
python analyze_model_accuracy.py > model_accuracy_report.txt
echo "✓ Results saved to: model_accuracy_report.txt"
echo ""

echo "=================================="
echo "All analyses complete!"
echo "=================================="
echo ""
echo "Generated files:"
echo "  - analysis_report.txt              (Comprehensive analysis)"
echo "  - model_comparison.txt             (Model-by-model comparison with belief change patterns)"
echo "  - usmle_sample_annotations.csv     (Raw data export)"
echo "  - persuasion_examples.txt          (Detailed persuasion cases with highlights)"
echo "  - persuasion_examples.json         (Structured persuasion data)"
echo "  - case_type_analysis.txt           (Analysis by Bayesian/Behavioral decisions)"
echo "  - case_annotation_counts.txt       (Annotation counts per case)"
echo "  - full_case_annotation_report.txt  (Complete annotation status report)"
echo "  - model_accuracy_report.txt        (Model accuracy on test questions)"
echo ""
echo "See SUMMARY.md for key findings!"
echo ""
