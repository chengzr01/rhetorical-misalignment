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

echo "=================================="
echo "All analyses complete!"
echo "=================================="
echo ""
echo "Generated files:"
echo "  - analysis_report.txt      (Comprehensive analysis)"
echo "  - model_comparison.txt     (Model-by-model comparison)"
echo "  - usmle_sample_annotations.csv  (Raw data export)"
echo ""
echo "See SUMMARY.md for key findings!"
echo ""
