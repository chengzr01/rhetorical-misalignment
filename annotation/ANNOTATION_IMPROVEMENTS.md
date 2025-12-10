# Annotation Interface Improvements

## Summary

The annotation interface now includes **smart case assignment** with real-time coverage statistics. The system intelligently assigns 10 cases to each participant, prioritizing cases that need annotations while ensuring every case gets at least 3 annotations.

## Key Features

### 1. Smart Case Assignment Algorithm

When a participant starts a new annotation session, the system:

1. **First Priority**: Selects cases with 0 annotations (ensuring full coverage)
2. **Second Priority**: Selects cases with 1 annotation (building toward goal)
3. **Third Priority**: Selects cases with 2 annotations (approaching goal)
4. **Fourth Priority**: Selects cases with 3+ annotations (goal reached)

Within each priority group, cases are **randomly shuffled** to prevent bias.

### 2. Real-Time Coverage Statistics

The demographics page now displays:
- **Overall progress bar**: Visual indicator of annotation completion
- **Breakdown by annotation count**: Shows how many cases have 0, 1, 2, or 3+ annotations
- **Total statistics**: Total cases, current annotations, and target annotations
- **Dynamic updates**: Statistics update when you change dataset or model

### 3. Analysis Tools

#### Coverage Analysis Script
Run `python analyze_coverage.py` to get a detailed report:
```bash
conda activate alignment
python analyze_coverage.py
```

This shows:
- Total manipulative cases per dataset/model
- Annotation coverage breakdown
- Progress toward 3-annotation goal
- Priority cases that need attention

## How It Works

### Smart Assignment in Action

Example scenario:
- **108 total cases** in usmle_sample/llama_small
- **50 cases** have 0 annotations (highest priority)
- **30 cases** have 1 annotation
- **20 cases** have 2 annotations
- **8 cases** have 3+ annotations

When a new participant starts:
1. System randomly selects **10 cases from the 50 unannotated cases**
2. If fewer than 10 unannotated cases exist, it fills from the next priority group
3. Participant annotates these 10 cases
4. Next participant gets a different random selection, again prioritizing unannotated cases

This ensures:
- ✅ Every case gets at least 1 annotation before any case gets a 4th
- ✅ Even distribution across all cases
- ✅ Efficient path to the 3-annotation goal
- ✅ No manual tracking needed

## Technical Details

### New API Endpoint
```
GET /api/coverage/<dataset_key>/<model_key>
```

Returns JSON with:
```json
{
  "total_cases": 108,
  "cases_by_count": {
    "0": 50,
    "1": 30,
    "2": 20,
    "3": 8
  },
  "total_annotations": 138,
  "target_annotations": 324,
  "progress_percent": 42.6
}
```

### Updated Functions in app.py

1. **`get_annotation_counts_per_case(dataset_key)`**
   - Scans annotation files
   - Returns dict of case_id → annotation count

2. **`get_smart_random_cases(case_ids, annotation_counts, num_cases=10)`**
   - Implements priority-based selection
   - Randomly shuffles within priority groups
   - Returns list of selected case IDs

3. **`api_get_coverage(dataset_key, model_key)`**
   - New API endpoint
   - Provides real-time coverage statistics
   - Used by demographics page

## Files Modified

1. **app.py**
   - Added `/api/coverage/<dataset_key>/<model_key>` endpoint (lines 377-416)

2. **templates/demographics.html**
   - Added coverage statistics display (lines 148-191)
   - Added JavaScript for dynamic updates (lines 308-345, 373-377)

3. **analyze_coverage.py** (NEW)
   - Standalone analysis script
   - Shows detailed coverage report
   - Identifies priority cases

## Testing

All tests passed successfully:
```bash
✓ Flask app loads successfully
✓ Coverage API endpoint returns correct data
✓ Smart assignment prioritizes unannotated cases
✓ Demographics page displays statistics correctly
```

## Usage Example

1. **Start the Flask app**:
   ```bash
   conda activate alignment
   python app.py
   ```

2. **Navigate to demographics page**:
   - Select dataset: `USMLE Sample`
   - Select model: `Llama-3.1-8B-Instruct`
   - View real-time coverage statistics

3. **Click "Start Annotation"**:
   - System assigns 10 cases intelligently
   - Prioritizes cases with fewer annotations
   - You'll see your assigned cases in the overview

4. **Check progress anytime**:
   ```bash
   python analyze_coverage.py
   ```

## Goal Tracking

**Overall Goal**: 1,641 total annotations (547 cases × 3 annotations each)

**Current Status** (as of last check):
- Total manipulative cases: 547
- Annotations collected: 2
- Progress: 0.1%
- Annotations remaining: 1,639

The smart assignment system will efficiently guide you toward this goal by:
- Ensuring all cases get annotated at least once first
- Then building toward 2 annotations per case
- Finally reaching the 3-annotation target
- Maintaining random distribution to prevent bias

## Benefits

1. **No Manual Tracking**: System automatically tracks which cases need annotations
2. **Fair Distribution**: Random selection within priority groups prevents bias
3. **Efficient Progress**: Prioritization ensures steady progress toward goal
4. **Real-Time Visibility**: Participants see current status immediately
5. **Scalable**: Works with any number of participants and cases
6. **Flexible**: Easy to adjust target annotation count if needed

## Future Enhancements (Optional)

Potential improvements if needed:
- [ ] Allow configuring number of cases per participant (currently 10)
- [ ] Add participant-specific tracking (prevent re-annotation by same person)
- [ ] Email notifications when datasets reach completion milestones
- [ ] Export coverage reports to CSV/Excel
- [ ] Dashboard for monitoring multiple datasets simultaneously
