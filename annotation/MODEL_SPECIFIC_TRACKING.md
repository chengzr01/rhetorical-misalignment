# Model-Specific Annotation Tracking

## Summary of Changes

The annotation system has been updated to track annotations **separately for each model**. This means:

- **Llama-3.1-8B** (llama_small): 108 manipulative cases → needs 324 annotations
- **Llama-3.3-70B** (llama): 122 manipulative cases → needs 366 annotations
- **Llama-3.1-405B** (llama_large): 127 manipulative cases → needs 381 annotations
- **DeepSeek-V3** (deepseek): 190 manipulative cases → needs 570 annotations

Each model's cases are tracked independently, so when a participant selects a model, they only see and annotate cases for that specific model.

## What Changed

### 1. Removed Coverage Statistics from UI
**Before**: The demographics page showed a progress box with coverage statistics.

**After**: The progress box has been removed. Participants only see:
- Brief description of smart case assignment
- Priority levels (0 → 1 → 2 → 3+ annotations)
- No visibility into current progress

**Rationale**: Participants don't need to see this information; it's handled automatically in the background.

### 2. Model-Specific Annotation Tracking
**Before**: Annotations were counted across all models in a dataset. If someone annotated case X for llama_small, it would affect case X for all models.

**After**: Each model tracks its own annotations separately:
- Annotating case X for **llama_small** only affects llama_small's counters
- Case X for **llama** (70B) is still unannotated and can be assigned
- Each model maintains independent progress toward the 3-annotation goal

### 3. Backend Changes

#### New field in annotations: `model_key`
All annotations now include:
```json
{
  "annotator_id": "...",
  "dataset": "usmle_sample",
  "model_key": "llama_small",    ← NEW FIELD
  "case_id": "usmle_sample_42",
  ...
}
```

#### Updated function: `get_annotation_counts_per_case()`
```python
# Before
def get_annotation_counts_per_case(dataset_key):
    # Counted ALL annotations in dataset

# After
def get_annotation_counts_per_case(dataset_key, model_key=None):
    # Can filter by specific model
    # Only counts annotations for that model
```

#### Updated smart assignment
```python
# Now passes model_key to ensure model-specific tracking
annotation_counts = get_annotation_counts_per_case(dataset_key, model_key)
```

## How It Works Now

### Participant Flow

1. **Participant A** starts annotation:
   - Selects: USMLE Sample, Llama-3.1-8B
   - System loads: 108 manipulative cases for llama_small
   - System checks: How many of these 108 cases have been annotated for llama_small?
   - System assigns: 10 cases (prioritizing unannotated ones)

2. **Participant B** starts annotation:
   - Selects: USMLE Sample, Llama-3.3-70B
   - System loads: 122 manipulative cases for llama (different from above!)
   - System checks: How many of these 122 cases have been annotated for llama?
   - System assigns: 10 cases (independent from Participant A's work)

### Example Scenario

**Case ID**: `usmle_sample_42`

| Model | Annotations | Status |
|-------|-------------|--------|
| llama_small | 3 | ✅ Goal reached |
| llama | 1 | ⚠️ Need 2 more |
| llama_large | 0 | 🔴 Unannotated |
| deepseek | 2 | ⚠️ Need 1 more |

Even though it's the same case ID, **each model tracks it separately**. This is important because:
- Different models make different predictions
- We want 3 human annotations per model to judge each model's performance
- Models have different manipulative case sets

## Technical Details

### Files Modified

1. **app.py**
   - Line 207-249: Updated `get_annotation_counts_per_case()` to filter by model
   - Line 476: Pass `model_key` when getting annotation counts
   - Line 405: API endpoint uses model-specific counting
   - Line 866, 927: Store `model_key` in new annotations

2. **analyze_coverage.py**
   - Line 62-104: Updated function signature to accept `model_key`
   - Line 122: Pass `model_key` when analyzing coverage

3. **templates/demographics.html**
   - Removed coverage statistics display (lines 148-191 deleted)
   - Removed JavaScript for fetching stats (updateCoverageStats function)
   - Simplified UI to just show assignment description

### Backward Compatibility

**Old annotations** (created before this update):
- Do NOT have `model_key` field
- Will NOT be counted by model-specific tracking
- This is fine - only 1-2 old annotations exist
- All new annotations will have `model_key` and work correctly

**If you need to migrate old annotations**, you can manually add `model_key` based on the `agent_model` field.

## Testing Results

All tests passed:

```bash
✓ Flask app loads correctly
✓ Coverage API returns model-specific data
✓ Smart assignment selects different cases for different models
✓ Analysis script shows per-model breakdown
✓ No cross-contamination between models
```

### Test Output

```
Model: llama_small
  Total manipulative cases: 108
  Current annotations: 0
  Selected 10 cases: [unique to llama_small]

Model: llama
  Total manipulative cases: 122
  Current annotations: 0
  Selected 10 cases: [unique to llama, different from above]
```

## Benefits

1. **Independent Progress Tracking**: Each model reaches the 3-annotation goal independently
2. **Fair Distribution**: Cases for llama_small don't compete with cases for deepseek
3. **Accurate Metrics**: Know exactly which models need more annotations
4. **Scalable**: Easy to add new models without affecting existing tracking
5. **Cleaner Data**: No confusion about which model an annotation belongs to

## Usage

### For Participants
**No change in workflow** - just select your model and start annotating as before. The system handles everything automatically.

### For Administrators

**Check progress per model**:
```bash
python analyze_coverage.py
```

Output shows each model separately:
```
================================================================================
Dataset: USMLE Sample, Model: llama_small
================================================================================
  Total manipulative cases: 108
  Annotation Coverage:
    ✗ Not annotated yet:          108 cases (100.0%)
    ...
  Progress: 0/324 (0.0%)
```

**API for real-time stats**:
```bash
curl http://localhost:8000/api/coverage/usmle_sample/llama_small
```

Returns:
```json
{
  "total_cases": 108,
  "cases_by_count": {"0": 108, "1": 0, "2": 0, "3": 0},
  "total_annotations": 0,
  "target_annotations": 324,
  "progress_percent": 0.0
}
```

## Future Work (Optional)

If needed, you can:
- Add a separate admin dashboard to view all models at once
- Export per-model statistics to CSV
- Set up alerts when a model reaches completion milestones
- Migrate old annotations by adding `model_key` based on `agent_model`

But the current system is complete and working as designed!
