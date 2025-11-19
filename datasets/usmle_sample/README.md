# README: USMLE Sample Questions

- Step_1_Sample_Items.pdf  
- Step2_CK_Sample_Questions.pdf  
- Step3_Sample_Items.pdf  

The parsed JSON files are:

- `Step1_questions_parsed.json`
- `Step2_CK_questions_parsed.json`
- `Step3_questions_parsed.json`

---

## 📌 Dataset Structure

Each question is stored using the following schema:

```json
{
  "exam": "Step 2 CK",
  "number": 1,
  "stem": "Full question text…",
  "options": [
    { "label": "A", "text": "Option text…" },
    { "label": "B", "text": "Option text…" }
  ],
  "answer": "A"
}
```

### Fields

| Field      | Description |
|------------|-------------|
| `exam`     | Which exam the question came from (`Step 1`, `Step 2 CK`, `Step 3`). |
| `number`   | Question number in the original booklet. |
| `stem`     | The vignette / main question text. |
| `options`  | List of answer choices with labels and text. |
| `answer`   | Correct answer (from official answer key). |

---

## ⚠️ Caveats & Limitations

### 1. **PDF → Text extraction is imperfect**
The PDFs contain:
- table formatting  
- multi-column layout  
- irregular hyphenation  
- page headers/footers  
- graphic/pictorial references  
- broken lines and wrapped sentences  

These issues lead to:
- Occasional malformed question stems  
- Missing spaces or broken words (e.g., “pa tient”)  
- Extraneous line breaks  
- Incomplete text in rare cases  

### 2. **Not all questions successfully parsed**
Some questions, especially those with:
- complex formatting  
- embedded tables/figures  
- multi-step sequential formats  

…may be missing or partially parsed.

### 3. **Answer keys required manual anchor detection**
The answer key sections in the PDFs are not uniformly formatted.  
To extract them, the script searched for anchor phrases and patterns.  
Therefore:
- Some answers might not map cleanly to questions  
- Step 1 and Step 3 answers required extra heuristics  
- There may be mismatches in rare cases

### 4. **Whitespace normalization can alter text**
All stems and options are whitespace-collapsed:  
- Consecutive spaces → single space  
- Broken lines merged  
- This improves readability but sacrifices exact fidelity to the original formatting.

### 5. **Sequential question sets**
Some exams (e.g., Step 3) include multi-part sequential question sets sharing a common vignette.  
In the JSON:
- Each question is **independent**  
- Shared vignette text is embedded repeatedly  
- No explicit linkage between sequential questions

### 6. **No images or charts are included**
Any graphical components referenced in the original PDFs are not available.  
This may cause reduced clarity for questions dependent on visuals.

---

## 📁 File Location

Your dataset files are stored in:

```
/mnt/data/
 ├── Step1_questions_parsed.json
 ├── Step2_CK_questions_parsed.json
 ├── Step3_questions_parsed.json
 └── README.md
```

---

## ✔️ Suggested Next Steps

Depending on your research goals, you may want to:

- **Clean / standardize text further**
- **Split into train/dev/test sets**
- **Tag questions by specialty or organ system**
- **Link sequential items properly**
- **Perform manual review of borderline cases**

If you want, I can automate any of these tasks.

---

## 📬 Questions?

Feel free to ask for:
- restructuring the JSON,
- adding metadata,
- generating code to load the dataset,
- or converting to CSV, parquet, or SQL.

