import os
import time
import csv
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = "AIzaSyCqrDfPpsiGb_6sRdDqULUKHLsXlA5aYVw"

prompt = """
You are an expert ML research evaluator. Your task is to read a student's machine learning project report and assign a precise numerical score across four dimensions. 

## HOW TO SCORE PRECISELY

For each dimension, use the "Anchor Checklist" method:
1. Read all anchor descriptors for that dimension.
2. Count which specific criteria the report satisfies.
3. Use the Point Mapping Table to convert satisfied criteria → exact score.
4. If a report is between two levels, use the +/- adjustment rules.

Do NOT round to "safe" numbers like 15, 20, 25. Be precise.

---

## DIMENSION 1: Technical Quality (0–40 points)

### Anchor Checklist — check each box the report satisfies:

**Baseline & Comparison**
- [ ] T1: A simple baseline model is included (e.g., majority class, logistic regression, rule-based)
- [ ] T2: The baseline is compared numerically against the main model

**Data Handling**
- [ ] T3: Train/Validation/Test splits are clearly defined with no leakage
- [ ] T4: Data preprocessing steps are described and justified

**Model Design**
- [ ] T5: Architecture choices are explicitly justified (not just stated)
- [ ] T6: Hyperparameters (learning rate, batch size, epochs, etc.) are reported
- [ ] T7: At least one hyperparameter is justified with reasoning or a search

**Evaluation**
- [ ] T8: Metrics chosen are appropriate for the task (e.g., F1 for imbalanced data, BLEU for translation)
- [ ] T9: Multiple metrics are reported (not just accuracy alone)

**Error Analysis**
- [ ] T10: A confusion matrix OR failure case analysis is included
- [ ] T11: The report investigates *why* the model fails on specific examples (qualitative or quantitative)
- [ ] T12: Visualizations like saliency maps, attention weights, or feature importance are present

### Point Mapping:
| Criteria Satisfied | Score Range | How to pick within range |
|--------------------|-------------|--------------------------|
| 10–12 criteria     | 35–40       | 35 base + (extra depth beyond checklist) × up to 5 |
| 7–9 criteria       | 25–34       | 25 base + 1.5 pts per extra criterion |
| 3–6 criteria       | 10–24       | 10 base + 2 pts per criterion |
| 0–2 criteria       | 0–9         | 0 base + 3 pts per criterion |

**Fine-grained tie-breaking (e.g., 33 vs 35):**
- Award 35+ ONLY if T11 and T12 are both satisfied (deep error analysis present)
- Deduct 2 pts if results cannot be reproduced (no seeds, no code, no dataset info)
- Deduct 1 pt for each metric that is clearly wrong for the task type

---

## DIMENSION 2: Novelty (0–20 points)

### Anchor Checklist:

**Dataset**
- [ ] N1: Dataset is non-standard (not MNIST, CIFAR-10/100, IMDB, Titanic, Iris, or other well-known "toy" datasets)
- [ ] N2: Dataset was collected or curated by the student themselves

**Task**
- [ ] N3: The task itself is non-standard or problem framing is original
- [ ] N4: The student identified a gap in prior work and designed the task to fill it

**Architecture / Algorithm**
- [ ] N5: A standard architecture (BERT, YOLO, ResNet, etc.) is used with meaningful modifications
- [ ] N6: A hybrid or entirely new architecture is proposed
- [ ] N7: A new loss function, training objective, or optimization strategy is introduced

**Insights**
- [ ] N8: The student draws a novel conclusion not found in existing papers

### Point Mapping:
| Criteria Satisfied | Score |
|--------------------|-------|
| N6 or N7 satisfied (true architectural novelty) | 18–20 |
| N1 + N3 + N5 (adapted model, novel task/data) | 13–17 |
| N1 alone OR N5 alone (new data OR minor modification) | 10–12 |
| None, or only standard toy datasets with no modifications | 0–9 |

**Fine-grained tie-breaking (e.g., 16 vs 18):**
- Score 18–20 ONLY if the student can articulate *why* their architectural change helps theoretically or empirically
- Score 19–20 requires N8 (a novel insight or finding not in prior work)
- Score 13 vs 17: closer to 17 if N2 (self-collected data) is satisfied; closer to 13 if data is just an obscure Kaggle dataset

---

## DIMENSION 3: Significance / Impact (0–20 points)

### Anchor Checklist:

**Problem Value**
- [ ] S1: The problem has a clear real-world application (not purely academic)
- [ ] S2: The problem affects a significant population or system (not niche hobby use)
- [ ] S3: Solving this problem has measurable consequences (cost, health, safety, access)

**Feasibility & Scope**
- [ ] S4: The project scope is realistic for a student project (not "solve all of NLP")
- [ ] S5: The student demonstrates awareness of deployment or real-world constraints

**Domain**
- [ ] S6: The domain is high-stakes (healthcare, climate, accessibility, safety, education)
- [ ] S7: The student references domain experts or field-specific literature

### Point Mapping:
| Criteria Satisfied | Score |
|--------------------|-------|
| S1 + S3 + S6 satisfied | 18–20 |
| S1 + S2 satisfied, but not S6 | 12–17 |
| S1 satisfied alone (vague "useful") | 8–11 |
| No real-world application identified | 0–7 |

**Fine-grained tie-breaking (e.g., 17 vs 19):**
- Score 19–20: requires S3 + S6 + S7 all satisfied (high-stakes domain with domain-grounded justification)
- Score 18: S3 + S6 satisfied but no domain citations
- Score 15–17: S1 + S2 satisfied (clear and meaningful but not critical)
- Deduct 3 pts if scope is wildly unrealistic (e.g., "this will replace radiologists globally")

---

## DIMENSION 4: Clarity & Presentation (0–20 points)

### Anchor Checklist:

**Structure**
- [ ] C1: The report has all standard sections: Abstract, Introduction, Related Work, Methodology, Results, Conclusion
- [ ] C2: The abstract explicitly states the problem, method, and key result (all three)
- [ ] C3: A Related Work section exists with ≥5 cited academic papers

**Figures & Tables**
- [ ] C4: All figures have axis labels and titles
- [ ] C5: All figures are referenced in the body text (not floating orphans)
- [ ] C6: Tables are formatted cleanly (no raw CSV dumps or code output screenshots)

**Writing**
- [ ] C7: Writing is clear enough that a non-expert could understand the motivation
- [ ] C8: Technical terms are defined when first introduced
- [ ] C9: No large raw code blocks embedded in the narrative (code is in an appendix or repo link)

### Point Mapping:
| Criteria Satisfied | Score |
|--------------------|-------|
| 8–9 criteria | 18–20 |
| 5–7 criteria | 12–17 |
| 3–4 criteria | 8–11 |
| 0–2 criteria | 0–7 |

**Fine-grained tie-breaking (e.g., 17 vs 19):**
- Score 19–20: C2 + C3 + C4 + C5 ALL satisfied (publication-ready figures and abstract)
- Score 18: C3 or C4 missing but everything else strong
- Score 15 vs 17: closer to 17 if writing is fluent; closer to 15 if writing is mechanical but correct
- Deduct 2 pts if citations are informal (e.g., "according to a Medium article" or Wikipedia)

---

## OUTPUT FORMAT

Provide your evaluation in this exact structure:

**DIMENSION 1 — Technical Quality: [X]/40**
Criteria satisfied: [list the T-codes, e.g., T1, T3, T4, T7]
Criteria missing: [list the T-codes]
Justification: [2–3 sentences explaining the score, referencing specific evidence from the report]

**DIMENSION 2 — Novelty: [X]/20**
Criteria satisfied: [list N-codes]
Criteria missing: [list N-codes]
Justification: [2–3 sentences]

**DIMENSION 3 — Significance: [X]/20**
Criteria satisfied: [list S-codes]
Criteria missing: [list S-codes]
Justification: [2–3 sentences]

**DIMENSION 4 — Clarity & Presentation: [X]/20**
Criteria satisfied: [list C-codes]
Criteria missing: [list C-codes]
Justification: [2–3 sentences]

**TOTAL SCORE: [X]/100**

**Overall Summary:** [3–4 sentences summarizing the report's greatest strength and most critical weakness]
"""

# Define the structured output schema for the API
class EvaluationResult(BaseModel):
    technical_score: int = Field(description="Score out of 40")
    novelty_score: int = Field(description="Score out of 20")
    significance_score: int = Field(description="Score out of 20")
    clarity_score: int = Field(description="Score out of 20")
    total_score: int = Field(description="Sum of all scores out of 100")

def evaluate_projects():
    # Initialize the client. It automatically picks up the GEMINI_API_KEY environment variable.
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Map the ground truth labels to their respective folder names
    folders = {"Outstanding": "Outstanding", "Normal": "Normal"}
    output_csv = "phase2_baseline_scores_gemini_3_flash.csv"
    
    # Read already processed files to avoid duplicates
    processed_files = set()
    file_exists = os.path.isfile(output_csv)
    
    if file_exists:
        with open(output_csv, mode='r', encoding='utf-8') as file:
            reader = csv.reader(file)
            # Skip header row
            next(reader, None)
            for row in reader:
                if row: # Check if row is not empty
                    processed_files.add(row[0]) # filename is the first column
                    
    print(f"Found {len(processed_files)} previously processed files. Resuming...")
    
    with open(output_csv, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # Write headers only if the file is new
        if not file_exists:
            writer.writerow([
                "filename", "technical_score", "novelty_score", 
                "significance_score", "clarity_score", "total_score", "ground_truth"
            ])
            
        for ground_truth, folder_path in folders.items():
            if not os.path.exists(folder_path):
                print(f"Directory not found: {folder_path}")
                continue
                
            for filename in os.listdir(folder_path):
                if not filename.lower().endswith('.pdf'):
                    continue
                    
                # Check if the file was already processed
                if filename in processed_files:
                    print(f"Skipping {filename} (Already processed)")
                    continue
                    
                filepath = os.path.join(folder_path, filename)
                print(f"\nProcessing: {filename} (Category: {ground_truth})")
                
                max_retries = 5
                retry_delay = 15
                
                for attempt in range(max_retries):
                    uploaded_file = None
                    try:
                        # Upload the file to the Gemini File API
                        uploaded_file = client.files.upload(file=filepath)
                        
                        # Wait a brief moment to ensure the file is processed on Google's end
                        time.sleep(2)
                                                
                        # Execute the model call with structured output
                        response = client.models.generate_content(
                            model='gemini-3-flash-preview',
                            contents=[uploaded_file, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=EvaluationResult,
                                temperature=0.1, 
                            )
                        )
                        
                        # Parse the returned JSON
                        result = json.loads(response.text)
                        
                        # Write the row to the CSV
                        writer.writerow([
                            filename,
                            result.get("technical_score"),
                            result.get("novelty_score"),
                            result.get("significance_score"),
                            result.get("clarity_score"),
                            result.get("total_score"),
                            ground_truth
                        ])
                        
                        # Flush the file buffer to ensure data is saved immediately in case of a crash
                        file.flush() 
                        
                        print(f"Success! {filename} scored {result.get('total_score')}/100")
                        
                        # Delete the file from the API storage to clear quota
                        client.files.delete(name=uploaded_file.name)
                        
                        # Standard delay to pace requests
                        time.sleep(3)
                        break 
                        
                    except Exception as e:
                        error_msg = str(e)
                        print(f"Attempt {attempt + 1} failed: {error_msg}")
                        
                        # Clean up the file if it was uploaded but the generation failed
                        if uploaded_file:
                            try:
                                client.files.delete(name=uploaded_file.name)
                            except:
                                pass
                                
                        # Handle rate limits with exponential backoff
                        if "429" in error_msg or "ResourceExhausted" in error_msg:
                            print(f"Rate limit encountered. Sleeping for {retry_delay} seconds...")
                            time.sleep(retry_delay)
                            retry_delay *= 2 
                        else:
                            print("Non rate limit error encountered. Skipping to next file.")
                            break 

# Run the execution
evaluate_projects()
print("\nBatch evaluation complete. Check phase2_baseline_scores.csv for results.")