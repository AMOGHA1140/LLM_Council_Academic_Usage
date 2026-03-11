import os
import time
import csv
import json
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

GEMINI_API_KEY = "<API-KEY>"

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
    output_csv = "phase2_baseline_scores.csv"
    
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
                        
                        prompt = "Read this project report. Provide a score for Technical Quality (out of 40), Novelty (out of 20), Significance (out of 20), and Clarity (out of 20). Return the individual scores and the total score."
                        
                        # Execute the model call with structured output
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
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