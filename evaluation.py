import sys
import pandas as pd
from sklearn.metrics import roc_auc_score

def evaluate_predictions(csv_path):
    # Load the dataset
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Could not find the file at {csv_path}")
        sys.exit(1)

    # Ensure required columns exist
    required_columns = ['filename', 'total_score', 'ground_truth']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: CSV must contain the following columns: {required_columns}")
        sys.exit(1)

    print(f"--- Evaluation Results for {csv_path} ---\n")

    # 1. Group Distribution Statistics
    print("1. Score Distributions:")
    stats = df.groupby('ground_truth')['total_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats.to_string())
    print("\n")

    # 2. Precision at K
    # Count how many actual 'Outstanding' papers exist in the dataset
    k_value = len(df[df['ground_truth'] == 'Outstanding'])
    
    if k_value == 0:
        print("Error: No 'Outstanding' papers found in the ground_truth column.")
        sys.exit(1)

    # Sort the dataframe by the AI's total score in descending order
    df_sorted = df.sort_values(by='total_score', ascending=False)
    
    # Take the top K predictions
    top_k_predictions = df_sorted.head(k_value)
    
    # Count how many of those top K are actually 'Outstanding'
    correct_in_top_k = len(top_k_predictions[top_k_predictions['ground_truth'] == 'Outstanding'])
    precision_at_k = (correct_in_top_k / k_value) * 100

    print(f"2. Rank-Based Evaluation (Precision at {k_value}):")
    print(f"Out of the {k_value} actual Outstanding papers, the AI placed {correct_in_top_k} in its Top {k_value}.")
    print(f"Accuracy Metric: {precision_at_k:.2f}%\n")

    # 3. ROC-AUC Score
    # Convert string labels to binary (Outstanding = 1, Normal = 0)
    df['binary_label'] = df['ground_truth'].apply(lambda x: 1 if x == 'Outstanding' else 0)
    
    # Calculate AUC
    try:
        auc_score = roc_auc_score(df['binary_label'], df['total_score'])
        print("3. Threshold-Independent Probability (ROC-AUC):")
        print(f"AUC Score: {auc_score:.4f}")
        print("(0.5 = Random Guessing, 1.0 = Perfect Separation)")
    except ValueError:
        print("3. ROC-AUC:")
        print("Could not calculate AUC. Ensure you have both Outstanding and Normal classes in the dataset.")

if __name__ == "__main__":
    # Ensure the user provided a file path argument
    if len(sys.argv) != 2:
        print("Usage: python evaluate_baseline.py <path_to_csv_file>")
        sys.exit(1)
        
    csv_file_path = sys.argv[1]
    evaluate_predictions(csv_file_path)