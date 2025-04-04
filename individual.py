#!/usr/bin/env python3
import json
import pandas as pd

def main():
    # 1. Load JSON data from file (adjust the filename/path to your actual data file)
    with open("analysis_results/all_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 2. Create a list to store per-repo records for the individual dimension metrics
    records = []
    
    for repo in data:
        # --- Identify the country from metadata ---
        country = repo.get("metadata", {}).get("country", "Unknown")
        
        # --- Extract metrics for the individual dimension ---
        # Example: from commit_history, repo_structure, etc.
        
        commit_history = repo.get("commit_history", {})
        total_commits = commit_history.get("total_commits", 0)
        commit_frequency_per_month = commit_history.get("commit_frequency", 0.0)
        
        # Suppose 'repo_structure' includes booleans for presence of readme, license, etc.
        repo_structure = repo.get("repo_structure", {})
        has_readme = 1 if repo_structure.get("has_readme", False) else 0
        has_contribution_guide = 1 if repo_structure.get("has_contribution_guide", False) else 0
        has_code_of_conduct = 1 if repo_structure.get("has_code_of_conduct", False) else 0
        
        # You can similarly extract other fields like open_issues, merged_pr_percentage, etc.
        # For demonstration, we'll keep it simple:
        
        records.append({
            "country": country,
            "total_commits": total_commits,
            "commit_frequency_per_month": commit_frequency_per_month,
            "has_readme": has_readme,
            "has_contribution_guide": has_contribution_guide,
            "has_code_of_conduct": has_code_of_conduct,
        })
    
    # 3. Convert the records to a Pandas DataFrame
    df = pd.DataFrame(records)
    
    # 4. Group by country and compute the average (or other aggregations) of numeric columns
    # Adjust the columns to match your desired metrics
    numeric_columns = [
        "total_commits",
        "commit_frequency_per_month",
        "has_readme",
        "has_contribution_guide",
        "has_code_of_conduct"
    ]
    
    grouped = df.groupby("country")[numeric_columns].mean().reset_index()
    
    # 5. Print the resulting table
    print("=== Aggregated Individual Dimension Metrics by Country ===")
    print(grouped.to_string(index=False))
    
    # 6. Optionally, save to CSV
    output_csv = "individual_dimension_by_country.csv"
    grouped.to_csv(output_csv, index=False)
    print(f"\nResults saved to '{output_csv}'")

if __name__ == "__main__":
    main()
