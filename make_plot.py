#!/usr/bin/env python3
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import glob

def process_codecarbon_file(file_path):
    """
    Process a CodeCarbon CSV file to extract relevant metrics.
    
    This function does NOT assume the CSV has a 'country_name' column.
    Instead, we parse the country from the directory structure:
      codecarbon_results/codecarbon/<COUNTRY>/<REPO_NAME>/emissions.csv
    Then we compute total power as the sum of cpu_power, gpu_power, and ram_power.
    
    Returns a DataFrame with columns: ['country', 'total_power', 'emissions'].
    """
    # Example path: codecarbon_results/codecarbon/France/youckan/emissions.csv
    # We want to extract "France" as the country.
    path_obj = Path(file_path)
    # path_obj.parent is the repository folder; parent.parent is the country folder.
    country = path_obj.parent.parent.name
    
    df = pd.read_csv(file_path)
    
    # Check for required columns (adjust if your CSV differs)
    required_cols = ['cpu_power', 'gpu_power', 'ram_power', 'emissions']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in {file_path}")
    
    # Calculate total power consumption (Watts) = CPU + GPU + RAM power
    df['total_power'] = df['cpu_power'] + df['gpu_power'] + df['ram_power']
    
    # Add the extracted country column
    df['country'] = country
    
    return df[['country', 'total_power', 'emissions']]

def main():
    # Update the base directory to where your CodeCarbon CSV files are located.
    BASE_DIR = os.path.join("codecarbon_results", "codecarbon")
    
    # Look for 'emissions.csv' files recursively in the BASE_DIR.
    codecarbon_files = glob.glob(os.path.join(BASE_DIR, "**", "emissions.csv"), recursive=True)
    
    if not codecarbon_files:
        print("No CodeCarbon CSV files found under", BASE_DIR)
        return
    
    # Process each file and collect the data.
    df_list = []
    for file in codecarbon_files:
        try:
            df = process_codecarbon_file(file)
            df_list.append(df)
        except Exception as e:
            print(f"Error processing {file}: {e}")
    
    if not df_list:
        print("No valid CodeCarbon data found.")
        return
    
    # Concatenate all data into a single DataFrame.
    all_data = pd.concat(df_list, ignore_index=True)
    
    # Group by country and compute average total power and average emissions.
    grouped = all_data.groupby('country').agg({
        'total_power': 'mean',
        'emissions': 'mean'
    }).reset_index()
    
    # Plotting: Create a dual-axis bar chart.
    countries = grouped['country']
    x = np.arange(len(countries))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    # Plot average total power on the primary y-axis.
    bars1 = ax1.bar(x - width/2, grouped['total_power'], width, label="Avg Total Power (Watts)", color='#56b3fa')
    
    # Plot average emissions on the secondary y-axis.
    bars2 = ax2.bar(x + width/2, grouped['emissions'], width, label="Avg Emissions (kg CO2)", color='#ffb000')
    
    ax1.set_xlabel("Country")
    ax1.set_ylabel("Avg Total Power (Watts)", color='blue')
    ax2.set_ylabel("Avg Emissions (kg CO2)", color='orange')
    ax1.set_xticks(x)
    ax1.set_xticklabels(countries, rotation=45, ha='right')
    
    # Add legends.
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    
    plt.title("Average Total Power and Emissions per Country (CodeCarbon)")
    plt.tight_layout()
    
    # Save and show the plot.
    plt.savefig("codecarbon_avg_power_emissions.png")
    print("Plot saved as 'codecarbon_avg_power_emissions.png'")
    plt.show()

if __name__ == "__main__":
    main()

