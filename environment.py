#!/usr/bin/env python3
import os
import subprocess
import tempfile
import pandas as pd
from codecarbon import EmissionsTracker
from pathlib import Path
import time

# === Configuration ===
OUTPUT_DIR = "codecarbon_results"
TOP_REPOS_CSV = "analysis_results/top_5_repos_per_country.csv"
CLONE_TIMEOUT_SECONDS = 30

def ensure_dependencies():
    # Ensure CodeCarbon is installed
    try:
        from codecarbon import EmissionsTracker
    except ImportError:
        subprocess.run(["pip", "install", "codecarbon"], check=True)
    
    # Create OUTPUT_DIR if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

def clone_repo(repo_url, target_dir):
    try:
        # Use a shallow clone for faster cloning
        subprocess.run(
            ["git", "clone", "--depth", "1", repo_url, target_dir],
            check=True,
            capture_output=True,
            timeout=CLONE_TIMEOUT_SECONDS
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        print(f"Error cloning {repo_url}: {e}")
        return False

def find_python_file(repo_dir):
    """Find a suitable Python file to run."""
    for root, _, files in os.walk(repo_dir):
        for file in files:
            if file.endswith(".py") and not file.startswith("__"):
                full_path = os.path.join(root, file)
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                        if "if __name__ == '__main__':" in content or "def main():" in content:
                            return full_path
                except Exception as e:
                    print(f"Error reading {full_path}: {e}")
    return None

def detect_and_run_repo(repo_dir, repo_url, country):
    """
    Determine how to run the repository.
    Checks (in order) for:
      - A Dockerfile (build image & run container)
      - A docker-compose.yml file (use docker-compose up)
      - A Makefile (run 'make run')
      - Otherwise, attempt to locate a Python entry point.
    Runs the repository under CodeCarbon tracking and returns the emissions (kg)
    and energy consumed (kWh).
    """
    # Extract a clean repo name from the URL.
    repo_name = repo_url.split('/')[-1].replace('.git', '')
    
    # Create a directory for CodeCarbon outputs for this repo.
    codecarbon_dir = os.path.join(OUTPUT_DIR, "codecarbon", country, repo_name)
    os.makedirs(codecarbon_dir, exist_ok=True)
    
    # Initialize CodeCarbon tracker
    tracker = EmissionsTracker(
        project_name=f"{country}_{repo_name}",
        output_dir=codecarbon_dir,
        measure_power_secs=5,
    )
    
    # Determine run strategy
    run_strategy = None
    if os.path.exists(os.path.join(repo_dir, "Dockerfile")):
        run_strategy = "docker"
    elif os.path.exists(os.path.join(repo_dir, "docker-compose.yml")):
        run_strategy = "docker-compose"
    elif os.path.exists(os.path.join(repo_dir, "Makefile")):
        run_strategy = "make"
    else:
        run_strategy = "python"
    
    cmd = None
    if run_strategy == "docker":
        # Build the docker image.
        build_cmd = ["docker", "build", "-t", repo_name.lower(), "."]
        print("Building Docker image with command:", build_cmd)
        build_process = subprocess.run(build_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if build_process.returncode != 0:
            print("Docker build failed:", build_process.stderr.decode())
            return None, None, None
        # Run the container.
        cmd = ["docker", "run", "--rm", repo_name.lower()]
    elif run_strategy == "docker-compose":
        cmd = ["docker-compose", "up", "--build"]
    elif run_strategy == "make":
        cmd = ["make", "run"]
    elif run_strategy == "python":
        python_file = find_python_file(repo_dir)
        if python_file:
            relative_path = os.path.relpath(python_file, repo_dir)
            cmd = ["python", relative_path]
        else:
            print("No runnable file found in", repo_url)
            return None, None, None

    print(f"Using run strategy '{run_strategy}' with command:", cmd)
    # Change working directory to the repository directory.
    original_dir = os.getcwd()
    os.chdir(repo_dir)
    tracker.start()
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        timeout = 60  # seconds; adjust as needed
        print(f"Running for {timeout} seconds...")
        time.sleep(timeout)
        # For docker-compose, bring down the containers; otherwise, terminate.
        if run_strategy == "docker-compose":
            subprocess.run(["docker-compose", "down"])
        else:
            process.terminate()
            process.wait()
        cc_emissions = tracker.stop()
        cc_energy = getattr(tracker, '_last_measured_energy', None)
        return cc_emissions, cc_energy, None
    except Exception as e:
        print(f"Error running {repo_url}: {e}")
        tracker.stop()
        return None, None, None
    finally:
        os.chdir(original_dir)
        print("Returned to original directory:", os.getcwd())

def main():
    ensure_dependencies()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    if not os.path.exists(TOP_REPOS_CSV):
        raise FileNotFoundError(f"{TOP_REPOS_CSV} not found.")
    
    df = pd.read_csv(TOP_REPOS_CSV)
    print(f"Found {len(df)} top repositories to analyze with CodeCarbon")
    
    results = []
    for _, row in df.iterrows():
        country = row['country']
        repo_url = row['repo_link']
        runnability_score = row.get('runnability_score', None)
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = os.path.join(tmp_dir, "repo")
            if clone_repo(repo_url, target_dir):
                cc_emissions, cc_energy, _ = detect_and_run_repo(target_dir, repo_url, country)
                if cc_emissions is not None or cc_energy is not None:
                    results.append({
                        'country': country,
                        'repo_link': repo_url,
                        'runnability_score': runnability_score,
                        'codecarbon_emissions_kg': cc_emissions,
                        'codecarbon_energy_consumed_kwh': cc_energy,
                    })
                else:
                    print(f"No CodeCarbon data collected for {repo_url}")
            else:
                print(f"Skipping {repo_url} due to cloning failure")
    
    results_df = pd.DataFrame(results)
    output_csv = os.path.join(OUTPUT_DIR, "codecarbon_results.csv")
    results_df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
if __name__ == "__main__":
    main()
