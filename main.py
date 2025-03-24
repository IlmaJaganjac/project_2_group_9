import requests
import pandas as pd
import datetime
import time
import os
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# GitHub API token (create one at https://github.com/settings/tokens)
# Store this in a .env file as GITHUB_TOKEN=your_token_here
TOKEN = os.getenv('GITHUB_TOKEN')

# Headers for GitHub API requests
headers = {
    'Authorization': f'token {TOKEN}',
    'Accept': 'application/vnd.github.v3+json'
}

def get_repos(org):
    """Get all repositories for an organization"""
    repos = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/orgs/{org}/repos?per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching repos for {org}: {response.status_code}")
            print(response.json())
            break
        
        current_repos = response.json()
        if not current_repos:
            break
            
        repos.extend(current_repos)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
        
    return repos

def get_contributors(repo_full_name):
    """Get contributors for a repository"""
    contributors = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/contributors?per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching contributors for {repo_full_name}: {response.status_code}")
            break
            
        current_contributors = response.json()
        if not current_contributors:
            break
            
        contributors.extend(current_contributors)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
        
    return contributors

def get_issues(repo_full_name):
    """Get closed issues for a repository"""
    all_issues = []
    
    # Get closed issues
    closed_issues = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/issues?state=closed&per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching closed issues for {repo_full_name}: {response.status_code}")
            break
            
        current_issues = response.json()
        if not current_issues:
            break
            
        closed_issues.extend(current_issues)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
    
    # Get open issues
    open_issues = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/issues?state=open&per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching open issues for {repo_full_name}: {response.status_code}")
            break
            
        current_issues = response.json()
        if not current_issues:
            break
            
        open_issues.extend(current_issues)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
        
    all_issues = {'open': open_issues, 'closed': closed_issues}
    return all_issues

def get_pull_requests(repo_full_name):
    """Get pull requests for a repository"""
    all_prs = []
    
    # Get closed PRs
    closed_prs = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/pulls?state=closed&per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching closed PRs for {repo_full_name}: {response.status_code}")
            break
            
        current_prs = response.json()
        if not current_prs:
            break
            
        closed_prs.extend(current_prs)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
    
    # Get open PRs
    open_prs = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/pulls?state=open&per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching open PRs for {repo_full_name}: {response.status_code}")
            break
            
        current_prs = response.json()
        if not current_prs:
            break
            
        open_prs.extend(current_prs)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
        
    all_prs = {'open': open_prs, 'closed': closed_prs}
    return all_prs

def get_commits(repo_full_name):
    """Get commits for a repository"""
    commits = []
    page = 1
    while True:
        response = requests.get(
            f'https://api.github.com/repos/{repo_full_name}/commits?per_page=100&page={page}',
            headers=headers
        )
        if response.status_code != 200:
            print(f"Error fetching commits for {repo_full_name}: {response.status_code}")
            break
            
        current_commits = response.json()
        if not current_commits:
            break
            
        commits.extend(current_commits)
        page += 1
        
        # Respect GitHub's rate limiting
        time.sleep(1)
        
    return commits

def check_file_exists(repo_full_name, file_path):
    """Check if a file exists in a repository"""
    response = requests.get(
        f'https://api.github.com/repos/{repo_full_name}/contents/{file_path}',
        headers=headers
    )
    return response.status_code == 200

def get_workflows(repo_full_name):
    """Check if the repository has GitHub Actions workflows"""
    response = requests.get(
        f'https://api.github.com/repos/{repo_full_name}/actions/workflows',
        headers=headers
    )
    if response.status_code != 200:
        return []
    return response.json().get('workflows', [])

def calculate_metrics(repos, org_name, gov_id):
    """Calculate sustainability metrics for each repository"""
    metrics = []
    current_date = datetime.datetime.now()
    
    for repo in tqdm(repos, desc=f"Processing {org_name} repositories"):
        repo_full_name = repo['full_name']
        repo_created_at = datetime.datetime.strptime(repo['created_at'], '%Y-%m-%dT%H:%M:%SZ')
        repo_age_days = (current_date - repo_created_at).days
        repo_age_months = repo_age_days / 30.44  # Average days in a month
        
        # Get additional data
        contributors = get_contributors(repo_full_name)
        issues = get_issues(repo_full_name)
        prs = get_pull_requests(repo_full_name)
        commits = get_commits(repo_full_name)
        
        # Calculate issue resolution time
        issue_resolution_times = []
        for issue in issues['closed']:
            if 'created_at' in issue and 'closed_at' in issue:
                created_at = datetime.datetime.strptime(issue['created_at'], '%Y-%m-%dT%H:%M:%SZ')
                closed_at = datetime.datetime.strptime(issue['closed_at'], '%Y-%m-%dT%H:%M:%SZ')
                resolution_time = (closed_at - created_at).total_seconds() / 3600  # in hours
                issue_resolution_times.append(resolution_time)
        
        avg_issue_resolution_time = sum(issue_resolution_times) / len(issue_resolution_times) if issue_resolution_times else None
        
        # Calculate last commit date
        last_commit_date = None
        last_commit_age = None
        if commits:
            last_commit_date_str = commits[0]['commit']['committer']['date']
            last_commit_date = datetime.datetime.strptime(last_commit_date_str, '%Y-%m-%dT%H:%M:%SZ')
            last_commit_age = (current_date - last_commit_date).days
        
        # Check for documentation files
        has_readme = check_file_exists(repo_full_name, 'README.md')
        has_license = check_file_exists(repo_full_name, 'LICENSE')
        has_contributing = check_file_exists(repo_full_name, 'CONTRIBUTING.md')
        
        # Check for CI/CD
        has_cicd = len(get_workflows(repo_full_name)) > 0
        
        # Calculate percentage of external contributions
        owner_login = repo_full_name.split('/')[0]
        external_prs = [pr for pr in prs['closed'] + prs['open'] if pr['user']['login'] != owner_login]
        external_pr_percentage = len(external_prs) / len(prs['closed'] + prs['open']) * 100 if len(prs['closed'] + prs['open']) > 0 else 0
        
        # Gather metrics
        repo_metrics = {
            'government': gov_id,
            'org_name': org_name,
            'repo_name': repo['name'],
            'repo_full_name': repo_full_name,
            
            # Activity & Maintenance Metrics
            'repo_age_days': repo_age_days,
            'repo_age_months': repo_age_months,
            'commit_frequency_per_month': len(commits) / repo_age_months if repo_age_months > 0 else 0,
            'avg_issue_resolution_time_hours': avg_issue_resolution_time,
            'open_issues_percentage': len(issues['open']) / (len(issues['open']) + len(issues['closed'])) * 100 if (len(issues['open']) + len(issues['closed'])) > 0 else 0,
            'last_commit_date': last_commit_date,
            'days_since_last_commit': last_commit_age,
            
            # Collaboration & Community Engagement
            'num_contributors': len(contributors),
            'num_forks': repo['forks_count'],
            'num_stars': repo['stargazers_count'],
            'merged_pr_percentage': len(prs['closed']) / (len(prs['closed']) + len(prs['open'])) * 100 if (len(prs['closed']) + len(prs['open'])) > 0 else 0,
            'external_pr_percentage': external_pr_percentage,
            
            # Code & Documentation Quality
            'has_readme': has_readme,
            'has_license': has_license,
            'has_contributing': has_contributing,
            'has_cicd': has_cicd,
            
            # Raw counts for reference
            'total_commits': len(commits),
            'open_issues': len(issues['open']),
            'closed_issues': len(issues['closed']),
            'open_prs': len(prs['open']),
            'closed_prs': len(prs['closed'])
        }
        
        metrics.append(repo_metrics)
        
    return metrics

def load_existing_metrics(file_path):
    """Load existing metrics from a CSV file if it exists"""
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    return None

def save_metrics(metrics, file_path):
    """Append new metrics to the existing CSV file instead of overwriting"""
    new_df = pd.DataFrame(metrics)

    # If the file exists, append data without overwriting
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.drop_duplicates(subset=['repo_full_name', 'government'], keep='last', inplace=True)
        combined_df.to_csv(file_path, index=False)
    else:
        # If the file doesn’t exist, create it
        new_df.to_csv(file_path, index=False)

    return len(new_df), len(new_df) 

    # """Save metrics to a CSV file"""
    # # Create DataFrame from new metrics
    # new_df = pd.DataFrame(metrics)
    
    # # Check if file already exists and load it
    # existing_df = load_existing_metrics(file_path)
    
    # if existing_df is not None:
    #     # Combine existing and new data
    #     combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    #     # Remove duplicates based on repo_full_name and government
    #     combined_df = combined_df.drop_duplicates(subset=['repo_full_name', 'government'], keep='last')
    #     combined_df.to_csv(file_path, index=False)
    #     return len(new_df), len(combined_df)
    # else:
    #     # If file doesn't exist, just save the new data
    #     new_df.to_csv(file_path, index=False)
    #     return len(new_df), len(new_df)

def process_organization(org_name, gov_id, output_file):
    """Process a single organization and add to the combined dataset"""
    print(f"Fetching repositories for {org_name}...")
    repos = get_repos(org_name)
    print(f"Found {len(repos)} repositories.")
    
    print(f"Calculating sustainability metrics for {org_name}...")
    metrics = calculate_metrics(repos, org_name, gov_id)
    
    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')
        print("Created 'data' directory")
    
    # Save metrics to the combined file
    new_count, total_count = save_metrics(metrics, output_file)
    print(f"Added {new_count} repositories from {org_name} to dataset. Total repositories: {total_count}")
    
    return metrics

def main():
    """Main function to process organizations"""
    # Define output file path
    output_file = "data/government_repos_sustainability_metrics.csv"
    
    # Define organizations to scrape
    # Format: [organization_name, government_identifier]
    organizations = [
        # Netherlands
        #["dataoverheid", "netherlands"],
        # USA  
        #["usagov", "usa"],   
        # Germany            
        #["opencode18", "germany"],
        # France
        #["etalab", "france"] 
        # Greece     
        #["govgr", "greece"]              
    ]
    
    # Process each organization
    for org_name, gov_id in organizations:
        process_organization(org_name, gov_id, output_file)
    
    print(f"All metrics saved to {output_file}")

if __name__ == "__main__":
    main()