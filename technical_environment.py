import os
import csv
import time
import base64
from pathlib import Path
import pandas as pd
import requests
import tempfile
import subprocess
import shutil
from google import genai
import math
from tqdm import tqdm
import re
import json
from collections import Counter
import networkx as nx
from datetime import datetime
from urllib.parse import urlparse
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import seaborn as sns
from matplotlib.ticker import MaxNLocator
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Configuration
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MIN_SIZE_THRESHOLD_MB = 0.1   # Minimum repo size in MB (e.g., 100 KB)
SIZE_THRESHOLD_MB = 100       # Maximum repo size in MB to avoid extremely large projects
CLONE_TIMEOUT_SECONDS = 30    # Timeout in seconds for cloning a repository
MAX_RETRIES = 3               # Max retries for API calls
ADVANCED_SCORE_THRESHOLD = 7  # M

# Set up API clients
headers = {
    "Authorization": f"token {GITHUB_TOKEN}",
    "Accept": "application/vnd.github.v3+json"
}

# Configure Gemini
client = genai.Client(api_key=GEMINI_API_KEY)

SENTIMENT_DICT = {
    'good': 2, 'great': 3, 'awesome': 4, 'fixed': 2, 'happy': 3, 'rejoice': 4, 'amazing': 3, 'love': 3, 
    'wow': 3, 'precious': 3, ':)': 2, ':-)': 2,
    'bad': -2, 'error': -2, 'terrible': -4, 'sad': -4, 'hate': -4, 'stupid': -3, 'sigh': -1, ':(': -2, ':-(': -2, 
    'cry': -4, 'not': -2, 'cool': 2,
    'very': 1, 'really': 1, '!!': 1, '!!!': 1, '???': -1
}

SUSTAINABLE_PATTERNS = {
    'documentation': [
        r'"""[\s\S]*?"""',  # Python docstrings
        r'/\*\*[\s\S]*?\*/',  # JSDoc comments
        r'/// <summary>[\s\S]*?</summary>',  # C# XML comments
        r'#\s*@param',  # Parameter documentation
        r'#\s*@return',  # Return value documentation
    ],
    'testing': [
        r'import\s+[\'"]testing[\'"]',
        r'import\s+[\'"]pytest[\'"]',
        r'import\s+[\'"]unittest[\'"]',
        r'@Test',
        r'test\w*\([^)]*\)\s*{',
        r'assert\w*\(',  # Assertions in tests
        r'expect\(',  # Expectations in tests
        r'mock\w*\(',  # Mocking in tests
    ],
    'modularity': [
        r'import\s+',
        r'from\s+[\w.]+\s+import',
        r'require\([\'"]',
        r'export\s+',
        r'class\s+\w+',  # Class definitions
        r'interface\s+\w+',  # Interface definitions
        r'function\s+\w+',  # Function definitions
        r'def\s+\w+',  # Python function definitions
    ],
    'error_handling': [
        r'try\s*{',
        r'try:',
        r'catch\s*\(',
        r'except',
        r'finally',
        r'throw\s+new\s+\w+',
        r'raise\s+',
        r'error\w*\s*=',  # Error variable assignments
        r'log\w*\s*\(\s*[\'"]error',  # Error logging
        r'console\.error',  # Console error logging
    ],
    'config_management': [
        r'\.env',
        r'config\.',
        r'process\.env',
        r'os\.environ',
        r'settings\.',  # Settings modules
        r'constants\.',  # Constants
        r'getenv\(',  # Environment variable access
    ],
    'adaptability': [
        r'interface\s+\w+',  # Interfaces for adaptability
        r'abstract\s+class',  # Abstract classes
        r'extends\s+\w+',  # Inheritance
        r'implements\s+\w+',  # Interface implementation
        r'@Override',  # Method overriding
        r'super\(',  # Parent class method calls
        r'factory\.',  # Factory pattern usage
    ],
    'security': [
        r'sanitize',  # Input sanitization
        r'validate',  # Input validation
        r'escape',  # Output escaping
        r'authenticate',  # Authentication
        r'authorize',  # Authorization
        r'encrypt',  # Encryption
        r'decrypt',  # Decryption
        r'hash',  # Hashing
        r'token',  # Token usage
        r'permission',  # Permission checks
    ],
    'scalability': [
        r'cache',  # Caching
        r'pool',  # Connection pooling
        r'queue',  # Queue usage
        r'async',  # Asynchronous code
        r'await',  # Await keyword
        r'parallel',  # Parallel processing
        r'concurrent',  # Concurrent processing
        r'thread',  # Threading
        r'worker',  # Worker processes/threads
    ]
}

UNSUSTAINABLE_PATTERNS = {
    'code_smells': [
        r'TODO',
        r'FIXME',
        r'HACK',
        r'XXX',
        r'WTF',  # More code smells
        r'NOSONAR',  # SonarQube suppression
        r'CHECKSTYLE:OFF',  # Checkstyle suppression
    ],
    'hard_coding': [
        r'API_KEY\s*=\s*[\'"][^\'"]+[\'"]',
        r'password\s*=\s*[\'"][^\'"]+[\'"]',
        r'SECRET\s*=\s*[\'"][^\'"]+[\'"]',
        r'http[s]?://[^\s<>"\']+',  # Hardcoded URLs
        r'[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}',  # IP addresses
    ],
    'long_methods': [
        r'function\s+\w+\([^)]*\)\s*{[\s\S]{1000,}}',
        r'def\s+\w+\([^)]*\)[\s\S]{1000,}return',
        r'public\s+\w+\s+\w+\([^)]*\)\s*{[\s\S]{1000,}}',  # Java/C# methods
    ],
    'deep_nesting': [
        r'if\s*\([^)]+\)\s*{\s*if\s*\([^)]+\)\s*{\s*if',
        r'for\s*\([^)]+\)\s*{\s*for\s*\([^)]+\)\s*{\s*for',  # Nested loops
        r'while\s*\([^)]+\)\s*{\s*while\s*\([^)]+\)\s*{\s*while',  # Nested while loops
    ],
    'large_classes': [
        r'class\s+\w+[^{]*{[\s\S]{3000,}}',  # Large classes (>3000 chars)
    ],
    'naming_issues': [
        r'\b[a-z]{1,2}\b',  # Very short variable names
        r'\b[A-Z0-9_]+\b',  # CONSTANTS used as variables
    ],
    'commented_code': [
        r'//\s*[a-zA-Z0-9]+\s*\([^)]*\)',  # Commented out function calls
        r'//\s*if\s*\(',  # Commented out if statements
        r'//\s*for\s*\(',  # Commented out for loops
        r'#\s*[a-zA-Z0-9]+\s*\([^)]*\)',  # Python commented out function calls
    ]
}

ENVIRONMENTAL_INDICATORS = {
    'high_computation': [
        r'train\(',  # ML model training
        r'fit\(',  # ML model fitting
        r'\.cuda',  # GPU usage
        r'gpu',  # GPU related code
        r'tensorflow',  # TensorFlow usage
        r'torch',  # PyTorch usage
        r'while\s*\(\s*true',  # Infinite loops
    ],
    'resource_efficient': [
        r'yield',  # Generators for memory efficiency
        r'streaming',  # Streaming APIs
        r'lazy',  # Lazy evaluation
        r'throttle',  # Rate limiting
        r'debounce',  # Debouncing
    ],
    'energy_awareness': [
        r'power',  # Power management
        r'battery',  # Battery optimization
        r'energy',  # Energy management
        r'sleep',  # Sleep modes
    ]
}

SOCIAL_INDICATORS = {
    'inclusive_design': [
        r'a11y',  # Accessibility
        r'accessibility',  # Accessibility
        r'aria-',  # ARIA attributes
        r'alt=',  # Alt text for images
        r'i18n',  # Internationalization
        r'l10n',  # Localization
    ],
    'privacy_focused': [
        r'gdpr',  # GDPR compliance
        r'consent',  # User consent
        r'privacy',  # Privacy consideration
        r'anonymize',  # Data anonymization
        r'pseudonymize',  # Data pseudonymization
    ],
    'ethical_considerations': [
        r'ethic',  # Ethics
        r'bias',  # Bias consideration
        r'fairness',  # Fairness consideration
        r'diversity',  # Diversity consideration
        r'inclusion',  # Inclusion consideration
    ]
}

def clone_repo(repo_url, target_dir):
    """Clone a GitHub repository to a local directory"""
    try:
        subprocess.run(["git", "clone", repo_url, target_dir], 
                      check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error cloning repository {repo_url}: {e}")
        return False

def get_code_files(repo_dir, exclude_dirs=None):
    """Get all relevant code files from the repository"""
    if exclude_dirs is None:
        exclude_dirs = ['node_modules', '.git', 'vendor', '__pycache__', 'build', 'dist', 'venv', 'env', '.venv']
    
    code_extensions = ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go', '.rb', '.php', '.cpp', '.c', '.h', '.swift', '.kt', '.rs']
    files = []
    
    for root, dirs, filenames in os.walk(repo_dir):
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for filename in filenames:
            if any(filename.endswith(ext) for ext in code_extensions):
                files.append(os.path.join(root, filename))
    
    return files

def analyze_code_patterns(files):
    """Analyze code files for sustainable and unsustainable patterns"""
    results = {
        'sustainable': {k: 0 for k in SUSTAINABLE_PATTERNS},
        'unsustainable': {k: 0 for k in UNSUSTAINABLE_PATTERNS},
        'environmental': {k: 0 for k in ENVIRONMENTAL_INDICATORS},
        'social': {k: 0 for k in SOCIAL_INDICATORS},
        'total_lines': 0,
        'file_count': len(files),
        'file_types': Counter(),
        'avg_file_size': 0,
        'function_count': 0,
        'class_count': 0,
        'comment_ratio': 0
    }
    
    total_size = 0
    total_comment_lines = 0
    
    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            lines = content.split('\n')
            line_count = len(lines)
            results['total_lines'] += line_count
            file_size = os.path.getsize(file_path)
            total_size += file_size
            comment_patterns = [r'^\s*#', r'^\s*//', r'^\s*/\*', r'^\s*\*', r'^\s*\*/']
            comment_lines = sum(1 for line in lines if any(re.match(pattern, line) for pattern in comment_patterns))
            total_comment_lines += comment_lines
            function_matches = len(re.findall(r'(function\s+\w+|def\s+\w+)', content))
            class_matches = len(re.findall(r'(class\s+\w+)', content))
            results['function_count'] += function_matches
            results['class_count'] += class_matches
            ext = os.path.splitext(file_path)[1]
            results['file_types'][ext] += 1
            for category, patterns in SUSTAINABLE_PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    results['sustainable'][category] += len(matches)
            for category, patterns in UNSUSTAINABLE_PATTERNS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    results['unsustainable'][category] += len(matches)
            for category, patterns in ENVIRONMENTAL_INDICATORS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    results['environmental'][category] += len(matches)
            for category, patterns in SOCIAL_INDICATORS.items():
                for pattern in patterns:
                    matches = re.findall(pattern, content)
                    results['social'][category] += len(matches)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
    
    if len(files) > 0:
        results['avg_file_size'] = total_size / len(files)
    if results['total_lines'] > 0:
        results['comment_ratio'] = total_comment_lines / results['total_lines']
    
    return results

def calculate_cyclomatic_complexity(repo_dir):
    """Estimate cyclomatic complexity using radon for Python files"""
    complexity_data = {
        "average": 0, 
        "max": 0, 
        "complex_functions": 0,
        "maintainability_index": 0,
        "halstead_metrics": {
            "volume": 0,
            "difficulty": 0,
            "effort": 0
        }
    }
    
    try:
        result = subprocess.run(["pip", "show", "radon"], capture_output=True, text=True)
        if "not found" in result.stderr:
            subprocess.run(["pip", "install", "radon"], check=True)
        result = subprocess.run(
            ["radon", "cc", repo_dir, "--average", "--total-average", "-s"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        avg_match = re.search(r'Average complexity: ([0-9.]+)', output)
        if avg_match:
            complexity_data["average"] = float(avg_match.group(1))
        complex_count = output.count(" C ")
        complexity_data["complex_functions"] = complex_count
        max_pattern = re.compile(r'- ([A-Z]) \(([0-9]+)\)')
        matches = max_pattern.findall(output)
        if matches:
            complexities = [int(m[1]) for m in matches]
            if complexities:
                complexity_data["max"] = max(complexities)
        result = subprocess.run(
            ["radon", "mi", repo_dir, "-s"],
            capture_output=True, text=True, check=True
        )
        output = result.stdout
        mi_values = []
        for line in output.splitlines():
            if " - " in line:
                try:
                    mi_value = float(line.split(" - ")[1].strip())
                    mi_values.append(mi_value)
                except:
                    pass
        if mi_values:
            complexity_data["maintainability_index"] = sum(mi_values) / len(mi_values)
        try:
            result = subprocess.run(
                ["radon", "hal", repo_dir],
                capture_output=True, text=True, check=True
            )
            output = result.stdout
            volume_values = []
            difficulty_values = []
            effort_values = []
            for line in output.splitlines():
                if "h1:" in line and "h2:" in line:
                    parts = line.split()
                    for part in parts:
                        if part.startswith("volume:"):
                            try:
                                volume_values.append(float(part.split(":")[1]))
                            except:
                                pass
                        elif part.startswith("difficulty:"):
                            try:
                                difficulty_values.append(float(part.split(":")[1]))
                            except:
                                pass
                        elif part.startswith("effort:"):
                            try:
                                effort_values.append(float(part.split(":")[1]))
                            except:
                                pass
            if volume_values:
                complexity_data["halstead_metrics"]["volume"] = sum(volume_values) / len(volume_values)
            if difficulty_values:
                complexity_data["halstead_metrics"]["difficulty"] = sum(difficulty_values) / len(difficulty_values)
            if effort_values:
                complexity_data["halstead_metrics"]["effort"] = sum(effort_values) / len(effort_values)
        except:
            pass
    except Exception as e:
        logging.error(f"Error calculating complexity metrics: {e}")
    
    return complexity_data

def analyze_dependency_graph(repo_dir):
    """Analyze the dependency graph of the repository"""
    dependency_data = {
        "total_dependencies": 0,
        "direct_dependencies": 0,
        "dev_dependencies": 0,
        "outdated_dependencies": 0,
        "dependency_graph": {
            "nodes": 0,
            "edges": 0,
            "avg_degree": 0
        }
    }
    
    package_json_path = os.path.join(repo_dir, "package.json")
    if os.path.exists(package_json_path):
        try:
            with open(package_json_path, 'r', encoding='utf-8') as f:
                package_data = json.load(f)
            dependencies = package_data.get("dependencies", {})
            dev_dependencies = package_data.get("devDependencies", {})
            dependency_data["direct_dependencies"] = len(dependencies)
            dependency_data["dev_dependencies"] = len(dev_dependencies)
            dependency_data["total_dependencies"] = len(dependencies) + len(dev_dependencies)
            lock_file_path = os.path.join(repo_dir, "package-lock.json")
            if os.path.exists(lock_file_path):
                try:
                    with open(lock_file_path, 'r', encoding='utf-8') as f:
                        lock_data = json.load(f)
                    if "dependencies" in lock_data:
                        all_deps = lock_data["dependencies"]
                        dependency_data["dependency_graph"]["nodes"] = len(all_deps)
                        G = nx.DiGraph()
                        for dep_name, dep_info in all_deps.items():
                            G.add_node(dep_name)
                            if "requires" in dep_info:
                                for req_name in dep_info["requires"]:
                                    G.add_edge(dep_name, req_name)
                        dependency_data["dependency_graph"]["edges"] = G.number_of_edges()
                        if G.number_of_nodes() > 0:
                            dependency_data["dependency_graph"]["avg_degree"] = G.number_of_edges() / G.number_of_nodes()
                except Exception as e:
                    logging.error(f"Error analyzing package-lock.json: {e}")
        except Exception as e:
            logging.error(f"Error analyzing package.json: {e}")
    
    requirements_path = os.path.join(repo_dir, "requirements.txt")
    if os.path.exists(requirements_path):
        try:
            with open(requirements_path, 'r', encoding='utf-8') as f:
                requirements = f.readlines()
            direct_deps = [line.strip() for line in requirements if line.strip() and not line.startswith('#')]
            dependency_data["direct_dependencies"] = len(direct_deps)
            dependency_data["total_dependencies"] = len(direct_deps)
        except Exception as e:
            logging.error(f"Error analyzing requirements.txt: {e}")
    
    return dependency_data

def check_test_coverage(repo_dir):
    """Try to determine test coverage"""
    coverage_data = {
        "has_tests": False, 
        "test_files": 0, 
        "test_to_code_ratio": 0,
        "test_lines": 0,
        "test_frameworks": []
    }
    
    test_dirs = ['tests', 'test', '__tests__', 'spec', 'unit_tests', 'integration_tests', 'e2e']
    test_patterns = ['*_test.py', '*_spec.js', 'test_*.py', '*Test.java', '*Spec.js', '*_test.go', '*_test.rb']
    test_frameworks = {
        'pytest': r'import\s+pytest',
        'jest': r'import\s+.*\s+from\s+[\'"]@testing-library',
        'mocha': r'(describe|it)\s*\(',
        'junit': r'import\s+.*\s+from\s+[\'"]junit',
        'unittest': r'import\s+unittest',
        'rspec': r'RSpec\.',
        'go_test': r'func\s+Test\w+\('
    }
    
    test_files = []
    code_files = []
    test_lines_count = 0
    detected_frameworks = set()
    
    for root, dirs, files in os.walk(repo_dir):
        in_test_dir = any(test_dir in root.split(os.path.sep) for test_dir in test_dirs)
        for file in files:
            file_path = os.path.join(root, file)
            if not any(file.endswith(ext) for ext in ['.py', '.js', '.jsx', '.ts', '.tsx', '.java', '.cs', '.go', '.rb', '.php']):
                continue
            is_test_file = in_test_dir or any(re.match(pattern.replace('*', '.*'), file) for pattern in test_patterns)
            if is_test_file:
                test_files.append(file_path)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        test_lines_count += len(content.split('\n'))
                        for framework, pattern in test_frameworks.items():
                            if re.search(pattern, content):
                                detected_frameworks.add(framework)
                except Exception as e:
                    logging.error(f"Error reading test file {file_path}: {e}")
            else:
                code_files.append(file_path)

    coverage_data["has_tests"] = len(test_files) > 0
    coverage_data["test_files"] = len(test_files)
    coverage_data["test_lines"] = test_lines_count
    coverage_data["test_frameworks"] = list(detected_frameworks)
    if len(code_files) > 0:
        coverage_data["test_to_code_ratio"] = len(test_files) / len(code_files)
    
    coverage_files = [
        '.coveragerc',
        'coverage.xml',
        'coverage.json',
        'jest.config.js',
        'cypress.json',
        'codecov.yml',
        '.nycrc'
    ]
    coverage_data["has_coverage_config"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in coverage_files
    )
    
    return coverage_data

def analyze_commit_history(repo_dir):
    """Analyze the commit history for patterns"""
    commit_data = {
        "total_commits": 0,
        "commit_frequency": 0,
        "active_days": 0,
        "contributors": 0,
        "avg_commit_size": 0,
        "commit_message_quality": 0,
        "test_driven_commits": 0
    }
    
    try:
        result = subprocess.run(
            ["git", "-C", repo_dir, "rev-list", "--count", "HEAD"],
            capture_output=True, text=True, check=True
        )
        commit_data["total_commits"] = int(result.stdout.strip())
        result = subprocess.run(
            ["git", "-C", repo_dir, "log", "--format=%ad", "--date=short"],
            capture_output=True, text=True, check=True
        )
        commit_dates = result.stdout.strip().split('\n')
        unique_dates = set(commit_dates)
        commit_data["active_days"] = len(unique_dates)
        if commit_dates:
            try:
                first_date = datetime.strptime(commit_dates[-1], "%Y-%m-%d")
                last_date = datetime.strptime(commit_dates[0], "%Y-%m-%d")
                days_diff = (last_date - first_date).days + 1
                weeks = max(1, days_diff / 7)
                commit_data["commit_frequency"] = commit_data["total_commits"] / weeks
            except Exception as e:
                logging.error(f"Error calculating commit frequency: {e}")
        result = subprocess.run(
            ["git", "-C", repo_dir, "log", "--format=%ae"],
            capture_output=True, text=True, check=True
        )
        commit_emails = result.stdout.strip().split('\n')
        commit_data["contributors"] = len(set(commit_emails))
        result = subprocess.run(
            ["git", "-C", repo_dir, "log", "--format=%s", "--name-only"],
            capture_output=True, text=True, check=True
        )
        commit_logs = result.stdout.strip().split('\n\n')
        good_messagePatterns = [
            r'^(feat|fix|docs|style|refactor|test|chore)(\(.+\))?:',
            r'^[A-Z]',
            r'.{10,}',
        ]
        good_messages = 0
        test_commits = 0
        for i in range(0, len(commit_logs), 2):
            if i+1 < len(commit_logs):
                message = commit_logs[i]
                files = commit_logs[i+1].split('\n') if i+1 < len(commit_logs) else []
                if any(re.search(pattern, message) for pattern in good_messagePatterns):
                    good_messages += 1
                if any('test' in f.lower() for f in files):
                    test_commits += 1
        if commit_data["total_commits"] > 0:
            commit_data["commit_message_quality"] = good_messages / commit_data["total_commits"]
            commit_data["test_driven_commits"] = test_commits
    except Exception as e:
        logging.error(f"Error analyzing commit history: {e}")
    
    return commit_data

def analyze_repo_structure(repo_dir):
    """Analyze the repository structure"""
    structure = {
        "has_readme": False,
        "has_license": False,
        "has_gitignore": False,
        "has_ci_config": False,
        "has_dependency_manager": False,
        "has_docker": False,
        "has_contribution_guide": False,
        "has_code_of_conduct": False,
        "has_security_policy": False,
        "folder_depth": 0,
        "dependency_count": 0,
        "architecture_score": 0
    }
    
    structure["has_readme"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["README.md", "README", "readme.md"]
    )
    structure["has_license"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["LICENSE", "LICENSE.md", "license.txt"]
    )
    structure["has_gitignore"] = os.path.exists(os.path.join(repo_dir, ".gitignore"))
    structure["has_docker"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["Dockerfile", "docker-compose.yml", ".dockerignore"]
    )
    structure["has_contribution_guide"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["CONTRIBUTING.md", "CONTRIBUTE.md", ".github/CONTRIBUTING.md"]
    )
    structure["has_code_of_conduct"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["CODE_OF_CONDUCT.md", ".github/CODE_OF_CONDUCT.md"]
    )
    structure["has_security_policy"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["SECURITY.md", ".github/SECURITY.md", "security.md"]
    )
    ci_files = ['.travis.yml', '.github/workflows', 'circleci', 'jenkinsfile']
    structure["has_ci_config"] = any(
        os.path.exists(os.path.join(repo_dir, f)) if os.path.isfile(os.path.join(repo_dir, f))
        else os.path.isdir(os.path.join(repo_dir, f))
        for f in ci_files
    )
    structure["has_dependency_manager"] = any(
        os.path.exists(os.path.join(repo_dir, f)) for f in ["package.json", "requirements.txt", "Gemfile", "composer.json"]
    )
    max_depth = 0
    for root, dirs, files in os.walk(repo_dir):
        depth = root[len(repo_dir):].count(os.sep)
        if depth > max_depth:
            max_depth = depth
    structure["folder_depth"] = max_depth
    dep_files = [f for f in os.listdir(repo_dir) if f in ["package.json", "requirements.txt", "Gemfile", "composer.json"]]
    structure["dependency_count"] = len(dep_files)
    score = 0
    if structure["has_readme"]:
        score += 1
    if structure["has_license"]:
        score += 1
    if structure["has_gitignore"]:
        score += 1
    if structure["has_ci_config"]:
        score += 1
    if structure["has_dependency_manager"]:
        score += 1
    if structure["has_docker"]:
        score += 1
    if structure["has_contribution_guide"]:
        score += 1
    if structure["has_code_of_conduct"]:
        score += 1
    if structure["has_security_policy"]:
        score += 1
    structure["architecture_score"] = score / 9.0
    
    return structure

def read_repo_links(csv_path):
    """Read repository links from a CSV file with structure: country,org/group,repo_link"""
    repos = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                if len(row) >= 3:
                    repos.append({
                        'country': row[0],
                        'org': row[1],
                        'repo_link': row[2]
                    })
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
    return repos

def derive_gemini_scores(repo_dir, files):
    """Use the Gemini 2.0 Flash model to analyze repository code and derive additional sustainability scores."""
    code_samples = []
    max_total_chars = 100000  # Reduced to avoid Gemini API limits
    total_chars = 0
    for file in files:
        try:
            with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            if total_chars + len(code) < max_total_chars:
                code_samples.append(f"File: {file}\n{code}\n")
                total_chars += len(code)
            else:
                break
        except Exception as e:
            logging.error(f"Error reading file {file}: {e}")
    
    aggregated_code = "\n".join(code_samples)
    
    prompt = f"""
You are an expert code sustainability evaluator. Given the repository code samples, provide an evaluation with numerical scores (0-100) for the following metrics:

Please respond ONLY with a JSON object with the following structure and no other text:
{{
  "overall_sustainability": [score],
  "documentation_quality": [score],
  "testing_robustness": [score],
  "modularity_and_design": [score],
  "error_handling": [score],
  "security_best_practices": [score],
  "scalability_potential": [score],
  "environmental_efficiency": [score],
  "social_inclusiveness": [score],
  "critical_issues": ["issue1", "issue2", ...],
  "improvement_suggestions": ["suggestion1", "suggestion2", ...]
}}

Repository code:
{aggregated_code}
"""
    default_scores = {
        "overall_sustainability": 0,
        "documentation_quality": 0,
        "testing_robustness": 0,
        "modularity_and_design": 0,
        "error_handling": 0,
        "security_best_practices": 0,
        "scalability_potential": 0,
        "environmental_efficiency": 0,
        "social_inclusiveness": 0,
        "critical_issues": [],
        "improvement_suggestions": []
    }
    
    gemini_scores = default_scores.copy()
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            try:
                gemini_scores = json.loads(response.text)
                logging.info("Successfully parsed Gemini response as JSON")
                break
            except Exception as parse_error:
                logging.error(f"Error parsing Gemini response as JSON: {parse_error}")
                text = response.text
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    try:
                        json_str = json_match.group(0)
                        gemini_scores = json.loads(json_str)
                        logging.info("Extracted JSON successfully")
                        break
                    except Exception as e:
                        logging.error(f"Failed to extract JSON: {e}")
                        gemini_scores = default_scores.copy()
                        gemini_scores["raw_response"] = text
                else:
                    gemini_scores = default_scores.copy()
                    gemini_scores["raw_response"] = text
        except Exception as e:
            logging.error(f"Error calling Gemini model (attempt {attempt+1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                gemini_scores = default_scores.copy()
                gemini_scores["error"] = str(e)
    
    return gemini_scores

def analyze_commit_sentiment(commit_message):
    """Analyze sentiment of a commit message"""
    sentences = re.split(r'[.!?]+\s*', commit_message)
    sentences = [s.strip() for s in sentences if s.strip()]
    if not sentences:
        return 0
    max_pos_score = 0
    min_neg_score = 0
    for sentence in sentences:
        words = re.findall(r'\w+|[^\w\s]', sentence.lower(), re.UNICODE)
        pos_score = 0
        neg_score = 0
        modifier = 1
        prev_word = ''
        for word in words:
            if word in ['very', 'really']:
                modifier = 1.5
                continue
            elif word == 'not':
                modifier = -0.5
                continue
            elif word in ['!!', '!!!']:
                pos_score += 1 if pos_score > 0 else 0
                neg_score -= 1 if neg_score < 0 else 0
                continue
            elif word == '???':
                neg_score -= 1 if neg_score < 0 else 0
                continue
            if prev_word == 'not' and word == 'cool':
                score = SENTIMENT_DICT.get('cool', 0) * modifier
                neg_score = min(neg_score, score)
                modifier = 1
                prev_word = word
                continue
            score = SENTIMENT_DICT.get(word, 0)
            if score > 0:
                pos_score = max(pos_score, score * modifier)
            elif score < 0:
                neg_score = min(neg_score, score * modifier)
            modifier = 1
            prev_word = word
        max_pos_score = max(max_pos_score, pos_score)
        min_neg_score = min(min_neg_score, neg_score)
    if abs(max_pos_score) <= 1 and abs(min_neg_score) <= 1:
        return 0
    elif min_neg_score * 1.5 < max_pos_score:
        return max_pos_score
    else:
        return min_neg_score

def analyze_repository(repo_url, target_dir):
    """High-level function to analyze a repository."""
    logging.debug(f"Analyzing repository: {repo_url}")
    if not clone_repo(repo_url, target_dir):
        logging.warning(f"Failed to clone repository: {repo_url}")
        return None
    
    files = get_code_files(target_dir)
    
    overall_analysis = {
        "code_analysis": None,
        "complexity_analysis": None,
        "dependency_analysis": None,
        "test_coverage": None,
        "commit_history": None,
        "repo_structure": None,
        "gemini_scores": None,
        "social_metrics": None
    }
    
    if not files:
        logging.info(f"No code files found in repository: {repo_url}")
        overall_analysis["repo_structure"] = analyze_repo_structure(target_dir)
        overall_analysis["commit_history"] = analyze_commit_history(target_dir)
        overall_analysis["social_metrics"] = {
            'avg_sentiment': 0,
            'positive_ratio': 0,
            'negative_ratio': 0,
            'neutral_ratio': 0,
            'community_engagement': 0,
            'diversity_index': 0,
            'retention_rate': 0,
            'has_code_of_conduct': overall_analysis['repo_structure'].get('has_code_of_conduct', False),
            'has_contribution_guide': overall_analysis['repo_structure'].get('has_contribution_guide', False)
        }
        return overall_analysis
    
    code_analysis = analyze_code_patterns(files)
    complexity_analysis = calculate_cyclomatic_complexity(target_dir)
    dependency_analysis = analyze_dependency_graph(target_dir)
    test_coverage = check_test_coverage(target_dir)
    commit_history = analyze_commit_history(target_dir)
    repo_structure = analyze_repo_structure(target_dir)
    gemini_scores = derive_gemini_scores(target_dir, files)
    
    result = subprocess.run(
        ["git", "-C", target_dir, "log", "--format=%s"],
        capture_output=True, text=True, check=True
    )
    commit_messages = result.stdout.strip().split('\n')
    sentiment_scores = [analyze_commit_sentiment(msg) for msg in commit_messages if msg.strip()]
    total_commits = len(sentiment_scores)
    if total_commits > 0:
        avg_sentiment = sum(sentiment_scores) / total_commits
        positive_count = sum(1 for s in sentiment_scores if s > 1)
        negative_count = sum(1 for s in sentiment_scores if s < -1)
        neutral_count = total_commits - positive_count - negative_count
    else:
        avg_sentiment = 0
        positive_count = 0
        negative_count = 0
        neutral_count = 0
    
    result = subprocess.run(
        ["git", "-C", target_dir, "log", "--format=%ad", "--date=short"],
        capture_output=True, text=True, check=True
    )
    commit_dates = result.stdout.strip().split('\n')
    if commit_dates and commit_dates[0]:
        first_date = datetime.strptime(commit_dates[-1], "%Y-%m-%d")
        last_date = datetime.strptime(commit_dates[0], "%Y-%m-%d")
        repo_age_weeks = max(1, (last_date - first_date).days / 7)
    else:
        repo_age_weeks = 1
    
    issue_related_commits = sum(1 for msg in commit_messages if "issue" in msg.lower() or "pr" in msg.lower())
    community_engagement = issue_related_commits / repo_age_weeks
    
    result = subprocess.run(
        ["git", "-C", target_dir, "log", "--format=%ae"],
        capture_output=True, text=True, check=True
    )
    emails = result.stdout.strip().split('\n')
    domains = [email.split('@')[-1] if '@' in email else 'unknown' for email in emails]
    domain_counts = Counter(domains)
    total_contributors = len(emails)
    if total_contributors > 0:
        diversity_index = -sum((count / total_contributors) * math.log(count / total_contributors) for count in domain_counts.values() if count > 0)
    else:
        diversity_index = 0
    
    author_counts = Counter(emails)
    contributors_with_multiple_commits = sum(1 for count in author_counts.values() if count > 1)
    retention_rate = (contributors_with_multiple_commits / len(author_counts)) * 100 if author_counts else 0
    
    has_code_of_conduct = repo_structure.get('has_code_of_conduct', False)
    has_contribution_guide = repo_structure.get('has_contribution_guide', False)
    
    social_metrics = {
        'avg_sentiment': avg_sentiment,
        'positive_ratio': positive_count / total_commits if total_commits > 0 else 0,
        'negative_ratio': negative_count / total_commits if total_commits > 0 else 0,
        'neutral_ratio': neutral_count / total_commits if total_commits > 0 else 0,
        'community_engagement': community_engagement,
        'diversity_index': diversity_index,
        'retention_rate': retention_rate,
        'has_code_of_conduct': has_code_of_conduct,
        'has_contribution_guide': has_contribution_guide
    }
    
    overall_analysis.update({
        "code_analysis": code_analysis,
        "complexity_analysis": complexity_analysis,
        "dependency_analysis": dependency_analysis,
        "test_coverage": test_coverage,
        "commit_history": commit_history,
        "repo_structure": repo_structure,
        "gemini_scores": gemini_scores,
        "social_metrics": social_metrics
    })
    
    return overall_analysis

def calculate_runnability_score(analysis):
    """Calculate a runnability score based on advanced criteria for energy measurement suitability."""
    logging.debug("Calculating runnability score")
    if not analysis:
        logging.warning("Analysis is None, returning 0")
        return 0
    
    score = 0
    max_score = 15
    
    # Check for dependency manager
    repo_structure = analysis.get('repo_structure', {})
    logging.debug(f"repo_structure: {repo_structure}")
    if repo_structure.get('has_dependency_manager', False):
        score += 3
        logging.debug("Added 3 points for dependency manager")
    
    # Check for tests
    test_coverage = analysis.get('test_coverage', {})
    logging.debug(f"test_coverage: {test_coverage}")
    if (test_coverage and 
        isinstance(test_coverage, dict) and 
        test_coverage.get('has_tests', False) and 
        test_coverage.get('test_to_code_ratio', 0) > 0.1):
        score += 3
        logging.debug("Added 3 points for tests")
    
    # Check complexity
    complexity_analysis = analysis.get('complexity_analysis', {})
    logging.debug(f"complexity_analysis: {complexity_analysis}")
    complexity = complexity_analysis.get('average', 0)
    if 1 < complexity < 10:
        score += 2
        logging.debug("Added 2 points for complexity")
    
    # Check commit frequency
    commit_history = analysis.get('commit_history', {})
    logging.debug(f"commit_history: {commit_history}")
    commit_frequency = commit_history.get('commit_frequency', 0)
    if commit_frequency > 0.5:
        score += 2
        logging.debug("Added 2 points for commit frequency")
    
    # Check modularity
    code_analysis = analysis.get('code_analysis', {})
    logging.debug(f"code_analysis: {code_analysis}")
    sustainable = code_analysis.get('sustainable', {}) if code_analysis else {}
    modularity = sustainable.get('modularity', 0)
    if modularity > 5:
        score += 2
        logging.debug("Added 2 points for modularity")
    
    # Check environmental efficiency
    gemini_scores = analysis.get('gemini_scores', {})
    logging.debug(f"gemini_scores: {gemini_scores}")
    environmental_efficiency = gemini_scores.get('environmental_efficiency', 0) if isinstance(gemini_scores, dict) else 0
    if environmental_efficiency > 50:
        score += 2
        logging.debug("Added 2 points for environmental efficiency")
    
    # Check size constraint
    if code_analysis:
        avg_file_size = code_analysis.get('avg_file_size', 0)
        file_count = code_analysis.get('file_count', 0)
        total_size_mb = (avg_file_size * file_count) / (1024 * 1024) if file_count else 0
        if MIN_SIZE_THRESHOLD_MB < total_size_mb < SIZE_THRESHOLD_MB:
            score += 1
            logging.debug("Added 1 point for size constraint")
    
    final_score = (score / max_score) * 100
    logging.debug(f"Final runnability score: {final_score}")
    return final_score

def generate_runnability_plots(results, output_dir):
    """Generate plots for runnability analysis including energy consumption per language per country."""
    logging.debug("Generating runnability plots")
    data = []
    for r in results:
        if not r:
            continue
        try:
            runnability_score = calculate_runnability_score(r)
            file_count = r.get('code_analysis', {}).get('file_count', 0)
            commit_frequency = r.get('commit_history', {}).get('commit_frequency', 0)
            test_ratio = r.get('test_coverage', {}).get('test_to_code_ratio', 0)
            data.append({
                'country': r['metadata']['country'],
                'runnability_score': runnability_score,
                'file_count': file_count,
                'commit_frequency': commit_frequency,
                'test_ratio': test_ratio
            })
        except Exception as e:
            logging.error(f"Error processing result for plotting: {e}")
            continue
    
    if not data:
        logging.warning("No data available for plotting")
        return
    
    df = pd.DataFrame(data)
    
    # Plot 1: Runnability Score Distribution by Country
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='country', y='runnability_score', data=df)
    plt.xticks(rotation=45)
    plt.title('Runnability Score Distribution Across Countries')
    plt.xlabel('Country')
    plt.ylabel('Runnability Score (0-100)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runnability_by_country.png'))
    plt.close()
    
    # Plot 2: Scatter of File Count vs Runnability
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='file_count', y='runnability_score', hue='country', size='commit_frequency', data=df)
    plt.title('File Count vs Runnability Score')
    plt.xlabel('Number of Code Files')
    plt.ylabel('Runnability Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'file_count_vs_runnability.png'))
    plt.close()
    
    # Plot 3: Test Ratio vs Runnability
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='test_ratio', y='runnability_score', hue='country', data=df)
    plt.title('Test-to-Code Ratio vs Runnability Score')
    plt.xlabel('Test-to-Code Ratio')
    plt.ylabel('Runnability Score')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'test_ratio_vs_runnability.png'))
    plt.close()
    
    # New Plot: Average Energy Consumption by Programming Language per Country
    extension_mapping = {
        '.py': 'Python',
        '.js': 'JavaScript',
        '.jsx': 'JavaScript',
        '.ts': 'TypeScript',
        '.tsx': 'TypeScript',
        '.java': 'Java',
        '.cs': 'C#',
        '.go': 'Go',
        '.rb': 'Ruby',
        '.php': 'PHP',
        '.cpp': 'C++',
        '.c': 'C',
        '.h': 'C/C++',
        '.swift': 'Swift',
        '.kt': 'Kotlin',
        '.rs': 'Rust'
    }
    
    energy_mapping = {
        'C': 1,
        'C++': 1.34,
        'Rust': 1.03,
        'Ada': 1.7,
        'Java': 1.98,
        'Pascal': 2.14,
        'Chapel': 2.18,
        'Go': 3.23,
        'Swift': 2.79,
        'C#': 3.14,
        'JavaScript': 4.45,
        'TypeScript': 21.5,
        'Python': 75.88,
        'Ruby': 69.91,
        'PHP': 29.3,
        'Lua': 45.98,
        'Perl': 79.58,
        'JRuby': 46.54,
        'Racket': 7.91,
        'Haskell': 3.1
    }
    
    data = []
    for r in results:
        if not r or not r.get('code_analysis', {}).get('file_types'):
            continue
        try:
            most_common_ext = max(r['code_analysis']['file_types'], key=r['code_analysis']['file_types'].get)
            language = extension_mapping.get(most_common_ext, 'Unknown')
            energy_val = energy_mapping.get(language, 0)
            file_count = r['code_analysis'].get('file_count', 0)
            avg_file_size = r['code_analysis'].get('avg_file_size', 0)
            total_file_size_mb = (avg_file_size * file_count) / (1024 * 1024) if file_count else 0

            data.append({
                'country': r['metadata']['country'],
                'language': language,
                'energy': energy_val,
                'file_size_mb': total_file_size_mb
            })
        except Exception as e:
            logging.error(f"Error processing result for energy plot: {e}")
            continue

    df = pd.DataFrame(data)
    df = df[df['energy'] > 0]  # Filter out unknown energy values

    if df.empty:
        logging.warning("No data available for energy consumption plot")
        return

    # Aggregating data
    df_grouped = df.groupby(['country', 'language']).agg(
        total_energy=('energy', 'sum'),
        avg_file_size=('file_size_mb', 'mean'),
        repo_count=('language', 'count')
    ).reset_index()

    # Sorting countries and languages
    countries = df_grouped['country'].unique()
    languages = df_grouped['language'].unique()

    # Plotting
    plt.figure(figsize=(12, 8))
    sns.set_palette("Paired")

    # Create the stacked bar plot
    bottom_position = {country: 0 for country in countries}
    for language in languages:
        subset = df_grouped[df_grouped['language'] == language]
        heights = []
        for country in countries:
            row = subset[subset['country'] == country]
            if not row.empty:
                energy = row['total_energy'].values[0]
            else:
                energy = 0
            heights.append(energy)

        plt.bar(countries, heights, bottom=[bottom_position[c] for c in countries], 
                label=language, width=0.8, alpha=0.8)

        # Update bottom position
        for i, country in enumerate(countries):
            bottom_position[country] += heights[i]

    plt.title("Estimated Energy Consumption per Country and Language")
    plt.xlabel("Country")
    plt.ylabel("Estimated Energy Consumption (Joules)")
    plt.legend(title="Programming Language", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    # Save the plot
    plot_path = os.path.join(output_dir, 'country_language_energy.png')
    plt.savefig(plot_path)
    plt.close()

    logging.info(f"Plot saved at {plot_path}")

def social_dimension_analysis(results, output_dir):
    """Aggregate social dimension metrics and save to CSV"""
    logging.debug("Performing social dimension analysis")
    repo_data = []
    for r in results:
        if not r or 'social_metrics' not in r:
            logging.warning(f"Skipping result due to missing social_metrics: {r.get('metadata', {}).get('repo_link', 'Unknown')}")
            continue
        try:
            gemini_scores = r.get('gemini_scores', {})
            social_inclusiveness = gemini_scores.get('social_inclusiveness', 0) if isinstance(gemini_scores, dict) else 0
            repo_data.append({
                'country': r['metadata']['country'],
                'repo_link': r['metadata']['repo_link'],
                'avg_sentiment': r['social_metrics']['avg_sentiment'],
                'positive_ratio': r['social_metrics']['positive_ratio'],
                'negative_ratio': r['social_metrics']['negative_ratio'],
                'neutral_ratio': r['social_metrics']['neutral_ratio'],
                'contributors': r['commit_history']['contributors'],
                'social_inclusiveness': social_inclusiveness,
                'community_engagement': r['social_metrics']['community_engagement'],
                'diversity_index': r['social_metrics']['diversity_index'],
                'retention_rate': r['social_metrics']['retention_rate'],
                'has_code_of_conduct': r['social_metrics']['has_code_of_conduct'],
                'has_contribution_guide': r['social_metrics']['has_contribution_guide']
            })
        except Exception as e:
            logging.error(f"Error processing repository {r['metadata']['repo_link']}: {e}")
            continue
    
    if not repo_data:
        logging.warning("No valid repository data for social dimension analysis.")
        return pd.DataFrame()
    
    # Create DataFrame for repository-level data
    repo_df = pd.DataFrame(repo_data)
    
    # Aggregate by country
    country_stats = repo_df.groupby('country').agg({
        'avg_sentiment': 'mean',
        'positive_ratio': 'mean',
        'negative_ratio': 'mean',
        'neutral_ratio': 'mean',
        'contributors': 'sum',
        'social_inclusiveness': 'mean',
        'community_engagement': 'mean',
        'diversity_index': 'mean',
        'retention_rate': 'mean',
        'has_code_of_conduct': 'mean',
        'has_contribution_guide': 'mean'
    }).reset_index()
    
    # Save repository-level data to CSV
    repo_csv_path = os.path.join(output_dir, 'social_metrics_repos.csv')
    repo_df.to_csv(repo_csv_path, index=False)
    logging.info(f"Repository-level social metrics saved to {repo_csv_path}")
    
    # Save country-level aggregated data to CSV
    country_csv_path = os.path.join(output_dir, 'social_metrics_country.csv')
    country_stats.to_csv(country_csv_path, index=False)
    logging.info(f"Country-level social metrics saved to {country_csv_path}")
    
    return country_stats

if __name__ == "__main__":
    csv_path = "repo_links.csv"
    output_dir = "analysis_results"
    
    os.makedirs(output_dir, exist_ok=True)
    repos = read_repo_links(csv_path)
    logging.info(f"Found {len(repos)} repositories in CSV")
    
    # Limit to 5 repositories for debugging
    
    logging.info(f"Limited to {len(repos)} repositories for debugging")
    
    results = []
    country_repo_count = Counter()
    
    for repo_info in tqdm(repos, desc="Processing repositories"):
        country = repo_info['country']
        if country_repo_count[country] >= 30:
            logging.info(f"Skipping repository {repo_info['repo_link']} (country limit reached)")
            continue
        
        logging.info(f"Processing repository: {repo_info['repo_link']}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            target_dir = os.path.join(tmp_dir, "repo")
            logging.info(f"Cloning repository to {target_dir}")
            analysis = analyze_repository(repo_info['repo_link'], target_dir)
            
            if analysis:
                analysis['metadata'] = {
                    'country': repo_info['country'],
                    'org': repo_info['org'],
                    'repo_link': repo_info['repo_link']
                }
                results.append(analysis)
                country_repo_count[country] += 1
                repo_name = repo_info['repo_link'].split('/')[-1].replace('.git', '')
                file_name = f"{repo_info['country']}_{repo_name}.json"
                with open(os.path.join(output_dir, file_name), 'w', encoding='utf-8') as f:
                    json.dump(analysis, f, indent=2)
                logging.info(f"Analysis for {repo_info['repo_link']} saved to {os.path.join(output_dir, file_name)}")
            else:
                logging.warning(f"Failed to analyze repository: {repo_info['repo_link']}")
    
    with open(os.path.join(output_dir, "all_results.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    logging.info(f"All analyses completed. Combined results saved to {os.path.join(output_dir, 'all_results.json')}")
    
    country_stats = social_dimension_analysis(results, output_dir)
    logging.info(f"Social dimension analysis completed. Metrics saved in {output_dir}")
    
    runnability_results = []
    for result in results:
        try:
            score = calculate_runnability_score(result)
            runnability_results.append({
                'country': result['metadata']['country'],
                'repo_link': result['metadata']['repo_link'],
                'runnability_score': score
            })
        except Exception as e:
            logging.error(f"Error calculating runnability score for {result['metadata']['repo_link']}: {e}")
            continue
    
    df = pd.DataFrame(runnability_results)
    top_repos = df.groupby('country').apply(
        lambda x: x.nlargest(5, 'runnability_score'),
        include_groups=False
    ).reset_index(drop=True)
    
    top_repos['country'] = top_repos['repo_link'].apply(
        lambda x: next((r['metadata']['country'] for r in results if r['metadata']['repo_link'] == x), None)
    )
    
    top_repos_csv = os.path.join(output_dir, 'top_5_repos_per_country.csv')
    top_repos.to_csv(top_repos_csv, index=False)
    logging.info(f"Top 5 repos per country saved to {top_repos_csv}")
    
    generate_runnability_plots(results, output_dir)
    logging.info(f"Plots saved in {output_dir}")
