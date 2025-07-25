#!/usr/bin/env python3
"""
Git Commit Message Data Collector
Collects code diff -> commit message pairs from GitHub
"""

import requests
import json
import os
import time
import re
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

@dataclass
class CommitPair:
    diff: str
    message: str
    repo: str
    sha: str
    quality_score: float

class GitCommitCollector:
    def __init__(self, token: str):
        self.token = token
        self.headers = {'Authorization': f'token {token}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def get_popular_repos(self, language: str = None) -> List[Dict]:
        """Get popular repositories for commit data"""
        print(f"🔍 Finding popular repositories...")
        
        queries = [
            "stars:>1000 size:<50000",
            "stars:>500 language:python size:<30000",
            "stars:>500 language:javascript size:<30000",
            "stars:>200 topic:machine-learning",
            "stars:>200 topic:web-development"
        ]
        
        all_repos = []
        for query in queries:
            try:
                url = "https://api.github.com/search/repositories"
                params = {'q': query, 'sort': 'stars', 'per_page': 20}
                
                response = self.session.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    all_repos.extend(data.get('items', []))
                    time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"❌ Error in query: {e}")
        
        # Remove duplicates and sort by stars
        unique_repos = {repo['full_name']: repo for repo in all_repos}.values()
        sorted_repos = sorted(unique_repos, key=lambda x: x['stargazers_count'], reverse=True)
        
        print(f"✅ Found {len(sorted_repos)} repositories")
        return list(sorted_repos)[:50]  # Top 50 repos
    
    def get_commits(self, repo_full_name: str, max_commits: int = 100) -> List[Dict]:
        """Get commits from repository"""
        print(f"📁 Getting commits from {repo_full_name}...")
        
        commits = []
        page = 1
        
        while len(commits) < max_commits and page <= 5:  # Max 5 pages
            try:
                url = f"https://api.github.com/repos/{repo_full_name}/commits"
                params = {
                    'per_page': 30,
                    'page': page,
                    'since': '2023-01-01T00:00:00Z'  # Recent commits
                }
                
                response = self.session.get(url, params=params)
                if response.status_code != 200:
                    break
                
                page_commits = response.json()
                if not page_commits:
                    break
                
                commits.extend(page_commits)
                page += 1
                time.sleep(0.5)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error getting commits: {e}")
                break
        
        print(f"   ✅ Found {len(commits)} commits")
        return commits[:max_commits]
    
    def get_commit_diff(self, repo_full_name: str, sha: str) -> str:
        """Get commit diff"""
        try:
            url = f"https://api.github.com/repos/{repo_full_name}/commits/{sha}"
            response = self.session.get(url)
            
            if response.status_code == 200:
                commit_data = response.json()
                files = commit_data.get('files', [])
                
                # Build simplified diff
                diff_parts = []
                for file in files[:5]:  # Max 5 files per commit
                    filename = file.get('filename', '')
                    status = file.get('status', '')
                    additions = file.get('additions', 0)
                    deletions = file.get('deletions', 0)
                    
                    # Skip large files or binary files
                    if additions + deletions > 100 or not filename.endswith(('.py', '.js', '.jsx', '.ts', '.tsx')):
                        continue
                    
                    patch = file.get('patch', '')
                    if patch:
                        # Clean up patch for training
                        clean_patch = self.clean_diff(patch)
                        if clean_patch:
                            diff_parts.append(f"File: {filename} ({status})\n{clean_patch}")
                
                return '\n\n'.join(diff_parts)
            
        except Exception as e:
            print(f"❌ Error getting diff for {sha}: {e}")
        
        return ""
    
    def clean_diff(self, patch: str) -> str:
        """Clean diff patch for training"""
        lines = patch.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip diff headers
            if line.startswith('@@') or line.startswith('+++') or line.startswith('---'):
                continue
            
            # Keep added/removed lines
            if line.startswith('+') or line.startswith('-'):
                cleaned_lines.append(line)
            elif line.strip() and not line.startswith('\\'):
                cleaned_lines.append(' ' + line)  # Context line
        
        return '\n'.join(cleaned_lines[:20])  # Max 20 lines
    
    def is_quality_commit(self, commit: Dict) -> bool:
        """Check if commit message is good quality"""
        message = commit.get('commit', {}).get('message', '')
        
        # Skip merge commits
        if message.lower().startswith('merge'):
            return False
        
        # Skip very short messages
        if len(message.strip()) < 10:
            return False
        
        # Skip automated commits
        automated_patterns = [
            'bump version', 'update dependencies', 'auto-generated',
            'automated', 'bot:', 'dependabot', 'renovate'
        ]
        
        message_lower = message.lower()
        if any(pattern in message_lower for pattern in automated_patterns):
            return False
        
        # Prefer commits with good structure
        good_patterns = [
            'fix:', 'feat:', 'add:', 'update:', 'refactor:', 'docs:',
            'fix ', 'add ', 'update ', 'refactor ', 'implement'
        ]
        
        return any(pattern in message_lower for pattern in good_patterns)
    
    def calculate_commit_quality(self, commit: Dict, diff: str) -> float:
        """Calculate quality score for commit"""
        message = commit.get('commit', {}).get('message', '')
        score = 0.0
        
        # Message length (sweet spot 20-100 chars)
        msg_len = len(message)
        if 20 <= msg_len <= 100:
            score += 0.3
        elif 10 <= msg_len <= 150:
            score += 0.2
        
        # Has conventional commit format
        if re.match(r'^(feat|fix|docs|style|refactor|test|chore):', message.lower()):
            score += 0.3
        
        # Good action words
        action_words = ['fix', 'add', 'update', 'implement', 'refactor', 'improve']
        if any(word in message.lower() for word in action_words):
            score += 0.2
        
        # Diff quality
        if diff and len(diff.split('\n')) > 3:
            score += 0.2
        
        return min(score, 1.0)
    
    def collect_commit_data(self, target_pairs: int = 1500) -> List[CommitPair]:
        """Collect commit message training data"""
        print(f"🚀 Starting commit data collection (target: {target_pairs})")
        
        repos = self.get_popular_repos()
        all_pairs = []
        
        for repo in repos:
            if len(all_pairs) >= target_pairs:
                break
            
            repo_name = repo['full_name']
            commits = self.get_commits(repo_name, max_commits=50)
            
            repo_pairs = []
            for commit in commits:
                if len(all_pairs) >= target_pairs:
                    break
                
                # Check commit quality
                if not self.is_quality_commit(commit):
                    continue
                
                # Get commit diff
                sha = commit['sha']
                diff = self.get_commit_diff(repo_name, sha)
                
                if not diff or len(diff.strip()) < 50:
                    continue
                
                message = commit['commit']['message'].split('\n')[0]  # First line only
                quality_score = self.calculate_commit_quality(commit, diff)
                
                pair = CommitPair(
                    diff=diff,
                    message=message,
                    repo=repo_name,
                    sha=sha,
                    quality_score=quality_score
                )
                
                repo_pairs.append(pair)
                all_pairs.append(pair)
                
                time.sleep(0.3)  # Rate limiting
            
            print(f"   ✅ {repo_name}: {len(repo_pairs)} commit pairs")
        
        # Sort by quality
        all_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        
        print(f"🎉 Collected {len(all_pairs)} commit pairs")
        return all_pairs[:target_pairs]
    
    def save_commit_data(self, pairs: List[CommitPair], filename: str):
        """Save commit data"""
        data = []
        for pair in pairs:
            data.append({
                'code': pair.diff,  # Use diff as 'code'
                'comment': pair.message,  # Use message as 'comment'
                'language': 'diff',
                'repo': pair.repo,
                'sha': pair.sha,
                'quality_score': pair.quality_score,
                'comment_type': 'commit_message'
            })
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} commit pairs to {filename}")

def main():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    collector = GitCommitCollector(token)
    
    print("🎯 Git Commit Data Collection Starting...")
    print("Target: 1,500 high-quality commit message pairs")
    print("=" * 60)
    
    # Collect commit data
    commit_pairs = collector.collect_commit_data(target_pairs=1500)
    
    # Save data
    collector.save_commit_data(commit_pairs, 'data/enhanced_commit_pairs.json')
    
    print(f"\n🎉 Commit data collection completed!")
    print(f"📊 Total pairs: {len(commit_pairs)}")
    print(f"⭐ Average quality score: {sum(p.quality_score for p in commit_pairs) / len(commit_pairs):.2f}")

if __name__ == "__main__":
    main()