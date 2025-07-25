#!/usr/bin/env python3
"""
Advanced data collector for production-quality training data
Collects 5,000-10,000 high-quality code-comment pairs
"""

import requests
import time
import json
import os
import ast
import re
import threading
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib

load_dotenv()

@dataclass
class CodeCommentPair:
    code: str
    comment: str
    language: str
    repo: str
    file_path: str
    quality_score: float
    comment_type: str  # 'docstring', 'inline', 'block', 'jsdoc'

class AdvancedGitHubCollector:
    def __init__(self, token: str):
        self.token = token
        self.headers = {'Authorization': f'token {token}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.collected_pairs = []
        self.seen_hashes = set()  # Prevent duplicates
        
    def get_quality_repositories(self, language: str, min_stars: int = 50) -> List[Dict]:
        """Get high-quality repositories with good documentation"""
        print(f"🔍 Searching for {language} repositories...")
        
        # Multiple search queries for diversity
        queries = [
            f"language:{language} stars:>{min_stars} size:<50000 documentation",
            f"language:{language} stars:>{min_stars*2} size:<30000",
            f"language:{language} stars:>{min_stars} topic:tutorial",
            f"language:{language} stars:>{min_stars} topic:library",
            f"language:{language} stars:>{min_stars} topic:framework",
            f"language:{language} stars:>{min_stars} topic:api",
            f"language:{language} stars:>{min_stars} topic:machine-learning",
            f"language:{language} stars:>{min_stars} topic:web-development"
        ]
        
        all_repos = []
        
        for query in queries:
            try:
                for page in range(1, 6):  # 5 pages per query
                    url = "https://api.github.com/search/repositories"
                    params = {
                        'q': query,
                        'sort': 'stars',
                        'per_page': 30,
                        'page': page
                    }
                    
                    response = self.session.get(url, params=params)
                    if response.status_code != 200:
                        print(f"⚠️  API error: {response.status_code}")
                        break
                        
                    data = response.json()
                    if not data.get('items'):
                        break
                        
                    all_repos.extend(data['items'])
                    time.sleep(0.5)  # Rate limiting
                    
            except Exception as e:
                print(f"❌ Error in query '{query}': {e}")
                continue
        
        # Remove duplicates and sort by stars
        unique_repos = {repo['full_name']: repo for repo in all_repos}.values()
        sorted_repos = sorted(unique_repos, key=lambda x: x['stargazers_count'], reverse=True)
        
        print(f"✅ Found {len(sorted_repos)} unique repositories")
        return list(sorted_repos)[:200]  # Top 200 repos
    
    def get_code_files(self, repo_full_name: str, language: str) -> List[str]:
        """Get code files from repository with better filtering"""
        extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx', '.mjs']
        }
        
        try:
            # Try main branch first, then master
            for branch in ['main', 'master']:
                url = f"https://api.github.com/repos/{repo_full_name}/git/trees/{branch}?recursive=1"
                response = self.session.get(url)
                
                if response.status_code == 200:
                    data = response.json()
                    break
            else:
                return []
            
            files = []
            for item in data.get('tree', []):
                if item['type'] == 'blob':
                    for ext in extensions.get(language, []):
                        if item['path'].endswith(ext):
                            # Filter out test files, examples, and build files
                            path_lower = item['path'].lower()
                            if not any(skip in path_lower for skip in [
                                'test', 'spec', '__pycache__', 'node_modules', 
                                'build', 'dist', 'example', 'demo', '.min.js'
                            ]):
                                files.append(item['path'])
            
            # Prioritize files likely to have good documentation
            priority_files = []
            regular_files = []
            
            for file_path in files:
                path_lower = file_path.lower()
                if any(priority in path_lower for priority in [
                    'src/', 'lib/', 'core/', 'main', 'index', 'api/', 'model'
                ]):
                    priority_files.append(file_path)
                else:
                    regular_files.append(file_path)
            
            # Return prioritized files first, limit total
            return (priority_files + regular_files)[:30]
            
        except Exception as e:
            print(f"❌ Error getting files from {repo_full_name}: {e}")
            return []
    
    def get_file_content(self, repo_full_name: str, file_path: str) -> str:
        """Get file content with better error handling"""
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
        
        try:
            response = self.session.get(url)
            if response.status_code != 200:
                return ""
                
            data = response.json()
            if data.get('encoding') == 'base64':
                import base64
                content = base64.b64decode(data['content']).decode('utf-8', errors='ignore')
                return content
                
        except Exception as e:
            print(f"❌ Error fetching {file_path}: {e}")
            
        return ""
    
    def extract_python_pairs(self, code: str, repo: str, file_path: str) -> List[CodeCommentPair]:
        """Enhanced Python code-comment extraction"""
        pairs = []
        
        try:
            tree = ast.parse(code)
            lines = code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    # Get node source code
                    start_line = node.lineno - 1
                    end_line = getattr(node, 'end_lineno', start_line + 20)
                    
                    if end_line >= len(lines):
                        end_line = len(lines)
                    
                    node_lines = lines[start_line:end_line]
                    node_code = '\n'.join(node_lines)
                    
                    # Get docstring
                    docstring = ast.get_docstring(node)
                    
                    if docstring and self.is_quality_comment(docstring):
                        # Remove docstring from code
                        clean_code = self.remove_docstring_from_code(node_code, docstring)
                        
                        if clean_code.strip():
                            quality_score = self.calculate_quality_score(clean_code, docstring, 'python')
                            
                            pair = CodeCommentPair(
                                code=clean_code.strip(),
                                comment=docstring.strip(),
                                language='python',
                                repo=repo,
                                file_path=file_path,
                                quality_score=quality_score,
                                comment_type='docstring'
                            )
                            
                            # Check for duplicates
                            pair_hash = self.get_pair_hash(pair.code, pair.comment)
                            if pair_hash not in self.seen_hashes:
                                self.seen_hashes.add(pair_hash)
                                pairs.append(pair)
                                
        except Exception as e:
            print(f"❌ Error parsing Python code: {e}")
            
        return pairs
    
    def extract_javascript_pairs(self, code: str, repo: str, file_path: str) -> List[CodeCommentPair]:
        """Enhanced JavaScript code-comment extraction"""
        pairs = []
        
        try:
            # JSDoc pattern matching
            jsdoc_pattern = r'/\*\*(.*?)\*/'
            function_patterns = [
                r'function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}',
                r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>\s*\{[^}]*\}',
                r'(\w+)\s*:\s*function\s*\([^)]*\)\s*\{[^}]*\}',
                r'class\s+(\w+)\s*\{[^}]*\}'
            ]
            
            jsdoc_matches = list(re.finditer(jsdoc_pattern, code, re.DOTALL))
            
            for jsdoc_match in jsdoc_matches:
                jsdoc_end = jsdoc_match.end()
                comment = jsdoc_match.group(1).strip()
                
                if not self.is_quality_comment(comment):
                    continue
                
                # Find the next function/class after this comment
                for pattern in function_patterns:
                    func_matches = list(re.finditer(pattern, code[jsdoc_end:], re.DOTALL))
                    
                    for func_match in func_matches:
                        if func_match.start() < 200:  # Comment should be close to function
                            function_code = func_match.group(0)
                            
                            quality_score = self.calculate_quality_score(function_code, comment, 'javascript')
                            
                            pair = CodeCommentPair(
                                code=function_code,
                                comment=comment,
                                language='javascript',
                                repo=repo,
                                file_path=file_path,
                                quality_score=quality_score,
                                comment_type='jsdoc'
                            )
                            
                            pair_hash = self.get_pair_hash(pair.code, pair.comment)
                            if pair_hash not in self.seen_hashes:
                                self.seen_hashes.add(pair_hash)
                                pairs.append(pair)
                            break
                    
                    if pairs:  # Found a match, move to next JSDoc
                        break
                        
        except Exception as e:
            print(f"❌ Error parsing JavaScript code: {e}")
            
        return pairs
    
    def is_quality_comment(self, comment: str) -> bool:
        """Check if comment meets quality standards"""
        if not comment or len(comment.strip()) < 20:
            return False
            
        words = comment.split()
        if len(words) < 5:
            return False
            
        # Check for meaningful content
        meaningful_words = ['function', 'method', 'class', 'return', 'parameter', 
                          'argument', 'calculate', 'process', 'handle', 'create',
                          'update', 'delete', 'get', 'set', 'initialize', 'validate']
        
        if not any(word.lower() in comment.lower() for word in meaningful_words):
            return False
            
        # Avoid auto-generated or low-quality comments
        bad_patterns = ['todo', 'fixme', 'hack', 'temporary', 'placeholder']
        if any(pattern in comment.lower() for pattern in bad_patterns):
            return False
            
        return True
    
    def calculate_quality_score(self, code: str, comment: str, language: str) -> float:
        """Calculate quality score for code-comment pair"""
        score = 0.0
        
        # Length factors
        code_lines = len(code.split('\n'))
        comment_words = len(comment.split())
        
        if 5 <= code_lines <= 50:
            score += 0.3
        if 10 <= comment_words <= 100:
            score += 0.3
            
        # Content quality
        if any(keyword in comment.lower() for keyword in ['args:', 'returns:', 'parameters:', 'example:']):
            score += 0.2
            
        if any(keyword in comment.lower() for keyword in ['raises:', 'throws:', 'note:', 'warning:']):
            score += 0.1
            
        # Code complexity (more complex code deserves better comments)
        if language == 'python':
            if any(keyword in code for keyword in ['class ', 'def ', 'async ', 'await ']):
                score += 0.1
        elif language == 'javascript':
            if any(keyword in code for keyword in ['class ', 'function ', 'async ', 'await ']):
                score += 0.1
                
        return min(score, 1.0)
    
    def get_pair_hash(self, code: str, comment: str) -> str:
        """Generate hash for duplicate detection"""
        content = f"{code.strip()}{comment.strip()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def remove_docstring_from_code(self, code: str, docstring: str) -> str:
        """Remove docstring from code"""
        # Simple removal patterns
        patterns = [
            f'"""{docstring}"""',
            f"'''{docstring}'''",
            f'"{docstring}"',
            f"'{docstring}'"
        ]
        
        for pattern in patterns:
            code = code.replace(pattern, '')
            
        return code
    
    def process_repository(self, repo: Dict, language: str) -> List[CodeCommentPair]:
        """Process a single repository"""
        repo_name = repo['full_name']
        print(f"📁 Processing {repo_name} ({repo['stargazers_count']} ⭐)")
        
        files = self.get_code_files(repo_name, language)
        repo_pairs = []
        
        for file_path in files[:15]:  # Limit files per repo
            try:
                content = self.get_file_content(repo_name, file_path)
                if not content:
                    continue
                    
                if language == 'python':
                    pairs = self.extract_python_pairs(content, repo_name, file_path)
                else:
                    pairs = self.extract_javascript_pairs(content, repo_name, file_path)
                
                repo_pairs.extend(pairs)
                
                if len(pairs) > 0:
                    print(f"  📄 {file_path}: {len(pairs)} pairs")
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                print(f"  ❌ Error processing {file_path}: {e}")
                continue
        
        print(f"  ✅ {repo_name}: {len(repo_pairs)} total pairs")
        return repo_pairs
    
    def collect_language_data(self, language: str, target_pairs: int = 2500) -> List[CodeCommentPair]:
        """Collect data for a specific language"""
        print(f"\n🚀 Starting {language} data collection (target: {target_pairs} pairs)")
        
        repos = self.get_quality_repositories(language)
        all_pairs = []
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_repo = {
                executor.submit(self.process_repository, repo, language): repo 
                for repo in repos[:100]  # Process top 100 repos
            }
            
            for future in as_completed(future_to_repo):
                try:
                    pairs = future.result()
                    all_pairs.extend(pairs)
                    
                    print(f"📊 Progress: {len(all_pairs)}/{target_pairs} pairs collected")
                    
                    if len(all_pairs) >= target_pairs:
                        print(f"🎯 Target reached for {language}!")
                        break
                        
                except Exception as e:
                    repo = future_to_repo[future]
                    print(f"❌ Error processing {repo['full_name']}: {e}")
        
        # Sort by quality score and return best pairs
        all_pairs.sort(key=lambda x: x.quality_score, reverse=True)
        return all_pairs[:target_pairs]
    
    def save_enhanced_data(self, pairs: List[CodeCommentPair], filename: str):
        """Save data with enhanced metadata"""
        data = []
        for pair in pairs:
            data.append({
                'code': pair.code,
                'comment': pair.comment,
                'language': pair.language,
                'repo': pair.repo,
                'file_path': pair.file_path,
                'quality_score': pair.quality_score,
                'comment_type': pair.comment_type
            })
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved {len(data)} pairs to {filename}")

def main():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("❌ Please set GITHUB_TOKEN environment variable")
        return
    
    collector = AdvancedGitHubCollector(token)
    
    print("🎯 Advanced Data Collection Starting...")
    print("Target: 5,000+ high-quality code-comment pairs")
    print("=" * 60)
    
    # Collect Python data
    python_pairs = collector.collect_language_data('python', target_pairs=3000)
    collector.save_enhanced_data(python_pairs, 'data/enhanced_python_pairs.json')
    
    # Collect JavaScript data  
    js_pairs = collector.collect_language_data('javascript', target_pairs=2000)
    collector.save_enhanced_data(js_pairs, 'data/enhanced_javascript_pairs.json')
    
    # Combine all data
    all_pairs = python_pairs + js_pairs
    collector.save_enhanced_data(all_pairs, 'data/enhanced_all_pairs.json')
    
    print(f"\n🎉 Data collection completed!")
    print(f"📊 Total pairs: {len(all_pairs)}")
    print(f"🐍 Python pairs: {len(python_pairs)}")
    print(f"🟨 JavaScript pairs: {len(js_pairs)}")
    print(f"⭐ Average quality score: {sum(p.quality_score for p in all_pairs) / len(all_pairs):.2f}")

if __name__ == "__main__":
    main()