import requests
import time
import json
import os
import ast
import re
from pathlib import Path
from typing import List, Tuple, Dict
from dotenv import load_dotenv

load_dotenv()

class GitHubCodeCollector:
    def __init__(self, token: str):
        self.token = token
        self.headers = {'Authorization': f'token {token}'}
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        
    def search_repositories(self, language: str, min_stars: int = 100, max_repos: int = 50) -> List[Dict]:
        """Search for repositories with good documentation practices"""
        repos = []
        page = 1
        
        while len(repos) < max_repos:
            query = f"language:{language} stars:>{min_stars} size:<10000"
            url = f"https://api.github.com/search/repositories"
            params = {
                'q': query,
                'sort': 'stars',
                'per_page': 30,
                'page': page
            }
            
            try:
                response = self.session.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if not data.get('items'):
                    break
                    
                repos.extend(data['items'])
                page += 1
                time.sleep(1)  # Rate limiting
                
            except requests.exceptions.RequestException as e:
                print(f"Error fetching repositories: {e}")
                break
                
        return repos[:max_repos]
    
    def get_file_content(self, repo_full_name: str, file_path: str) -> str:
        """Get content of a specific file from repository"""
        url = f"https://api.github.com/repos/{repo_full_name}/contents/{file_path}"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            if data.get('encoding') == 'base64':
                import base64
                return base64.b64decode(data['content']).decode('utf-8')
                
        except Exception as e:
            print(f"Error fetching file {file_path}: {e}")
            
        return ""
    
    def get_repository_files(self, repo_full_name: str, language: str) -> List[str]:
        """Get list of code files from repository"""
        extensions = {
            'python': ['.py'],
            'javascript': ['.js', '.jsx', '.ts', '.tsx']
        }
        
        url = f"https://api.github.com/repos/{repo_full_name}/git/trees/main?recursive=1"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            files = []
            for item in data.get('tree', []):
                if item['type'] == 'blob':
                    for ext in extensions.get(language, []):
                        if item['path'].endswith(ext):
                            files.append(item['path'])
                            
            return files[:20]  # Limit files per repo
            
        except Exception as e:
            print(f"Error fetching repository tree: {e}")
            return []
    
    def extract_python_code_comment_pairs(self, code: str) -> List[Tuple[str, str]]:
        """Extract Python functions/classes with their docstrings"""
        pairs = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                    # Get the source code of the node
                    lines = code.split('\n')
                    start_line = node.lineno - 1
                    end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 10
                    
                    # Extract function/class code without docstring
                    node_lines = lines[start_line:end_line]
                    node_code = '\n'.join(node_lines)
                    
                    # Get docstring
                    docstring = ast.get_docstring(node)
                    
                    if docstring and len(docstring.split()) > 5:
                        # Remove docstring from code
                        clean_code = self.remove_docstring_from_code(node_code, docstring)
                        if clean_code.strip():
                            pairs.append((clean_code.strip(), docstring.strip()))
                            
        except Exception as e:
            print(f"Error parsing Python code: {e}")
            
        return pairs
    
    def extract_javascript_code_comment_pairs(self, code: str) -> List[Tuple[str, str]]:
        """Extract JavaScript functions with JSDoc comments"""
        pairs = []
        
        # Simple regex-based extraction for JavaScript
        # Look for JSDoc comments followed by functions
        jsdoc_pattern = r'/\*\*(.*?)\*/'
        function_pattern = r'function\s+(\w+)\s*\([^)]*\)\s*\{[^}]*\}'
        
        jsdoc_matches = list(re.finditer(jsdoc_pattern, code, re.DOTALL))
        function_matches = list(re.finditer(function_pattern, code, re.DOTALL))
        
        for jsdoc_match in jsdoc_matches:
            jsdoc_end = jsdoc_match.end()
            comment = jsdoc_match.group(1).strip()
            
            # Find the next function after this comment
            for func_match in function_matches:
                if func_match.start() > jsdoc_end and func_match.start() - jsdoc_end < 100:
                    function_code = func_match.group(0)
                    if len(comment.split()) > 5:
                        pairs.append((function_code, comment))
                    break
                    
        return pairs
    
    def remove_docstring_from_code(self, code: str, docstring: str) -> str:
        """Remove docstring from code"""
        # Simple removal - replace the docstring with empty string
        docstring_patterns = [
            f'"""{docstring}"""',
            f"'''{docstring}'''",
            f'"{docstring}"',
            f"'{docstring}'"
        ]
        
        for pattern in docstring_patterns:
            code = code.replace(pattern, '')
            
        return code
    
    def collect_data(self, language: str, output_file: str):
        """Main data collection function"""
        print(f"Starting data collection for {language}...")
        
        repos = self.search_repositories(language)
        print(f"Found {len(repos)} repositories")
        
        all_pairs = []
        
        for i, repo in enumerate(repos):
            print(f"Processing repository {i+1}/{len(repos)}: {repo['full_name']}")
            
            files = self.get_repository_files(repo['full_name'], language)
            
            for file_path in files:
                print(f"  Processing file: {file_path}")
                content = self.get_file_content(repo['full_name'], file_path)
                
                if content:
                    if language == 'python':
                        pairs = self.extract_python_code_comment_pairs(content)
                    else:
                        pairs = self.extract_javascript_code_comment_pairs(content)
                    
                    all_pairs.extend(pairs)
                    print(f"    Found {len(pairs)} code-comment pairs")
                
                time.sleep(0.5)  # Rate limiting
            
            if len(all_pairs) > 1000:  # Limit total pairs
                break
                
        print(f"Total pairs collected: {len(all_pairs)}")
        
        # Save data
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_pairs, f, indent=2, ensure_ascii=False)
        
        print(f"Data saved to {output_file}")
        return all_pairs

def main():
    token = os.getenv('GITHUB_TOKEN')
    if not token:
        print("Please set GITHUB_TOKEN environment variable")
        return
    
    collector = GitHubCodeCollector(token)
    
    # Collect Python data
    python_pairs = collector.collect_data('python', 'data/python_code_comments.json')
    
    # Collect JavaScript data
    js_pairs = collector.collect_data('javascript', 'data/javascript_code_comments.json')
    
    # Combine all data
    all_pairs = python_pairs + js_pairs
    with open('data/all_code_comments.json', 'w', encoding='utf-8') as f:
        json.dump(all_pairs, f, indent=2, ensure_ascii=False)
    
    print(f"Total dataset size: {len(all_pairs)} code-comment pairs")

if __name__ == "__main__":
    main()