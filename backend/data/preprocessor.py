import ast
import esprima  # for JavaScript parsing
from typing import List, Tuple

class CodeCommentExtractor:
    def extract_python_pairs(self, code: str) -> List[Tuple[str, str]]:
        tree = ast.parse(code)
        pairs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                code_snippet = ast.get_source_segment(code, node)
                docstring = ast.get_docstring(node)
                if docstring and len(docstring.split()) > 3:
                    clean_code = self.remove_docstring(code_snippet)
                    pairs.append((clean_code, docstring))
        return pairs
    def extract_javascript_pairs(self, code: str) -> List[Tuple[str, str]]:
        # Use esprima to parse JavaScript and extract comment pairs
        pass

if __name__ == "__main__":
    import os
    import pickle
    extractor = CodeCommentExtractor()
    # Example: load raw code from a file
    # with open('data/raw_code.py', 'r') as f:
    #     code = f.read()
    # pairs = extractor.extract_python_pairs(code)
    # with open('data/code_comment_pairs.pkl', 'wb') as f:
    #     pickle.dump(pairs, f) 