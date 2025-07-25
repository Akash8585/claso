#!/usr/bin/env python3
"""
Test script for the enhanced API with fast models
Run this after training completes to test the new models
"""

import requests
import json
import time

API_BASE = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    if response.status_code == 200:
        data = response.json()
        print("✅ Health check passed")
        print(f"   Comment model loaded: {data['comment_model_loaded']}")
        print(f"   Commit model loaded: {data['commit_model_loaded']}")
        return True
    else:
        print(f"❌ Health check failed: {response.status_code}")
        return False

def test_model_info():
    """Test model info endpoint"""
    print("\n📊 Testing model info endpoint...")
    response = requests.get(f"{API_BASE}/model-info")
    if response.status_code == 200:
        data = response.json()
        print("✅ Model info retrieved")
        
        if data['models']['comment']['loaded']:
            comment_model = data['models']['comment']
            print(f"   📝 Comment Model: {comment_model['parameters']:,} parameters")
            print(f"      Architecture: {comment_model['d_model']}d, {comment_model['encoder_layers']}+{comment_model['decoder_layers']} layers")
        
        if data['models']['commit']['loaded']:
            commit_model = data['models']['commit']
            print(f"   💬 Commit Model: {commit_model['parameters']:,} parameters")
            print(f"      Architecture: {commit_model['d_model']}d, {commit_model['encoder_layers']}+{commit_model['decoder_layers']} layers")
        
        print(f"   💻 Device: {data['system']['device']}")
        return True
    else:
        print(f"❌ Model info failed: {response.status_code}")
        return False

def test_comment_generation():
    """Test comment generation"""
    print("\n📝 Testing comment generation...")
    
    test_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
"""
    
    payload = {
        "code": test_code.strip(),
        "language": "python",
        "style": "docstring"
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate-comments", json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Comment generation successful")
        print(f"   ⏱️  Processing time: {data['processing_time']:.3f}s")
        print(f"   🎯 Confidence: {data['confidence']:.2f}")
        print(f"   📊 Generated {len(data['comments'])} comments:")
        
        for i, comment in enumerate(data['comments'], 1):
            print(f"\n   Comment {i}:")
            print(f"   {comment}")
        
        return True
    else:
        print(f"❌ Comment generation failed: {response.status_code}")
        if response.text:
            print(f"   Error: {response.text}")
        return False

def test_commit_generation():
    """Test commit message generation"""
    print("\n💬 Testing commit message generation...")
    
    test_diff = """
+def validate_email(email):
+    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
+    return re.match(pattern, email) is not None
+
+def process_user_data(data):
+    if validate_email(data.get('email')):
+        return {'status': 'valid', 'data': data}
+    return {'status': 'invalid', 'error': 'Invalid email'}
"""
    
    payload = {
        "diff": test_diff.strip()
    }
    
    start_time = time.time()
    response = requests.post(f"{API_BASE}/generate-commit-msg", json=payload)
    end_time = time.time()
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Commit message generation successful")
        print(f"   🎯 Confidence: {data['confidence']:.2f}")
        print(f"   💬 Generated message:")
        print(f"   \"{data['message']}\"")
        return True
    else:
        print(f"❌ Commit generation failed: {response.status_code}")
        if response.text:
            print(f"   Error: {response.text}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Enhanced Claso API")
    print("=" * 50)
    
    # Test health first
    if not test_health():
        print("\n❌ API is not healthy. Make sure the server is running:")
        print("   python main.py")
        return
    
    # Test model info
    test_model_info()
    
    # Test comment generation
    test_comment_generation()
    
    # Test commit generation
    test_commit_generation()
    
    print("\n🎉 All tests completed!")
    print("\n📋 Next steps:")
    print("1. Frontend should now work with enhanced models")
    print("2. Try the web interface at http://localhost:5173")
    print("3. API documentation at http://localhost:8000/docs")

if __name__ == "__main__":
    main()