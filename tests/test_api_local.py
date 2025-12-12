"""
Quick test script for the Fraud Detection API.
Run this after starting the API server to verify everything works.
"""

import requests
import json
from datetime import datetime

# API base URL
BASE_URL = "http://localhost:8000"

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_health_check():
    """Test the health check endpoint."""
    print_section("1. Health Check")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_root():
    """Test the root endpoint."""
    print_section("2. Root Endpoint")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_model_info():
    """Test the model info endpoint."""
    print_section("3. Model Info")
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_prediction():
    """Test the prediction endpoint."""
    print_section("4. Single Prediction")
    
    # Sample transaction (likely legitimate)
    transaction = {
        "Time": 0.0,
        "V1": -1.359807,
        "V2": -0.072781,
        "V3": 2.536347,
        "V4": 1.378155,
        "V5": -0.338321,
        "V6": 0.462388,
        "V7": 0.239599,
        "V8": 0.098698,
        "V9": 0.363787,
        "V10": 0.090794,
        "V11": -0.551600,
        "V12": -0.617801,
        "V13": -0.991390,
        "V14": -0.311169,
        "V15": 1.468177,
        "V16": -0.470401,
        "V17": 0.207971,
        "V18": 0.025791,
        "V19": 0.403993,
        "V20": 0.251412,
        "V21": -0.018307,
        "V22": 0.277838,
        "V23": -0.110474,
        "V24": 0.066928,
        "V25": 0.128539,
        "V26": -0.189115,
        "V27": 0.133558,
        "V28": -0.021053,
        "Amount": 149.62
    }
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=transaction
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n📊 Prediction Result:")
            print(f"   Is Fraud: {result['is_fraud']}")
            print(f"   Fraud Probability: {result['fraud_probability']:.4f}")
            print(f"   Confidence: {result['confidence']:.4f}")
            print(f"   Risk Level: {result['risk_level']}")
            return True
        else:
            print(f"Response: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_prediction():
    """Test the batch prediction endpoint."""
    print_section("5. Batch Prediction")
    
    # Sample transactions
    transactions = [
        {
            "Time": 0.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
            "V4": 1.378155, "V5": -0.338321, "V6": 0.462388, "V7": 0.239599,
            "V8": 0.098698, "V9": 0.363787, "V10": 0.090794, "V11": -0.551600,
            "V12": -0.617801, "V13": -0.991390, "V14": -0.311169, "V15": 1.468177,
            "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
            "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474,
            "V24": 0.066928, "V25": 0.128539, "V26": -0.189115, "V27": 0.133558,
            "V28": -0.021053, "Amount": 149.62
        },
        {
            "Time": 1.0, "V1": 1.191857, "V2": 0.266151, "V3": 0.166480,
            "V4": 0.448154, "V5": 0.060018, "V6": -0.082361, "V7": -0.078803,
            "V8": 0.085102, "V9": -0.255425, "V10": -0.166974, "V11": 1.612727,
            "V12": 1.065235, "V13": 0.489095, "V14": -0.143772, "V15": 0.635558,
            "V16": 0.463917, "V17": -0.114805, "V18": -0.183361, "V19": -0.145783,
            "V20": -0.069083, "V21": -0.225775, "V22": -0.638672, "V23": 0.101288,
            "V24": -0.339846, "V25": 0.167170, "V26": 0.125895, "V27": -0.008983,
            "V28": 0.014724, "Amount": 2.69
        }
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json={"transactions": transactions}
        )
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n📊 Batch Prediction Result:")
            print(f"   Total Transactions: {result['total_transactions']}")
            print(f"   Fraudulent: {result['fraud_count']}")
            print(f"   Legitimate: {result['total_transactions'] - result['fraud_count']}")
            print("\n   Individual Results:")
            for i, pred in enumerate(result['predictions'], 1):
                print(f"   Transaction {i}: {'🚨 FRAUD' if pred['is_fraud'] else '✅ LEGITIMATE'} "
                      f"(prob: {pred['fraud_probability']:.4f})")
            return True
        else:
            print(f"Response: {response.json()}")
            return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "🔍 " + "="*58)
    print("  FRAUD DETECTION API - TEST SUITE")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = {
        "Health Check": test_health_check(),
        "Root Endpoint": test_root(),
        "Model Info": test_model_info(),
        "Single Prediction": test_prediction(),
        "Batch Prediction": test_batch_prediction()
    }
    
    # Summary
    print_section("TEST SUMMARY")
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {test_name:.<50} {status}")
    
    print(f"\n  Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! API is working correctly.")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed. Check the output above.")
    
    print("="*60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        exit(1)
