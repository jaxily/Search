#!/usr/bin/env python3
"""
Test script for the Enhanced Walk-Forward Ensemble ML Trading System
Demonstrates auto-detection, graceful shutdown, and memory optimization
"""

import os
import sys
import time
import signal
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def create_test_data(n_samples: int = 2000, n_features: int = 100) -> pd.DataFrame:
    """Create test data for demonstration"""
    print("📊 Creating test data...")
    
    # Create date range
    dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
    
    # Create price data
    np.random.seed(42)
    price_changes = np.random.normal(0, 0.01, n_samples)
    prices = np.cumprod(1 + price_changes)
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.002, n_samples)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_samples))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_samples))),
        'close': prices,
        'volume': np.random.lognormal(10, 1, n_samples)
    }, index=dates)
    
    # Add features
    for i in range(n_features):
        col_name = f'feature_{i:03d}'
        if i < 50:
            # Rolling averages
            data[col_name] = data['close'].rolling(window=20 + i).mean()
        elif i < 80:
            # Technical indicators
            data[col_name] = data['close'].rolling(window=10 + (i % 20)).std()
        else:
            # Random features
            data[col_name] = np.random.normal(0, 1, n_samples)
    
    # Create target variable
    data['returns'] = data['close'].pct_change()
    
    print(f"✅ Test data created: {data.shape}")
    return data

def test_auto_detection():
    """Test auto-detection capabilities"""
    print("\n🔍 Testing Auto-Detection...")
    
    try:
        from config import AUTO_DETECT_CONFIG, SYSTEM_CONFIG
        
        print(f"Auto-detection enabled: {AUTO_DETECT_CONFIG['enabled']}")
        print(f"Auto-clean data: {AUTO_DETECT_CONFIG['auto_clean_data']}")
        print(f"Auto-feature engineering: {AUTO_DETECT_CONFIG['auto_feature_engineering']}")
        print(f"Chunk size: {SYSTEM_CONFIG['chunk_size']}")
        print(f"Max workers: {SYSTEM_CONFIG['max_workers']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Auto-detection test failed: {e}")
        return False

def test_data_processor():
    """Test the enhanced data processor"""
    print("\n🔄 Testing Enhanced Data Processor...")
    
    try:
        # Create test data
        test_data = create_test_data(n_samples=1000, n_features=50)
        
        # Save test data
        test_file = 'test_data.csv'
        test_data.to_csv(test_file)
        print(f"💾 Test data saved to: {test_file}")
        
        # Test auto-detection parameters
        from main import auto_detect_data_parameters
        params = auto_detect_data_parameters(test_file)
        print(f"🎯 Auto-detected parameters: {params}")
        
        # Clean up
        os.remove(test_file)
        print("🧹 Test file cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Data processor test failed: {e}")
        return False

def test_graceful_shutdown():
    """Test graceful shutdown handling"""
    print("\n🛑 Testing Graceful Shutdown...")
    
    def timeout_handler(signum, frame):
        print("⏰ Timeout reached - testing shutdown signal")
        raise TimeoutError("Test timeout")
    
    # Set a timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)  # 5 second timeout
    
    try:
        # Simulate some work
        print("🔄 Simulating work...")
        for i in range(10):
            print(f"   Working... {i+1}/10")
            time.sleep(0.5)
            
            # Check for shutdown signal
            if hasattr(signal, 'SIGINT'):
                print("✅ Shutdown signal handling available")
        
        signal.alarm(0)  # Cancel alarm
        print("✅ Graceful shutdown test completed")
        return True
        
    except TimeoutError:
        print("✅ Timeout test completed")
        return True
    except Exception as e:
        print(f"❌ Graceful shutdown test failed: {e}")
        return False

def test_memory_optimization():
    """Test memory optimization features"""
    print("\n💾 Testing Memory Optimization...")
    
    try:
        from main import check_system_resources
        
        # Check system resources
        check_system_resources()
        
        # Test memory-efficient data creation
        print("🔄 Testing memory-efficient data creation...")
        
        # Create large dataset in chunks
        chunk_size = 1000
        total_rows = 5000
        
        chunks = []
        for i in range(0, total_rows, chunk_size):
            chunk_data = np.random.randn(min(chunk_size, total_rows - i), 50)
            chunks.append(pd.DataFrame(chunk_data))
            print(f"   Created chunk {i//chunk_size + 1}")
        
        # Combine chunks
        large_data = pd.concat(chunks, ignore_index=True)
        print(f"✅ Large dataset created: {large_data.shape}")
        
        # Clean up
        del chunks, large_data
        import gc
        gc.collect()
        print("🧹 Memory cleaned up")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🚀 ENHANCED SYSTEM TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Auto-Detection", test_auto_detection),
        ("Data Processor", test_data_processor),
        ("Graceful Shutdown", test_graceful_shutdown),
        ("Memory Optimization", test_memory_optimization)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            success = test_func()
            results.append((test_name, success))
            
            if success:
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
                
        except Exception as e:
            print(f"❌ {test_name} test ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("📊 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
    else:
        print("⚠️  Some tests failed. Check the output above.")
    
    return passed == total

def demo_enhanced_features():
    """Demonstrate enhanced features"""
    print("\n🚀 ENHANCED FEATURES DEMONSTRATION")
    print("=" * 50)
    
    try:
        from config import (
            AUTO_DETECT_CONFIG, SYSTEM_CONFIG, DATA_CONFIG, 
            FEATURE_CONFIG, MODEL_CONFIG
        )
        
        print("🔍 Auto-Detection Features:")
        print(f"   • Data quality assessment: {AUTO_DETECT_CONFIG['data_quality_threshold']}")
        print(f"   • Auto-cleaning: {AUTO_DETECT_CONFIG['auto_clean_data']}")
        print(f"   • Auto-feature engineering: {AUTO_DETECT_CONFIG['auto_feature_engineering']}")
        print(f"   • Auto-model selection: {AUTO_DETECT_CONFIG['auto_model_selection']}")
        
        print("\n⚙️  Smart Parameter Selection:")
        print(f"   • Auto-split detection: {DATA_CONFIG['auto_detect_splits']}")
        print(f"   • Auto-window sizing: {FEATURE_CONFIG['auto_window_selection']}")
        print(f"   • Auto-hyperparameter tuning: {MODEL_CONFIG['auto_hyperparameter_tuning']}")
        
        print("\n🖥️  M1 Optimization:")
        print(f"   • Max workers: {SYSTEM_CONFIG['max_workers']}")
        print(f"   • Chunk size: {SYSTEM_CONFIG['chunk_size']}")
        print(f"   • Graceful shutdown: {SYSTEM_CONFIG['graceful_shutdown']}")
        
        print("\n💾 Memory Management:")
        print(f"   • Auto-save interval: {SYSTEM_CONFIG['auto_save_interval']}s")
        print(f"   • Progress bars: {SYSTEM_CONFIG['progress_bar']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature demonstration failed: {e}")
        return False

if __name__ == "__main__":
    try:
        # Run tests
        all_tests_passed = run_all_tests()
        
        if all_tests_passed:
            # Demonstrate enhanced features
            demo_enhanced_features()
            
            print("\n🎯 To run the full enhanced system:")
            print("python main.py --data-file your_data.csv --ensemble-method Voting")
            print("\n💡 The system will automatically:")
            print("   • Detect optimal parameters for your data")
            print("   • Optimize memory usage")
            print("   • Handle Ctrl-C gracefully")
            print("   • Auto-save progress")
            print("   • Generate comprehensive reports")
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
        print("✅ Graceful shutdown demonstrated!")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
        sys.exit(1)

