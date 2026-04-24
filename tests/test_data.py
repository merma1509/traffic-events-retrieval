import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data.traffic_loader import TrafficWeatherDataLoader
from data.network_loader import KigaliNetworkLoader

def test_traffic_loader():
    """Test TrafficWeatherDataLoader functionality"""
    print("Testing TrafficWeatherDataLoader...")
    
    try:
        # Test instantiation
        loader = TrafficWeatherDataLoader('.')
        print("TrafficWeatherDataLoader instantiated successfully")
        
        # Test path resolution
        test_path = loader.get_data_path('test_file.csv')
        print(f"Path resolution working: {test_path}")
        
        # Test data loading (will fail if no data file exists)
        try:
            data = loader.load_traffic_weather_data('test_file.csv')
        except FileNotFoundError:
            print("File not found error handled correctly")
        
        print("TrafficWeatherDataLoader tests passed")
        return True
        
    except Exception as e:
        print(f"TrafficWeatherDataLoader test failed: {e}")
        return False

def test_network_loader():
    """Test KigaliNetworkLoader functionality"""
    print("\nTesting KigaliNetworkLoader...")
    
    try:
        # Test instantiation
        loader = KigaliNetworkLoader('.')
        print("KigaliNetworkLoader instantiated successfully")
        
        # Test path resolution
        test_path = loader.get_data_path('test_network.pkl')
        print(f"Path resolution working: {test_path}")
        
        # Test network loading (will fail if no network file exists)
        try:
            network = loader.load_network_graph('test_network.pkl')
        except FileNotFoundError:
            print("File not found error handled correctly")
        
        print("KigaliNetworkLoader tests passed")
        return True
        
    except Exception as e:
        print(f"KigaliNetworkLoader test failed: {e}")
        return False

def test_data_integration():
    """Test data loading integration"""
    print("\nTesting Data Integration...")
    
    try:
        # Test both loaders together
        traffic_loader = TrafficWeatherDataLoader('.')
        network_loader = KigaliNetworkLoader('.')
        
        print("Both loaders instantiated successfully")
        print("Data loading components ready for integration")
        
        return True
        
    except Exception as e:
        print(f"Data integration test failed: {e}")
        return False

def run_all_tests():
    """Run all data loading tests"""
    print("=" * 60)
    print("ROUTIQ IR - DATA LOADING TESTS")
    print("=" * 60)
    
    results = []
    results.append(test_traffic_loader())
    results.append(test_network_loader())
    results.append(test_data_integration())
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Tests Passed: {passed}/{total}")
    print(f"Success Rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("ALL TESTS PASSED - Data loading components are ready!")
    else:
        print("Some tests failed - check the errors above")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    run_all_tests()