import os
import requests
import json

def test_image_accuracy():
    """Simple script to test model accuracy with known images"""
    
    # Define test images with expected results
    test_cases = [
        {
            'path': r'C:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\4th Year Project\Datasets\Testing\glioma_tumor\image(1).jpg',
            'expected': 'glioma_tumor'
        },
        {
            'path': r'C:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\4th Year Project\Datasets\Testing\meningioma_tumor\image(1).jpg',
            'expected': 'meningioma_tumor'
        },
        {
            'path': r'C:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\4th Year Project\Datasets\Testing\no_tumor\image(1).jpg',
            'expected': 'no_tumor'
        },
        {
            'path': r'C:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\4th Year Project\Datasets\Testing\pituitary_tumor\image(1).jpg',
            'expected': 'pituitary_tumor'
        }
    ]
    
    url = 'http://localhost:5000/api/debug/prediction'
    results = []
    
    print("Testing Model Accuracy...")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases):
        if os.path.exists(test_case['path']):
            try:
                with open(test_case['path'], 'rb') as f:
                    files = {'image': f}
                    response = requests.post(url, files=files)
                    
                if response.status_code == 200:
                    data = response.json()
                    predicted = data['predicted_class']
                    confidence = data['confidence']
                    
                    is_correct = predicted == test_case['expected']
                    
                    print(f"Test {i+1}: {os.path.basename(test_case['path'])}")
                    print(f"Expected: {test_case['expected']}")
                    print(f"Predicted: {predicted}")
                    print(f"Confidence: {confidence:.2%}")
                    print(f"Correct: {'✓' if is_correct else '✗'}")
                    print("-" * 30)
                    
                    results.append({
                        'expected': test_case['expected'],
                        'predicted': predicted,
                        'confidence': confidence,
                        'correct': is_correct
                    })
                else:
                    print(f"Error testing {test_case['path']}: {response.status_code}")
                    
            except Exception as e:
                print(f"Error with {test_case['path']}: {e}")
        else:
            print(f"File not found: {test_case['path']}")
    
    # Calculate accuracy
    if results:
        correct_count = sum(1 for r in results if r['correct'])
        accuracy = correct_count / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        
        print("=" * 50)
        print("SUMMARY:")
        print(f"Total Tests: {len(results)}")
        print(f"Correct: {correct_count}")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"Average Confidence: {avg_confidence:.2%}")
        print("=" * 50)
        
        if accuracy < 0.5:
            print("❌ Poor accuracy - check class mapping or retrain model")
        elif accuracy < 0.75:
            print("⚠️ Moderate accuracy - consider model improvements")
        else:
            print("✅ Good accuracy")

if __name__ == "__main__":
    test_image_accuracy()