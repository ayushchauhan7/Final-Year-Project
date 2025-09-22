import requests
import os

# Test the API
def test_health():
    response = requests.get('http://localhost:5000/api/health')
    print("Health Check:", response.json())

def test_classes():
    response = requests.get('http://localhost:5000/api/classes')
    print("Classes:", response.json())

def test_prediction():
    # You'll need to provide a test image path
    test_image_path = r'c:\Users\Abhay Tyagi\OneDrive - ABES\Desktop\4th Year Project\Datasets\Testing\no_tumor\image1.jpg'
    
    if os.path.exists(test_image_path):
        with open(test_image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5000/api/predict', files=files)
            print("Prediction:", response.json())
    else:
        print("Test image not found")

if __name__ == '__main__':
    print("Testing Brain Tumor Detection API...")
    test_health()
    test_classes()
    test_prediction()