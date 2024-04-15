import joblib
import pandas as pd

def classify():
    loaded_model = joblib.load("My_KNN_Model.model")
    # 2. Classify the sample with the trained model
    data = { 'SepalLengthCm' : [5.8], 
            'SepalWidthCm': [2.8], 
            'PetalLengthCm': [5.1], 
            'PetalWidthCm': [2.4]
    }
    sample = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    class_label = loaded_model.predict(sample)
    return {
        'code': 200,
        'message': f'Sample is classified as {class_label[0]}',
        'class': class_label[0]
    }

def test_classify():
    assert classify() == {
        'code': 200,
        'message': f'Sample is classified as Iris-virginica',
        'class': 'Iris-virginica'
    }