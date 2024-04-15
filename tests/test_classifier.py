import joblib
import pandas as pd

def classifier(sepalLength, sepalWidth, petalLength, petalWidth):
    loaded_model = joblib.load("My_KNN_Model.model")
    # sepalLength = request.args['sepalLength']
    # sepalWidth = request.args['sepalWidth']
    # petalLength = request.args['petalLength']
    # petalWidth = request.args['petalWidth']

    data = { 'SepalLengthCm' : [sepalLength], 
            'SepalWidthCm': [sepalWidth], 
            'PetalLengthCm': [petalLength], 
            'PetalWidthCm': [petalWidth]
    }
    sample = pd.DataFrame(data, columns=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'])

    class_label = loaded_model.predict(sample)
    print(class_label)
    return {
        'code': 200,
        'message': f'Sample is classified as {class_label[0]}',
        'class': class_label[0]
    }

def test_classifier():
    assert classifier(1.5, 2.5, 3.5, 4.5) == {
        'code': 200,
        'message': 'Sample is classified as Iris-versicolor',
        'class': 'Iris-versicolor'
    }