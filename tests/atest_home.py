from flask import Flask, jsonify, request, render_template

def home():
    return render_template("index.html")

def test_home():
    assert home() == render_template("index.html")
#     assert home() == """<html>
#     <body>
#         <h1>IRIS Classifier</h1>
#         <h3>Please provide values for prediction</h3>

#         <form action='/final_classifier' method='POST' enctype="multipart/form-data">
#             <label>Sepal Length</label><input type='text' name='sepalLength'>
#             <label>Sepal Width</label><input type='text' name='sepalWidth'>
#             <label>Petal Length</label><input type='text' name='petalLength'>
#             <label>Petal Width</label><input type='text' name='petalWidth'>

#             <input type="submit" name='classify' value='Classify'>

#         </form>

#     </body>
# </html>"""