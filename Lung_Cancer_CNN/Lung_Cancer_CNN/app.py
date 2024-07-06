from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
import keras
from keras.preprocessing import image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = set(['jpg', 'jpeg', 'png'])

# Load the pre-trained model
model_path = 'D:/my_model.h5'
model = keras.models.load_model(model_path)

# Function to check if the file type is allowed
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Route to handle the file upload
@app.route('/', methods=['GET', 'POST'])
def upload_file():
	if request.method == 'POST':
		# Check if the file is uploaded
		if 'file' not in request.files:
			return redirect(request.url)
		file = request.files['file']
		# Check if the file is allowed
		if file and allowed_file(file.filename):
			# Save the file to the uploads folder
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			# Make prediction
			img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
			img = image.load_img(img_path, target_size=(200, 200))
			x = image.img_to_array(img)
			x = np.expand_dims(x, axis=0)
			images = np.vstack([x])
			y_predict = model.predict(images, batch_size=10)
			prediction = label[np.argmax(y_predict)]
			return render_template('index.html', prediction=prediction, file_name=filename)
	return render_template('index.html')

# Route to handle the uploaded file
@app.route('/uploads/<filename>')
def uploaded_file(filename):    
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
	app.run(debug=True)