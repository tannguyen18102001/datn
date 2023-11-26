from flask import Flask, render_template, request, session
import os
from werkzeug.utils import secure_filename
import glob
import cv2
import numpy as np
from skimage.util.shape import view_as_blocks
from keras.models import load_model
from keras import backend as K
from PIL import Image
import re


UPLOAD_FOLDER = os.path.join('static', 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pgm'}

# The default folder name for static files should be "static"
app = Flask(__name__, template_folder='templates', static_folder='static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'You can write anything, is just a test'

def load_images(path_pattern):
    files=glob.glob(path_pattern)
    X=[]
    for f in sorted(files):
        I = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        patches = view_as_blocks(I, (256, 256))
        for i in range(patches.shape[0]):
            for j in range(patches.shape[1]):
                X.append( [ patches[i,j] ] )
    X=np.array(X)
    return X


def remove_pgm_extension(string):
    return re.sub(r"\.pgm$", "", string)

def Tanh3(x):
    tanh3 = K.tanh(x)*3
    return tanh3

def convert_pgm_to_jpg(pgm_filename, jpg_filename):
    # Read the .pgm file
    with Image.open(pgm_filename) as image:
        # Convert the image to RGB format
        image = image.convert('RGB')
        # Save the image as a .jpg file
        image.save(jpg_filename)

def convert_jpg_to_pgm(jpg_filename, pgm_filename):
    # Open the JPEG image
    with Image.open(jpg_filename) as image:
        # Convert the image to grayscale
        gray_image = image.convert('L')
        # Save the grayscale image as a PGM file
        gray_image.save(pgm_filename)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/', methods=("POST", "GET"))
def upload_file():
    if request.method == 'POST':
        uploaded_img = request.files['uploaded-file']
        img_filename = secure_filename(uploaded_img.filename)
        # Upload file to database (defined uploaded folder in static path)
        uploaded_img.save(os.path.join(app.config['UPLOAD_FOLDER'], img_filename))
        # Storing uploaded file path in flask session
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], img_filename)
        session['img_file_name'] = (img_filename)

        img_file_path = session.get('uploaded_img_file_path', None)
        image_file_name = session.get('img_file_name', None)
        jpg_file = remove_pgm_extension(image_file_name)

        input = load_images(img_file_path)
        
        in_test = np.rollaxis(input,1,4)
        

        path_model = 'model\model.hdf5'
        model = load_model(path_model,custom_objects={'Tanh3':Tanh3}, compile=True)
        predict = model.predict(in_test)
        result  = np.argmax(np.array(predict))
        print(result)
        if result ==0:
            output_jpg ='static\jpg\cover\\'
            label= "Cover"
        else :
            label = "UERD"
            output_jpg =r'static\jpg\uerd\\'
        convert_pgm_to_jpg(img_file_path,output_jpg +jpg_file+'.jpg')
        return render_template('uploaded_image.html', user_image=output_jpg+jpg_file+'.jpg', result=label)
        #return render_template('uploaded_image.html')

@app.route('/predict')
def display_image():
    # Retrieving uploaded file
    img_file_path = session.get('uploaded_img_file_path', None)
    image_file_name = session.get('img_file_name', None)
    jpg_file = remove_pgm_extension(image_file_name)

    input = load_images(img_file_path)
    
    in_test = np.rollaxis(input,1,4)
    

    path_model = 'model\model.hdf5'
    model = load_model(path_model,custom_objects={'Tanh3':Tanh3}, compile=True)
    predict = model.predict(in_test)
    result  = np.argmax(np.array(predict))
    print(result)
    if result ==0:
        output_jpg ='static\jpg\cover\\'
        label= "Cover"
    else :
        label = "UERD"
        output_jpg =r'static\jpg\uerd\\'
    convert_pgm_to_jpg(img_file_path,output_jpg +jpg_file+'.jpg')
    return render_template('show_image.html', user_image=output_jpg+jpg_file+'.jpg', result=label)

if __name__ == '__main__':
    app.run(debug=True)