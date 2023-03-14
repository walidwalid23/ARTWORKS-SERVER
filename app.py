import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from torchvision import transforms
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import os
from PIL import Image
from urllib.request import urlopen
import json
import requests
# %matplotlib inline
import requests
import json
from flask_mail import Mail, Message
from dotenv import load_dotenv
import os
import threading

global embed

load_dotenv()

# uploaded logos folder
UPLOAD_FOLDER = 'uploads'
# allowed logos images extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# preparing the server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# THE DATABASE NAME MUST BE ADDED IN THE URI BEFORE THE ? OR THE DB OBJECT WILL BE NONE
# app.config['MONGO_URI'] = "mongodb+srv://walidwalid:"+os.getenv(
#    "DB_PASS")+"@cluster0.dwtedah.mongodb.net/Artworks_Features_DB?retryWrites=true&w=majority"

# mongo_client = PyMongo(app)
# extractedFeaturesCollec = mongo_client.db.extracted_features_col
# configuration of mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'stylebustersinc@gmail.com'
app.config['MAIL_PASSWORD'] = 'temlvwashllnqpbh'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True
mail = Mail(app)
# Instantiate a VGG model with our saved weights
vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)  # B
vgg_model.to(torch.device("meta"))
# load the model using your available device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.classifier[-1] = nn.Linear(4096, 23)
vgg_model.load_state_dict(torch.load('feature_extractor/vgg11_10_wikiart.pt',
                                     map_location=torch.device("cpu")))


@app.route('/sendArtwork', methods=['POST'])
def postRoute():
    print("in route")
    userEmail = request.form['email']
    image = request.files['image']
    artistNationality = request.form['artistNationality']

    if image and allowed_file(image.filename):
        imageName = secure_filename(image.filename)
        # store the image in the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
        # prepare a success response since we received the image from the user
        successResponse = jsonify(
            {"successMessage": "You will receive an email with the results shortly"})
    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})

    # run the below code after the Response is sent to user
    # @successResponse.call_on_close
    def on_close():
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"error": "no selected image"})

        if image and allowed_file(image.filename):
            imageName = secure_filename(image.filename)
            resultsFound = False
            # get image path
            artworkImgPath = 'uploads/'+imageName
            # Load The Input Image
            inputImage = Image.open(artworkImgPath)
            # getting feature vector of the main artwork image
            _transforms = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    # standardization of pixel colors
                    transforms.Normalize([0.485, 0.456, 0.406], [
                                         0.229, 0.224, 0.225])
                ])

            input_image_matrix = np.asarray(
                np.expand_dims(_transforms(inputImage), 0))
            input_image_vector = vgg_model.features(
                torch.tensor(input_image_matrix)).mean((2, 3))

            # getting feature vectors of the retrieved artworks images
            URL = "https://artworks-web-scraper.onrender.com/WalidArtworksApi?artistNationality=" + artistNationality
            headers = {
                'Content-Type': 'application/json'
            }
            # send a request to get stream of artworks json objects
            resp = requests.request(
                "GET", URL, headers=headers, stream=True)
            # print(resp.headers['content-type'])
            # print(resp.encoding)
            # we iterate by lines since we added new line after each response from server side
            for line in resp.iter_lines():
                if line:
                    # the remote hosts encodes chunks using utf-8 but localhost doesn't they use (https*)
                    decoded_chunk = line.decode('utf-8')
                # converting json to dict
                    decodedArtworkObj = json.loads(decoded_chunk)

                    retrievedArtworkDetails = decodedArtworkObj["artworkDetails"]
                    artworkImageUrl = decodedArtworkObj["artworkImageUrl"]
                    # extract features from each retrieved artwork image
                    retrievedImage = Image.open(urlopen(artworkImageUrl))

                    retrieved_image_matrix = np.asarray(
                        np.expand_dims(_transforms(retrievedImage), 0))
                    retrieved_image_vector = vgg_model.features(
                        torch.tensor(retrieved_image_matrix)).mean((2, 3))

                    # get the cosine similarity
                    cosine_similarity = torch.cosine_similarity(
                        input_image_vector, retrieved_image_vector)

                    print("Cosine Similarity of The Main Image and Image:" +
                          retrievedArtworkDetails + " is: " + str(cosine_similarity)+"%")
                    if cosine_similarity > 0.75:
                        print("MATCH")
                        resultsFound = True
                        # send email including the details of the matched logo
                        msg = Message('Found A Match!',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'We found a matched Artwork! \n Artwork Details: ' + \
                            retrievedArtworkDetails+'\n Artwork Image URL Is: '+artworkImageUrl + \
                            '\n Similarity Percentage is: ' + cosine_similarity
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)
            if resultsFound == False:
                # Send an email telling the user that no results were found
                msg = Message('No Results Found',
                              sender='stylebustersinc@gmail.com',
                              recipients=[userEmail]
                              )
                msg.body = 'No Matching Artworks Have Been Found'
                # send the email to the user (you must put the mail.send inside the app context)
                with app.app_context():
                    mail.send(msg)

    thread = threading.Thread(target=on_close)
    thread.start()
    return successResponse


# check if this file is being excuted from itself or being excuted by being imported as a module
if __name__ == "__main__":
    from waitress import serve
    print("server is running at port "+str(os.getenv("PORT")))
    serve(app, host="0.0.0.0", port=os.getenv("PORT"))
