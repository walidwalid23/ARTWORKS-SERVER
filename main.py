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
import json
from flask_mail import Mail, Message
from dotenv import load_dotenv
import os
import threading
import time
from io import BytesIO

global embed

load_dotenv()

# uploaded artworks folder
UPLOAD_FOLDER = 'uploads'
# allowed artworks images extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# preparing the server
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# configuration of mail
app.config['MAIL_SERVER'] = 'smtp-relay.sendinblue.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USERNAME'] = 'walid.tawfik2000@hotmail.com'
app.config['MAIL_PASSWORD'] = 'cNxhb5W6EpAIXDUj'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = False
mail = Mail(app)

# Instantiate a VGG model with our saved weights
vgg_model = models.vgg11(weights=models.VGG11_Weights.IMAGENET1K_V1)  # B
# load the model using your available device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vgg_model.classifier[-1] = nn.Linear(4096, 23)
vgg_model.load_state_dict(torch.load('feature_extractor/vgg11_10_wikiart.pt',
                                     map_location=torch.device(device)))
vgg_model.eval()


@app.route('/', methods=['GET'])
def root():
    return jsonify(
        {"you are at root": "This is the root of stylebusters artworks app"})


@app.route('/get-artworks', methods=['POST'])
def getStolenArtworks():
    print("in route")
    image = request.files['image']
    userEmail = request.form['email']
    artistNationality = request.form['artistNationality'] if 'artistNationality' in request.form and request.form['artistNationality'] != None else None
    material = request.form['material'] if 'material' in request.form and request.form['material'] != None else None
    timePeriod = request.form['timePeriod'] if 'timePeriod' in request.form and request.form['timePeriod'] != None else None

    if image and allowed_file(image.filename):
        imageName = secure_filename(image.filename)
        # store the image in the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
        # prepare a success response since we received the image from the user
        successResponse = jsonify(
            {"successMessage": "You Will Receive An Email With The Results Shortly"})
    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})

    # run the below code after the Response is sent to user
    # @successResponse.call_on_close
    def on_close():
        # using session:if several requests are being made to the same host, the underlying TCP connection will be reused (keep-alive)
        session = requests.Session()
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"errorMessage": "no selected image"})

        if image and allowed_file(image.filename):
            imageName = secure_filename(image.filename)
            resultsFound = False
            # get image path
            artworkImgPath = 'uploads/'+imageName
            # Load The Input Image
            inputImage = Image.open(artworkImgPath).convert('RGB')
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
            queries = ''
            queriesCount = 0
            if artistNationality != None:
                queries += 'artistNationality=' + artistNationality
                queriesCount += 1

            if material != None:
                if queriesCount > 0:
                    queries += "&"

                queries += 'materials_terms=' + material
                queriesCount += 1

            if timePeriod != None:
                if queriesCount > 0:
                    queries += "&"

                queries += 'major_periods=' + timePeriod
                queriesCount += 1

            URL = "https://artworks-web-scraping-production.up.railway.app/WalidArtworksApi?"+queries
            print(URL)
            headers = {
                'Content-Type': 'application/json',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',

            }
            # send a request to get stream of artworks json objects
            # sleep to make sure the main thread will finish first before requesting
            time.sleep(0.01)
            resp = session.get(URL, headers=headers, stream=True)
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
                    lastArtwork = decodedArtworkObj["lastArtwork"]
                # request the image with headers
                    time.sleep(0.5)
                    session2 = requests.Session()
                    resp = session2.get(artworkImageUrl,
                                        headers={
                                            "accept": "*/*",
                                            "content-type": "application/json",
                                            "dnt": "1",
                                            "origin": "https://www.artsy.net",
                                            "user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                                            'Connection': 'keep-alive',
                                            "sec-ch-ua": '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
                                            "accept-language": "en-US,en;q=0.9"

                                        })
                    # extract features from each retrieved artwork image
                    retrievedImage = Image.open(
                        BytesIO(resp.content)).convert('RGB')

                    retrieved_image_matrix = np.asarray(
                        np.expand_dims(_transforms(retrievedImage), 0))
                    retrieved_image_vector = vgg_model.features(
                        torch.tensor(retrieved_image_matrix)).mean((2, 3))

                    # get the cosine similarity
                    cosine_similarity = torch.cosine_similarity(
                        input_image_vector, retrieved_image_vector)
                    string_cosine_similarity = str(
                        cosine_similarity)[8:12]+" %"
                    print("Cosine Similarity of The Main Image and Image:" +
                          retrievedArtworkDetails + " is: " + string_cosine_similarity+" ")
                    if cosine_similarity >= 0.75:
                        print("MATCH")
                        resultsFound = True
                        # send email including the details of the matched logo
                        msg = Message('Found A Matched Artwork!',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'We found a matched Artwork! \n Artwork Details: ' + \
                            retrievedArtworkDetails+'\n Artwork Image URL Is: '+artworkImageUrl + \
                            '\n Similarity Percentage is: ' + string_cosine_similarity
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)
                    # close connection after the last image (can't be closed after the for loop due to a bug)
                    print("last artwork: " + str(lastArtwork))
                    if (retrievedArtworkDetails == lastArtwork):
                        print("in last artwork")
                        msg = Message('Search For Artworks Has Finished',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'The Search For Artworks Has Finished'
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)

                        resp.close()
                        break

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


# GET INSPIRED ARTWORKS ROUTE
@app.route('/get-artworks-inspiration', methods=['POST'])
def getInspiredArtworks():
    print("in route")
    image = request.files['image']
    userEmail = request.form['email']
    artistNationality = request.form['artistNationality'] if 'artistNationality' in request.form and request.form['artistNationality'] != None else None
    material = request.form['material'] if 'material' in request.form and request.form['material'] != None else None
    timePeriod = request.form['timePeriod'] if 'timePeriod' in request.form and request.form['timePeriod'] != None else None

    if image and allowed_file(image.filename):
        imageName = secure_filename(image.filename)
        # store the image in the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
        # prepare a success response since we received the image from the user
        successResponse = jsonify(
            {"successMessage": "You Will Receive An Email With The Results Shortly"})
    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})

    # run the below code after the Response is sent to user
    # @successResponse.call_on_close
    def on_close():
        # using session:if several requests are being made to the same host, the underlying TCP connection will be reused (keep-alive)
        session = requests.Session()
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"errorMessage": "no selected image"})

        if image and allowed_file(image.filename):
            imageName = secure_filename(image.filename)
            resultsFound = False
            # get image path
            artworkImgPath = 'uploads/'+imageName
            # Load The Input Image
            inputImage = Image.open(artworkImgPath).convert('RGB')
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
            queries = ''
            queriesCount = 0
            if artistNationality != None:
                queries += 'artistNationality=' + artistNationality
                queriesCount += 1

            if material != None:
                if queriesCount > 0:
                    queries += "&"

                queries += 'materials_terms=' + material
                queriesCount += 1

            if timePeriod != None:
                if queriesCount > 0:
                    queries += "&"

                queries += 'major_periods=' + timePeriod
                queriesCount += 1

            URL = "https://artworks-web-scraping-production.up.railway.app/WalidArtworksApi?"+queries
            print(URL)
            headers = {
                'Content-Type': 'application/json',
                'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',

            }
            # send a request to get stream of artworks json objects
            # sleep to make sure the main thread will finish first before requesting
            time.sleep(0.01)
            resp = session.get(URL, headers=headers, stream=True)
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
                    lastArtwork = decodedArtworkObj["lastArtwork"]
                # request the image with headers
                    time.sleep(0.5)
                    session2 = requests.Session()
                    resp = session2.get(artworkImageUrl,
                                        headers={
                                            "accept": "*/*",
                                            "content-type": "application/json",
                                            "dnt": "1",
                                            "origin": "https://www.artsy.net",
                                            "user-agent": 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.0.0 Safari/537.36',
                                            'Connection': 'keep-alive',
                                            "sec-ch-ua": '"Chromium";v="112", "Google Chrome";v="112", "Not:A-Brand";v="99"',
                                            "accept-language": "en-US,en;q=0.9"

                                        })
                    # extract features from each retrieved artwork image
                    retrievedImage = Image.open(
                        BytesIO(resp.content)).convert('RGB')

                    retrieved_image_matrix = np.asarray(
                        np.expand_dims(_transforms(retrievedImage), 0))
                    retrieved_image_vector = vgg_model.features(
                        torch.tensor(retrieved_image_matrix)).mean((2, 3))

                    # get the cosine similarity
                    cosine_similarity = torch.cosine_similarity(
                        input_image_vector, retrieved_image_vector)
                    string_cosine_similarity = str(
                        cosine_similarity)[8:12]+" %"
                    print("Cosine Similarity of The Main Image and Image:" +
                          retrievedArtworkDetails + " is: " + string_cosine_similarity+" ")
                    if cosine_similarity >= 0.70:
                        print("MATCH")
                        resultsFound = True
                        # send email including the details of the matched logo
                        msg = Message('Found A Matched Artwork!',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'We found a matched Artwork! \n Artwork Details: ' + \
                            retrievedArtworkDetails+'\n Artwork Image URL Is: '+artworkImageUrl + \
                            '\n Similarity Percentage is: ' + string_cosine_similarity
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)
                    # close connection after the last image (can't be closed after the for loop due to a bug)
                    print("last artwork: " + str(lastArtwork))
                    if (retrievedArtworkDetails == lastArtwork):
                        print("in last artwork")
                        msg = Message('Search For Artworks Has Finished',
                                      sender='stylebustersinc@gmail.com',
                                      recipients=[userEmail]
                                      )
                        msg.body = 'The Search For Artworks Has Finished'
                        # send the email to the user (you must put the mail.send inside the app context)
                        with app.app_context():
                            mail.send(msg)

                        resp.close()
                        break

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

# GET ALL ARTWORKS ROUTE


@app.route('/get-all-artworks', methods=['POST'])
def getAllArtworks():
    print("in get all artworks route")
    userEmail = request.form['email']
    image = request.files['image']
    artistsNationalities = ['American', 'Angolan', 'Argentine', 'Armenian', 'Australian', 'Austrian', 'Belarusian', 'Belgian', 'Beninese', 'Brazilian', 'British', 'Bulgarian', 'Burmese',
                            'Cameroonian', 'Canadian', 'Catalan', 'Chilean', 'Chinese', 'Chinese-American', 'Colombian',
                            'Congolese', 'Croatian', 'Cuban', 'Czech', 'Danish', 'Dominican', 'Dutch', 'Ecuadorian',
                            'Egyptian', 'English', 'Finnish', 'French', 'French-Canadian', 'Georgian', 'German', 'Ghanaian',
                            'Greek', 'Haitian', 'Hong Kong', 'Hungarian', 'Icelandic', 'Indian', 'Indonesian', 'Iranian', 'Iraqi',
                            'Irish', 'Israeli', 'Italian', 'Ivorian', 'Jamaican', 'Japanese', 'Japanese-American', 'Jordanian', 'Kenyan',
                            'Korean', 'Latvian', 'Lebanese', 'Lithuanian', 'Malaysian', 'Mexican', 'Moroccan', 'Mozambican', 'Netherlandish',
                            'New Zealand', 'Nigerian', 'Norwegian', 'Pakistani', 'Palestinian', 'Peruvian', 'Philippine', 'Polish', 'Portuguese',
                            'Puerto Rican', 'Romanian', 'Russian', 'Russian-American', 'Scottish', 'Senegalese', 'Serbian', 'Singaporean', 'Slovak',
                            'Slovene', 'South African', 'South Korean', 'Spanish', 'Sudanese', 'Swedish', 'Swiss', 'Syrian', 'Taiwanese', 'Thai',
                            'Turkish', 'Ugandan', 'Ukrainian', 'Uruguayan', 'Venezuelan', 'Vietnamese', 'Welsh', 'Zimbabwean', 'Other']

    if image and allowed_file(image.filename):
        imageName = secure_filename(image.filename)
        # store the image in the uploads folder
        image.save(os.path.join(app.config['UPLOAD_FOLDER'], imageName))
        # prepare a success response since we received the image from the user
        successResponse = jsonify(
            {"successMessage": "You Will Receive An Email With The Results Shortly"})
    else:
        return jsonify(
            {"errorMessage": "Invalid File Type"})

    # run the below code after the Response is sent to user
    # @successResponse.call_on_close
    def on_close():
        session = requests.Session()
        # If the user does not select a file, the browser submits an empty file without a filename.
        if image.filename == '':
            return jsonify({"error": "no selected image"})

        if image and allowed_file(image.filename):
            imageName = secure_filename(image.filename)
            resultsFound = False
            # get image path
            artworkImgPath = 'uploads/'+imageName
            # Load The Input Image
            inputImage = Image.open(artworkImgPath).convert('RGB')
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

            for artistNationality in artistsNationalities:
                print("at: "+artistNationality)
                # getting feature vectors of the retrieved artworks images
                URL = "https://artworks-web-scraping-production.up.railway.app/WalidArtworksApi?artistNationality=" + artistNationality
                headers = {
                    'Content-Type': 'application/json',
                    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36',
                }
                # send a request to get stream of artworks json objects
                # sleep to make sure the main thread will finish first before requesting
                time.sleep(0.001)
                resp = session.get(URL, headers=headers,
                                   stream=True)
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
                        lastArtwork = decodedArtworkObj["lastArtwork"]
                        # extract features from each retrieved artwork image
                        time.sleep(0.5)
                        retrievedImage = Image.open(
                            urlopen(artworkImageUrl)).convert('RGB')

                        retrieved_image_matrix = np.asarray(
                            np.expand_dims(_transforms(retrievedImage), 0))
                        retrieved_image_vector = vgg_model.features(
                            torch.tensor(retrieved_image_matrix)).mean((2, 3))

                        # get the cosine similarity
                        cosine_similarity = torch.cosine_similarity(
                            input_image_vector, retrieved_image_vector)

                        string_cosine_similarity = str(
                            cosine_similarity)[8:12]+" %"
                        print('at max price: ' +
                              str(decodedArtworkObj["maxPrice"]))
                        print("Cosine Similarity of The Main Image and Image:" +
                              retrievedArtworkDetails + " is: " + string_cosine_similarity)
                        if cosine_similarity > 0.75:
                            print("MATCH")
                            resultsFound = True
                            # send email including the details of the matched logo
                            msg = Message('Found A Matched Artwork!',
                                          sender='stylebustersinc@gmail.com',
                                          recipients=[userEmail]
                                          )
                            msg.body = 'We found a matched Artwork! \n Artwork Details: ' + \
                                retrievedArtworkDetails+'\n Artwork Image URL Is: '+artworkImageUrl + \
                                '\n Similarity Percentage is: ' + string_cosine_similarity
                            # send the email to the user (you must put the mail.send inside the app context)
                            with app.app_context():
                                mail.send(msg)
                        # close connection after the last image (can't be closed after the for loop due to a bug)
                        if (retrievedArtworkDetails == lastArtwork):
                            print(
                                "this is the last image for nationality: "+artistNationality)
                            resp.close()
                            break

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
    serve(app, port=os.getenv("PORT"))
