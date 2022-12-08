from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import OAuth2PasswordBearer

import torch
from networks import define_G
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import random

import firebase_admin
from firebase_admin import credentials, storage, firestore
from pydantic import BaseModel
import numpy as np
import os
from dotenv import load_dotenv

from fastapi.middleware.cors import CORSMiddleware

# structure:
# firebase storage: stores regular and translated images
# db entries: pokemon image names and associated elos
#
# two operations: get and post
#
# get: (has three different operations)
# (1) returns the names of two random translated pokemon images (for comparison game)
# (2) returns the top 10 pokemon and their elos
# (3) returns the current number of pokemon
#
# post: (has two different operations)
# (1) takes in the image names of two pokemon and whichever one was picked as 
#     stronger, updates their elos accordingly
# (2) /generate/ takes in the name of a pet image already uploaded to fb by frontend,
#     accesses that image, converts it to a pokemon, uploads that new
#     pokemon to fb, adds db entry with pokemon image name and starting elo,
#     post request returns name of translated pokemon in fb storage

# set up FastAPI
app = FastAPI()

# default get endpoint
@app.get("/")
def home():
    return {"message" : "Health Check Passed!"}

# set up CORS stuff
origins=["https://ditto-wheat.vercel.app/",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# set up key
load_dotenv()
KEY = {
    "type": os.environ.get("type"),
    "project_id": os.environ.get("project_id"),
    "private_key_id": os.environ.get("private_key_id"),
    "private_key": os.environ.get("private_key").replace(r'\n', '\n'),
    "client_email": os.environ.get("client_email"),
    "client_id": os.environ.get("client_id"),
    "auth_uri": os.environ.get("auth_uri"),
    "token_uri": os.environ.get("token_uri"),
    "auth_provider_x509_cert_url": os.environ.get("auth_provider_x509_cert_url"),
    "client_x509_cert_url": os.environ.get("client_x509_cert_url")
}

# set up Firebase access
cred = credentials.Certificate(KEY)
fb_app = firebase_admin.initialize_app(cred, {'storageBucket' : 'ditto-f2ed7.appspot.com'})
bucket = storage.bucket()
db = firestore.client()
collection = db.collection('pokemon')
db_values_collection = db.collection('db_values')

# set up API keys and authorization
api_keys = [os.environ.get("api_key1")]
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# code from https://testdriven.io/tips/6840e037-4b8f-4354-a9af-6863fb1c69eb/
def api_key_auth(api_key: str = Depends(oauth2_scheme)):
    if api_key not in api_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key"
        )

# set up model to generate pokemon
model_dict = torch.load("model.pth")

generator = define_G(input_nc=3,output_nc=3,ngf=64,netG="resnet_9blocks", norm="instance")
generator.load_state_dict(model_dict)
generator.eval()

# set up transforms for model
encode = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((256, 256))
])

# returns two random pet image storage url's
@app.get('/randomtwo/', dependencies=[Depends(api_key_auth)])
async def random_two():
    # get current number of generated pokemon
    num_pokemon = db_values_collection.document('totals').get().to_dict()['num_pokemon']

    # sample two random pokemon
    picked = random.sample(range(1, num_pokemon+1), 2)
    res1 = collection.where('id', '==', picked[0]).limit(1).get()[0]
    res2 = collection.where('id', '==', picked[1]).limit(1).get()[0]

    name1 = res1.id
    name2 = res2.id

    url1 = res1.get('public_url')
    url2 = res2.get('public_url')

    elo1 = res1.get('elo')
    elo2 = res2.get('elo')

    return {"name1" : name1, "url1" : url1, "elo1" : elo1, "name2" : name2, "url2" : url2, "elo2" : elo2}

# returns the storage url's for the top ten pokemon
@app.get('/topten/', dependencies=[Depends(api_key_auth)])
async def top_ten():
    top_ten_pokemon = collection.order_by('elo', direction=firestore.Query.DESCENDING).limit(10).get()
    res = {}
    for i in range(len(top_ten_pokemon)):
        # res[str(i)] = top_ten_pokemon[i].get('public_url')
        # res[str(i) + "_elo"] = top_ten_pokemon[i].get('elo')
        res[i] = [top_ten_pokemon[i].get('public_url'), top_ten_pokemon[i].get('elo')]
    return res

# returns the current # of pokemon
@app.get('/numpokemon/', dependencies=[Depends(api_key_auth)])
async def num_pokemon():
    res = num_pokemon = db_values_collection.document('totals').get().to_dict()['num_pokemon']
    return res

# set up PetImage class with PyDantic
class PetImage(BaseModel):
    name: str

@app.post('/generate/', dependencies=[Depends(api_key_auth)])
async def translate_pokemon(pet_image : PetImage):
    # check if pokemon with this name already exists
    if collection.document(pet_image.name).get().exists:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Pokemon With This Name Already Exists")

    # get image from firebase and transform it
    blob = bucket.get_blob("Pets/" + pet_image.name)
    blob.download_to_filename(r"./tmp_files/tmp_image.png")
    image = encode(Image.open("./tmp_files/tmp_image.png").convert('RGB'))

    # evaluate model on pet image
    with torch.no_grad():
        res = generator(image)

    # create name for new pokemon image
    tmp = pet_image.name.split(".")
    res_name = "".join(tmp[:-1]) + "_converted." + tmp[-1]

    # get current number of generated pokemon
    num_pokemon = db_values_collection.document('totals').get().to_dict()['num_pokemon']

    # upload image to firebase
    save_image(res, "./tmp_files/tmp_image.png")
    blob = bucket.blob("Pokemon/" + res_name)
    blob.upload_from_filename("./tmp_files/tmp_image.png")
    blob.make_public()
    public_url = blob.public_url

    # set db values for new pokemon
    collection.document(pet_image.name).set({
        'elo' : 1600,
        'public_url' : public_url,
        'id' : num_pokemon + 1
    })

    # increment total number of generated pokemon
    db_values_collection.document('totals').update({
        'num_pokemon' : num_pokemon + 1
    })

    return {"public_url" : public_url}

# set up ComparePetElo class with PyDantic
class ComparePetElo(BaseModel):
    name1: str
    name2: str
    winner: int

@app.post('/updateelo/', dependencies=[Depends(api_key_auth)])
async def translate_pokemon(comparison : ComparePetElo):
    # uses ELO rating system from https://en.wikipedia.org/wiki/Elo_rating_system

    pokemon_1 = collection.document(comparison.name1).get().to_dict()
    pokemon_2 = collection.document(comparison.name1).get().to_dict()

    scoring_pt1 = 0
    scoring_pt2 = 0

    old_elo1 = pokemon_1['elo']
    old_elo2 = pokemon_2['elo']

    win_prob1 = 1 / (10 ** ((old_elo2 - old_elo1) / 400) + 1)
    win_prob2 = 1 / (10 ** ((old_elo1 - old_elo2) / 400) + 1)

    if comparison.winner == 0.5:
        scoring_pt1, scoring_pt2 = 0.5, 0.5
    elif comparison.winner == 0:
        scoring_pt1 = 1
    else:
        scoring_pt2 = 1

    new_elo1 = old_elo1 + 20 * (scoring_pt1-win_prob1)
    new_elo2 = old_elo2 + 20 * (scoring_pt2-win_prob2)
    
    pokemon_1_elo_jump = new_elo1 - old_elo1
    pokemon_2_elo_jump = new_elo2 - old_elo2

    # set new elos in db
    collection.document(comparison.name1).update({
        'elo' : new_elo1
    })
    collection.document(comparison.name2).update({
        'elo' : new_elo2
    })

    return {"Pokemon_1_Elo_Jump" : pokemon_1_elo_jump, "Pokemon_2_Elo_Jump" : pokemon_2_elo_jump,}

