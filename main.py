from fastapi import FastAPI
import torch
from networks import define_G
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import firebase_admin
from firebase_admin import credentials, storage, firestore
from pydantic import BaseModel
import numpy as np

# structure:
# firebase storage: stores regular and translated images
# db entries: pokemon image names and associated elos
#
# two operations: get and post
#
# get: (has two different operations)
# (1) returns the names of two random translated pokemon images (for comparison game)
# (2) returns the top 10 pokemon and their elos
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

# set up Firebase access
cred = credentials.Certificate("key.json")
fb_app = firebase_admin.initialize_app(cred, {'storageBucket' : 'ditto-f2ed7.appspot.com'})
storage_url = 'gs://ditto-f2ed7.appspot.com'
bucket = storage.bucket()
db = firestore.client()
collection = db.collection('pokemon')

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

# set up PetImage class with PyDantic
class PetImage(BaseModel):
    name: str

@app.post('/generate/')
async def translate_pokemon(pet_image : PetImage):
    # get image from firebase and transform it
    blob = bucket.get_blob("Pets/" + pet_image.name)
    blob.download_to_filename(r"./tmp_files/tmp_image.png")
    image = encode(Image.open("./tmp_files/tmp_image.png"))

    # evaluate model on pet image
    with torch.no_grad():
        res = generator(image)

    # create name for new pokemon image
    tmp = pet_image.name.split(".")
    res_name = "".join(tmp[:-1]) + "_converted." + tmp[-1]

    # set db values for new pokemon
    pokemon_storage_url = storage_url + "/Pokemon/" + res_name
    collection.document(pet_image.name).set({
        'elo' : 1600,
        'storage_url' : pokemon_storage_url
    })

    # upload image to firebase
    save_image(res, "./tmp_files/tmp_image.png")
    blob = bucket.blob("Pokemon/" + res_name)
    blob.upload_from_filename("./tmp_files/tmp_image.png")

    return {"storage_url" : pokemon_storage_url}

# set up ComparePetElo class with PyDantic
class ComparePetElo(BaseModel):
    name1: str
    name2: str
    winner: int

@app.post('/updateelo/')
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

