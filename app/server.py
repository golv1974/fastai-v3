import aiohttp
import asyncio
import uvicorn
from fastai import *
from fastai.vision import *
import os, glob
from PIL import Image
from io import BytesIO
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import HTMLResponse, JSONResponse
from starlette.staticfiles import StaticFiles
import matplotlib.pyplot as plt
import skimage
from skimage import feature
#from skimage.filter import canny
from skimage import io, color
from skimage import measure # to find shape contour
from skimage.io import imsave
import numpy as np

export_file_url = 'https://drive.google.com/uc?export=download&id=18xe1te-kckUs8HJnUwYN-ND39Du3_lz4'
export_file_name = 'export8.pkl'

classes = ['crataegus', 'juglans', 'ailanthus', 'salix', 'aesculus', 'morus', 'ilex', 'populus', 'betula', 'pyrus', 'robinia', 'ulmus', 'carpinus', 'alnus', 'prunus', 'quercus', 'fraxinus', 'acer', 'frangula', 'tilia', 'corylus', 'ginkgo', 'gleditsia', 'fagus', 'elaeagnus', 'malus', 'catalpa', 'sorbus', 'platanus']
path = Path(__file__).parent

app = Starlette()
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_headers=['X-Requested-With', 'Content-Type'])
app.mount('/static', StaticFiles(directory='app/static'))


async def download_file(url, dest):
    if dest.exists(): return
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            data = await response.read()
            with open(dest, 'wb') as f:
                f.write(data)


async def setup_learner():
    await download_file(export_file_url, path / export_file_name)
    try:
        learn = load_learner(path, export_file_name)
        return learn
    except RuntimeError as e:
        if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
            print(e)
            message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
            raise RuntimeError(message)
        else:
            raise


loop = asyncio.get_event_loop()
tasks = [asyncio.ensure_future(setup_learner())]
learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
loop.close()


@app.route('/')
async def homepage(request):
    html_file = path / 'view' / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.route('/analyze', methods=['POST'])
async def analyze(request):
    img_data = await request.form()
    img_bytes = await (img_data['file'].read())
    img = open_image(BytesIO(img_bytes))
    #result_image = Image.fromarray(c_array)
    #result_image.save(img_dir, 'PNG')
    #im1 = img.save("geeks.jpg")
    #img2= plt.imread("geeks.jpg")
    #lina_gray = color.rgb2gray(img2)
    #edges = feature.canny(lina_gray, sigma=1)
    #plt.imshow(edges, cmap='gray')
    #plt.savefig("img3.jpg")
    #plt.close()
    #with open("img3.jpg", "rb") as image:
        #f = image.read()
        #b = bytearray(f)
    #camera = io.imread("img3.jpg")
    #im = Image.fromarray(camera)
    #im.save("tmp.jpg")
    #img5 = open_image(BytesIO(b))
    #img4 = open_image(BytesIO("img3.jpg"))
    #img = open_image(BytesIO(im))
    #prediction = learn.predict(img)[0]
    preds,tensor,probs=learn.predict(img)
    classes=learn.data.classes
    def top_5_preds(preds):    
        preds_s = preds.argsort(descending=True)
        preds_s=preds_s[:5]    
        return preds_s
    def top_5_pred_labels(preds, classes):
        top_5 = top_5_preds(preds)
        labels = []
        confidence=[]
        for i in top_5:
            x=classes[i]
            p=preds[i]
            labels.append(x)
            confidence.append(p)
    top_5_predictions = top_5_pred_labels(probs,classes)
    return JSONResponse({'result': str(top_5_predictions)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
