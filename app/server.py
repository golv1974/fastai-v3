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
from skimage import io, color
from skimage import measure # to find shape contour
from skimage.io import imsave
import numpy as np

export_file_url = 'https://drive.google.com/uc?export=download&id=1dd56DxY6LVqIDBzwgXLIdCApr9CH6CuI'
export_file_name = 'export6.pkl'

classes = ['crataegus', 'juglans', 'ailanthus', 'salix', 'aesculus', 'morus', 'ilex', 'populus', 'betula', 'pyrus', 'robinia', 'ulmus', 'carpinus', 'alnus', 'prunus', 'laburnum', 'quercus', 'fraxinus', 'acer', 'frangula', 'tilia', 'corylus', 'ginkgo', 'gleditsia', 'fagus', 'elaeagnus', 'malus', 'catalpa', 'sorbus', 'platanus']
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
    lina_gray = color.rgb2gray(img)
    contours = measure.find_contours(lina_gray, 0.5)
    fig, ax = plt.subplots()
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("img2.jpg")
    plt.close(fig)
    camera = io.imread("img2.jpg")
    im = Image.fromarray(camera)
    #img2 = open_image(BytesIO("img2.jpg"))
    prediction = learn.predict(im)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
