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
from skimage import io, color
from skimage import measure # to find shape contour
from skimage.io import imsave
import numpy as np

export_file_url = 'https://drive.google.com/uc?export=download&id=1VcrnfXqHW9ieB_D2vXToP_pDHxYPWU40'
export_file_name = 'export7.pkl'

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
    im1 = img.save("geeks.jpg")
    img2= plt.imread("geeks.jpg")
    lina_gray = color.rgb2gray(img2)
    contours = canny(lina_gray, sigma=1)
    fig, ax = plt.subplots()
    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig("img3.jpg")
    with open("img3.jpg", "rb") as image:
        f = image.read()
        b = bytearray(f)
    plt.close(fig)
    camera = io.imread("img3.jpg")
    im = Image.fromarray(camera)
    #im.save("tmp.jpg")
    img5 = open_image(BytesIO(b))
    #img4 = open_image(BytesIO("img3.jpg"))
    #img = open_image(BytesIO(im))
    prediction = learn.predict(img5)[0]
    return JSONResponse({'result': str(prediction)})


if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app=app, host='0.0.0.0', port=5000, log_level="info")
