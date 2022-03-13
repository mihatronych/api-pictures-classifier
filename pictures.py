import pickle
from PIL import Image
import torch
from torchvision import transforms
import os
import io
import json
import requests
import urllib.request

def read_pickle(name):
    objects = []
    with open(os.path.dirname(os.path.abspath(__file__)) + '/' + name + '.pickle', 'rb') as f:
        while True:
            try:
                objects.append(pickle.load(f, encoding="latin1"))
            except EOFError:
                break
        return objects[0]


def classify_pic(p):
    model = read_pickle("data2")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    tfms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    urllib.request.urlretrieve(p, 'gfg.png')

    img = Image.open('gfg.png', 'r')

    img = img.convert('RGB')

    preproc_img = tfms(img).unsqueeze(0)

    print(preproc_img.shape)  # torch.Size([1, 3, 224, 224])

    # Classify
    model.eval()
    with torch.no_grad():
        outputs = model(preproc_img)
    # with torch.no_grad():
    #    outputs = model(preproc_img)

    # Print predictions Точность 61.04%
    print('-----')
    ans = []
    for idx in torch.topk(outputs, k=2).indices.squeeze(0).tolist():
        prob = torch.softmax(outputs, dim=1)[0, idx].item()
        if idx==0:
            ans.append("{p:.2f}%".format(p=prob * 100))  # z='toxic'
        else:
            ans.append("{p:.2f}%".format(p=prob * 100)) # z='untoxic'
    os.remove('gfg.png')
    return ans

def scan_pic(p):
    try:
        urllib.request.urlretrieve(p, 'gfg.png')

        img = Image.open('gfg.png', 'r')

        url_api = "https://api.ocr.space/parse/image"
        #img = img.convert('RGB')
        img = img.convert('L')

        b = io.BytesIO()

        img.save(b, 'jpeg')

        os.remove('gfg.png')

        im_bytes = b.getvalue()

        result = requests.post(
            url_api,
            files = {"screenshot.jpg": im_bytes},
            data = {"apikey": "K86626091088957",
                "language": "rus"}
        )
        result = result.content.decode()
        result = json.loads(result)
        print(result)
        parsed_results = result.get("ParsedResults")[0]
        text_detected = parsed_results.get("ParsedText")
        return text_detected

    except:
        return "ОШИБКА ПРИ РАСПОЗНАВАНИИ"


if __name__ == '__main__':
    # print(classify_pic('C:/a-toxic-classifier/image_118.jpg'))
    # print(scan_pic(('C:/a-toxic-classifier/meme.jpg')))
    # https://i.siteapi.org/Gsz8OOaiXvgJzxIqWwWf3bDI0qk=/fit-in/1400x1000/center/top/s.siteapi.org/596efcf47164ff2.ru/img/etuhz80ezm040ok4o0kgowskssk0cc
    print(scan_pic(('https://i.siteapi.org/Gsz8OOaiXvgJzxIqWwWf3bDI0qk=/fit-in/1400x1000/center/top/s.siteapi.org/596efcf47164ff2.ru/img/etuhz80ezm040ok4o0kgowskssk0cc')))