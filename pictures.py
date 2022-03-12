import pickle
from PIL import Image
import torch
from torchvision import transforms
import os

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
    img = Image.fromarray(p.astype("uint8"))
    #img = Image.open(p,'r')
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
    return ans



if __name__ == '__main__':
    print(classify_pic('C:/a-toxic-classifier/image_118.jpg'))
    #print(scan_pic(('image_118.jpg')))