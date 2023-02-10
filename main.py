from mingpt.utils import sample
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.model import GPT, GPTConfig
import logging
from omegaconf import OmegaConf
from taming.models import cond_transformer, vqgan
from PIL import Image
from matplotlib import pyplot as plt
import json
import urllib3
import boto3
import botocore
import concurrent.futures
import os
from tqdm import tqdm
import einops
from torch.utils.data import Dataset
import math
from torch.nn import functional as F
import torch.nn as nn
import torch
import numpy as np
import sys



print("torch.cuda.device_count()", torch.cuda.device_count())
print("torch.get_num_threads()", torch.get_num_threads())
print("torch.backends.mps.is_available()", torch.backends.mps.is_available())
print("torch.backends.mps.is_built()", torch.backends.mps.is_built())
print("torch.cuda.is_available()", torch.cuda.is_available())



logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# Это немного запутанно, но цель здесь
# - это просто для того, чтобы подготовить все к предстоящей демонстрации.

# Загрузка предварительно обученной модели VQGAN
print('Downloading VQGAN model weights')

# Установите требования к VQGAN
print('Installing requirements')
# !git clone https: // github.com/CompVis/taming-transformers & > / dev/null

# Настройка VQGAN
sys.path.append('./tt')


def load_vqgan_model(config_path, checkpoint_path):
    config = OmegaConf.load(config_path)
    if config.model.target == 'taming.models.vqgan.VQModel':
        model = vqgan.VQModel(**config.model.params)
        model.eval().requires_grad_(False)
        model.init_from_ckpt(checkpoint_path)
    elif config.model.target == 'taming.models.cond_transformer.Net2NetTransformer':
        parent_model = cond_transformer.Net2NetTransformer(
            **config.model.params)
        parent_model.eval().requires_grad_(False)
        parent_model.init_from_ckpt(checkpoint_path)
        model = parent_model.first_stage_model
    else:
        raise ValueError(f'unknown model type: {config.model.target}')
    del model.loss
    return model


with open('open_images_validation_captions.jsonl', 'r') as json_file:
    json_list = list(json_file)

print('Loading VQGAN model')
# device = torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vqgan_model = load_vqgan_model(
    'vqgan_im1024.yaml',
    'vqgan_im1024.ckpt'
).to(device)

print('Dataset Prep')

for json_str in json_list[:5]:
    result = json.loads(json_str)


def _download_single_image(arguments):
    if os.path.exists(arguments["dest_file_path"]):
        return

    try:
        with open(arguments["dest_file_path"], "wb") as dest_file:
            arguments["s3_client"].download_fileobj(
                "open-images-dataset",
                arguments["image_file_object_path"],
                dest_file,
            )

    except urllib3.exceptions.ProtocolError as error:
        logging.Logger.warning(
            f"Unable to download image {arguments['image_file_object_path']} -- skipping",
            error,
        )


def download_images_by_id(image_ids, section, images_directory):
    # мы загрузим изображения из AWS S3, поэтому нам понадобится клиент boto S3
    s3_client = boto3.client(
        's3',
        config=botocore.config.Config(signature_version=botocore.UNSIGNED),
    )
    # создайте повторяющийся список аргументов функции
    # который мы сопоставим с функцией загрузки
    download_args_list = []
    for image_id in image_ids:
        image_file_name = image_id + ".jpg"
        download_args = {
            "s3_client": s3_client,
            "image_file_object_path": section + "/" + image_file_name,
            "dest_file_path": os.path.join(images_directory, image_file_name),
        }
        download_args_list.append(download_args)

    # используйте ThreadPoolExecutor для параллельной загрузки изображений
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # используйте исполнитель для сопоставления функции загрузки с итерируемым набором аргументов
        list(
            tqdm(
                executor.map(_download_single_image, download_args_list),
                total=len(download_args_list),
                desc="Downloading images"
            )
        )


def get_openimages(n_images=2000):
    download_images_by_id(
        [
            json.loads(s)['image_id'] for s in json_list[:n_images]
        ],
        'validation',
        'ims/'
    )
    data = [json.loads(s) for s in json_list[:n_images]]
    return data


type(vqgan_model)  # Наша модель vqgan загружена

# @title измененный набор данных теперь включает текстовые маркеры перед графическими

max_text_length = 128
vocab = 'abcdefghijklmnopqrstuvwxzy '


def encode_char(c):
    if c in vocab:
        return vocab.index(c)
    return 50  # 'special character'


class PatchDataset(Dataset):

    def __init__(self, image_fns, labels, block_size=255, max_text_length=128):

        # Перебирайте изображения, получая токены VQGAN из версий размером 256 пикселей и встраиваний эмбендингов CLIP
        self.ims = []  # Изображения, закодированные VQGAN
        self.conds = []

        for fn, caption in tqdm(zip(image_fns, labels)):
            self.ims.append(fn)
            self.conds.append(caption)

        # 1024 возможных кода VQGAN + наша кодировка текста
        chars = range(1024+53)
        data_size, vocab_size = len(image_fns), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size  # + max_text_length << TODO?
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.ims)  # was len(self.data) - self.block_size

    def __getitem__(self, idx):

        fn = self.ims[idx]
        caption = self.conds[idx]

        # Закодируйте изображение с помощью vegan
        pil_im = Image.open(fn).convert('RGB').resize((256, 256))
        im_tensor = torch.tensor(np.array(pil_im)).permute(2, 0, 1) / 255
        with torch.no_grad():
            z, a, b = vqgan_model.encode(
                im_tensor.to(device).unsqueeze(0) * 2 - 1
            )
        im_idxs = b[-1]  # 16*16

        # Закодируйте текст:
        char_idxs = [encode_char(c) for c in caption.lower()[:max_text_length]]
        while len(char_idxs) < max_text_length:
            char_idxs += [51]
        # 52 - это конец текстовых токенов.
        char_idxs += [52]
        # На данный момент они будут конфликтовать с токенами от vegan, поэтому мы добавляем 1024
        char_idxs = [c+1024 for c in char_idxs]

        # Комбинируем
        dix = [self.stoi[int(s)] for s in char_idxs]
        dix += [self.stoi[int(s)] for s in im_idxs]

        # Разделить на x и y
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


# 2 тысячи на быструю демонстрацию, 20 тысяч на лучшую попытку обучения
data = get_openimages(n_images=100)

# Наш новый набор данных
image_fns = ['ims/'+d['image_id']+'.jpg' for d in data]
labels = [d['caption'] for d in data]
dset = PatchDataset(image_fns, labels, max_text_length=max_text_length)
x, y = dset[0]
x.shape, y.shape, x[-3:], y[-3:]  # Y - это x, смещенный на 1.

# @title Обучение модели делает брррр...
block_size = 255+max_text_length+1  # Чтобы также установить кондиционирование
mconf = GPTConfig(
    dset.vocab_size, block_size,
    n_layer=8, n_head=8, n_embd=512
)
model = GPT(mconf)

# Обучение
print("TRAIN START")
tconf = TrainerConfig(
    max_epochs=10, batch_size=32, learning_rate=6e-4,
    lr_decay=True, warmup_tokens=512*20, final_tokens=2*len(dset)*block_size,
    num_workers=0
)  # num_workers=0, чтобы избежать некоторых ошибок многопроцессорной обработки
trainer = Trainer(model, dset, None, tconf)
trainer.train()
print("TRAIN DONE")

print("SAVE MODEL")
f = open("model", "a")
f.write(model)
f.close()
print("SAVE MODEL DONE!")

# @title Генерируем

prompt = 'cat'  # @param

# Закодируйте промт так, как мы это делаем в наборе данных
char_idxs = [encode_char(c) for c in prompt.lower()[:max_text_length]]
while len(char_idxs) < max_text_length:
    char_idxs += [51]
char_idxs += [52]
char_idxs = [c+1024 for c in char_idxs]

# Брррр...
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for i in tqdm(range(9)):
    x = torch.tensor(
        [dset.stoi[s] for s in char_idxs],
        dtype=torch.long
    )[None, ...].to(device)
    y = sample(model, x, 256, temperature=1., sample=True, top_k=200)[0]
    completion = [dset.itos[int(i)] for i in y]
    ccc = completion[-256:]
    ccc = [min(c, 1023) for c in ccc]
    with torch.no_grad():
        zs = vqgan_model.quantize.get_codebook_entry(
            torch.tensor(ccc).to(device), (1, 16, 16, 256)
        )
        axs[i % 3, i//3].imshow(vqgan_model.decode(zs).add(
            1).div(2).cpu().squeeze().permute(1, 2, 0).clip(0, 1)
        )
