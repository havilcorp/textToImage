FROM python:3.9.6

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install matplotlib
RUN pip install taming-transformers
RUN pip install datasets
RUN curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fckpts%2Flast.ckpt&dl=1' > vqgan_im1024.ckpt
RUN curl -L 'https://heibox.uni-heidelberg.de/d/8088892a516d4e3baf92/files/?p=%2Fconfigs%2Fmodel.yaml&dl=1' > vqgan_im1024.yaml
RUN pip install einops
RUN pip install omegaconf
RUN pip install openimages
RUN pip install pytorch_lightning
RUN wget https://storage.googleapis.com/localized-narratives/annotations/open_images_validation_captions.jsonl

COPY . .

CMD [ "python", "main.py" ]
