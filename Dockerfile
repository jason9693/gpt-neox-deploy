FROM huggingface/transformers-pytorch-deepspeed-latest-gpu

RUN apt-get update && \
    apt-get install -y && \
    apt-get install -y apt-utils wget && \
    apt-get -qq -y install curl && \
    apt-get install -y tar


RUN pip install --upgrade pip

WORKDIR /app
COPY . .

RUN ls -l

EXPOSE 80

CMD ["python", "main.py"]