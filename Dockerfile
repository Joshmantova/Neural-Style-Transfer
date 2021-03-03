FROM python:3.8

COPY . /app
WORKDIR /app/src

RUN pip install --no-cache-dir -r ../requirements.txt

ENTRYPOINT [ "Python" ]
CMD ["style_transfer.py"]