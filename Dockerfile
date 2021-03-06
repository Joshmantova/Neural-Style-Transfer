FROM python:3.8

COPY . /app
WORKDIR /app/src

RUN pip install --no-cache-dir -r ../requirements.txt

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "style_transfer_website.py" ]