FROM pytorch/pytorch

COPY . /app
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT [ "streamlit", "run" ]
CMD [ "src/style_transfer_website.py" ]