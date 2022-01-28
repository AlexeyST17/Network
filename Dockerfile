FROM python

RUN mkdir /usr/src/app

WORKDIR /usr/src/app

COPY . /usr/src/app

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python", "main.py"]
