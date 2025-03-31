FROM python:3.9

WORKDIR /loan_app
COPY . /loan_app

RUN pip install -r requirements.txt

CMD ["python", "loan_app.py"]
