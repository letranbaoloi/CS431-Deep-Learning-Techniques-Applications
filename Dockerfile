FROM python:3.12.7-slim


WORKDIR /irs

COPY requirements.txt /irs/requirements.txt


RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y git

RUN pip install git+https://github.com/openai/CLIP.git

COPY . /irs

# ENV FLASK_APP=app.py

EXPOSE 5000

ENTRYPOINT ["python"]
# CMD ["python", "-m", "flask", "run", "--host=0.0.0.0"]
CMD ["app.py"]

