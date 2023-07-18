FROM python:3.10-slim
# RUN apt-get update && apt-get install -y git
COPY . .
RUN pip install -r src/requirements.txt
WORKDIR /wd
ENV PYTHONPATH "${PYTHONPATH}:/wd"
ENTRYPOINT ["python", "src/simulate.py"]