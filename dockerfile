FROM python:3.11-slim
# RUN apt-get update && apt-get install -y git
COPY . .
RUN pip install -r src/requirements.txt
WORKDIR /wd
ENV PYTHONPATH "${PYTHONPATH}:/wd"
ENTRYPOINT ["python", "src/simulate.py"]
# default parameters for simulate.py: [runs, seed, num_processes]
CMD ["100", "12345", "4"]