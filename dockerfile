FROM python:3.11-slim
COPY . .
RUN pip install -r src/requirements.txt
WORKDIR /wd
ENV PYTHONPATH "${PYTHONPATH}:/wd"
ENTRYPOINT ["python", "src/simulate.py"]
# default parameters for simulate.py: [runs, seed, num_processes, plot_only]
CMD ["100", "12345", "4", "--sim"]