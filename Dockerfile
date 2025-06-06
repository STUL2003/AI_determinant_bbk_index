FROM python:3.12.10
RUN apt-get update && apt-get install -y libpq-dev
WORKDIR /app
COPY requirements.txt /app/
RUN pip install -r requirements.txt && pip cache purge
COPY RBERTTEST /app/RBERTTEST
ENV PYTHONPATH=/app
ENV DB_HOST=host.docker.internal
CMD ["uvicorn", "RBERTTEST.web.main:app", "--host", "0.0.0.0", "--port", "8000"]
