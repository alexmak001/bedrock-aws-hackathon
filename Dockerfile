FROM python:3.8
EXPOSE 8080
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt --upgrade-strategy eager
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]