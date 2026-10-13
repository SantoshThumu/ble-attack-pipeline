# Start from a small official image that already has Python 3.12
FROM python:3.12-slim

# Show Python's print output in the container log straight away
ENV PYTHONUNBUFFERED=1

# Work inside /app from here on
WORKDIR /app

# Install the libraries first, so this slow step is reused until requirements.txt changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the code and the dashboard settings
COPY src/ src/
COPY .streamlit/ .streamlit/

# The dashboard listens on this port
EXPOSE 8501

# On start: build the demo data if it is missing, then run the dashboard
CMD ["sh", "-c", "python -m src.demo_data && exec streamlit run src/dashboard.py --server.address 0.0.0.0 --server.port 8501 --server.headless true"]
