FROM python:3.12-slim-bookworm
ENV PYTHONUNBUFFERED=1 MPLBACKEND=Agg MANAGER_RUNTIME_VERSION=0.1.0
LABEL io.hass.name="Discord Bot Manager" io.hass.description="Persistent Python Discord bot manager" io.hass.type="addon" io.hass.version="0.1.0" io.hass.arch="aarch64|amd64"
WORKDIR /opt/manager
COPY seed/requirements.txt /opt/seed/requirements.txt
RUN pip install --no-cache-dir -r /opt/seed/requirements.txt
COPY manager/requirements.txt /opt/manager/requirements.txt
RUN python -m venv /opt/manager-venv && /opt/manager-venv/bin/pip install --no-cache-dir -r requirements.txt
COPY seed /opt/seed
COPY manager /opt/manager
CMD ["/opt/manager-venv/bin/python", "app.py"]
