FROM alpine

RUN apk --update add \
    python3 \
    python3-dev \
    py-pip

ADD . /opt/

RUN python3 -m pip install --break-system-packages --upgrade wheel
RUN python3 -m pip install --break-system-packages --upgrade /opt/.
