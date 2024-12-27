FROM python:3.10

COPY .env .
COPY requirements.txt .
COPY src/ /src

# Install C compiler and tools since the LLM is using a C++ backend
RUN apt update && apt install -y gcc clang clang-tools cmake python3 git curl 

RUN pip install -r requirements.txt

WORKDIR /src

CMD ["flask", "run", "--host=0.0.0.0"]