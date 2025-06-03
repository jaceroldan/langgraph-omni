# READ ME

## Requirements

- Python 3.13
- Docker 26+
- Postgresql 14+

## Creating deployment

1. Install Python using pyenv

    ```shell
    pyenv install 3.13
    pyenv local 3.13
    ```

2. Create a new virtualenv

    ```shell
    python -m venv venv
    source venv/bin/activate
    ```

    Use the following if on windows:

    ```shell
    python -m venv venv
    venv/Scripts/activate.bat
    ```

3. Install requirements

    ```shell
    pip install -r requirements.txt
    ```

4. Change your current directory to `deployment`

    ```shell
    cd deployment
    ```

5. Create a `.env` file inside the same directory as `docker-compose.yml` and add the following:

    ```env
    OPENAI_API_KEY=<YOUR-API-KEY-HERE>
    LANGSMITH_API_KEY=<YOUR-API-KEY-HERE>

    POSTGRES_HOST=host.docker.internal
    POSTGRES_USER=bposeatsuser
    POSTGRES_PASSWORD=bposeatspassword
    POSTGRES_PORT=5432
    POSTGRES_DB=bposeats

    API_URL=http://host.docker.internal:8000
    POSTGRES_URI=postgres://bposeatsuser:bposeatspassword@host.docker.internal:5432/bposeats?sslmode=disable
    PGVECTOR_CONNECTION_STRING=postgresql+psycopg2://bposeatsuser:bposeatspassword@host.docker.internal:5432/bposeats
    REDIS_URI=redis://langgraph-redis:6379
    ```

    **NOTE**: The format for `POSTGRES_URI` should be the following:

    ```env
    POSTGRES_URI=postgres://<DB_USER>:<DB_PASSWORD>@<DB_HOST>:5432/<DB_NAME>?sslmode=disable
    ```

    - Change the URL and environment variables according to the one you set in your `local_settings.py` inside bposeats
    - `DB_HOST` can be left as `host.docker.internal`

    This is also similar for `PGVECTOR_CONNECTION_STRING`

6. Create an image for your deployment (Always do this every time there is a new change.)

    ```shell
    langgraph build -t langgraph-image:latest
    ```

    **WARNING:** Remember that creating new images will take up space, run the following commands to free up space:

    ```shell
    docker rmi $(docker images -a)
    docker rm -vf $(docker ps -aq)
    ```

7. If everything is working, launch the deployment.

    ```shell
    docker compose up
    ```

8. To shutdown the container, do the following in terminal:

    ```shell
    docker compose down
    ```

## POSTGRES DB

Make sure to update your posgresql configs for POSTGRES memory. [document link](https://docs.google.com/document/d/1GaVL1j9g05SQYXdJGNjkqMeWNGbGYv_1I6J2QZucZ-s/edit?usp=sharing)

## Viewing your Deployed Graph

- API: http://localhost:8123
- Docs: http://localhost:8123/docs
- Langgraph Studio: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:8123

__PS__: you may change the API url by changing the port inside the `.yml` file

## FAQ

None so far

__Resources for deployments:__

- [Launching the deployment](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-6/creating.ipynb)

- [Connecting to a deployment](https://colab.research.google.com/github/langchain-ai/langchain-academy/blob/main/module-6/connecting.ipynb)
