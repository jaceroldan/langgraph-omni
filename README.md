# READ ME

## Requirements

- Python 3.13
- Docker
- Postgres
- Pyenv

## Creating deployment

1. Install Python using pyenv

    ```cmd
    pyenv install 3.13
    pyenv local 3.13
    ```

2. Create `docker-compose.yml` from `docker-compose example.yml`
3. Place your API keys inside your newly created `docker-compose.yml`.

    ```yml
    OPENAI_API_KEY: <YOUR-API-KEY-HERE>
    LANGSMITH_API_KEY: <YOUR-API-KEY-HERE>
    ```

4. Change your current directory to `deployment`

    ```cmd
    cd deployment
    ```
5. Create a new virtualenv

    ```cmd
    python -m venv venv
    source venv/bin/activate
    ```

    Use the following if on windows:

    ```cmd
    python -m venv venv
    venv/Scripts/activate.bat
    ```

6. Install requirements

    ```cmd
    pip install -r requirements.txt
    ```

7. Create an image for your deployment (Always do this every time there is a new change.)

    ```cmd
    langgraph build -t my-image
    ```

8. If everything is working, launch the deployment.

    ```cmd
    docker-compose up
    ```

9. To shutdown the container, do the following in terminal:

    ```cmd
    docker-compose down
    ```

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
