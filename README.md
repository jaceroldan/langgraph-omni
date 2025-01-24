# READ ME

## Requirements
- Python 3.13
- Docker
- Postgres

## Creating deployment

1. Install required packages
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

5. Create an image for your deployment

    ```cmd
    langgraph build -t my-image
    ```

6. If everything is working, lauch the deployment.

    ```cmd
    docker-compose up
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
