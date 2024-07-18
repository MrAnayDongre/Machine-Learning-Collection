# Run for development
# python server.py
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="localhost", port=8000, reload=True)
