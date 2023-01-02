class StreamNotAvailable(Exception):
    def __init__(self, url: str):
        super().__init__(f"Stream {url} is not available!")
