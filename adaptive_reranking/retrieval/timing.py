import time

class TimedEncoder:
    def __init__(self, encoder):
        self.encoder = encoder
        self.query_encoding_seconds = 0.0
        self.corpus_encoding_seconds = 0.0

    def encode_queries(self, *args, **kwargs):
        start = time.perf_counter()
        encoded = self.encoder.encode_queries(*args, **kwargs)
        self.query_encoding_seconds += time.perf_counter() - start
        return encoded

    def encode_corpus(self, *args, **kwargs):
        start = time.perf_counter()
        encoded = self.encoder.encode_corpus(*args, **kwargs)
        self.corpus_encoding_seconds += time.perf_counter() - start
        return encoded
