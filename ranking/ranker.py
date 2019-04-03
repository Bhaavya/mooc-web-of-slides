import metapy

idx = metapy.index.make_inverted_index('slides-config.toml')
ranker = metapy.index.OkapiBM25()
query = metapy.index.Document()
query.content()
top_docs = ranker.score(idx, query, num_results=10)