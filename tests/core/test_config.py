from hseb.core.config import Config, IndexArgsMatrix, SearchArgsMatrix


class TestConfig:
    def test_loading(self):
        yaml_content = """
engine: nixiesearch
image: nixiesearch/nixiesearch:0.6.3
dataset: 
  dim: 384
  name: hseb-benchmark/msmarco
  query: "query-all-MiniLM-L6-v2-1K"
  corpus: "corpus-all-MiniLM-L6-v2-1K"
batch_size: 1024
experiments:
  - tag: test
    index:
      quant: [float32]
      m: [16]
      ef_construction: [64]
    search:
      ef_search: [16,32,64]
      filter_selectivity: [10, 90, 100]
"""
        loaded = Config.from_yaml(yaml_content)
        assert loaded.engine == "nixiesearch"

    def test_index_args_expand(self):
        args = IndexArgsMatrix(m=[1, 2], ef_construction=[3, 4], quant=["float32"], kwargs={"foo": [5, 6]})
        expanded = args.expand()
        assert len(expanded) == 8

    def test_search_args_expand(self):
        args = SearchArgsMatrix(ef_search=[1, 2], filter_selectivity=[3, 4], kwargs={"foo": [5, 6]})
        expanded = args.expand()
        assert len(expanded) == 8
