from hseb.core.config import Config, IndexArgs, SearchArgs


class TestConfig:
    def test_loading(self):
        yaml_content = """
engine: nixiesearch
image: nixiesearch/nixiesearch:0.6.1-M1
dataset: 
  dim: 384
  name: nixiesearch/benchmark-msmarco
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

    def test_index_args_simple(self):
        value = IndexArgs.from_dict({"m": [1, 2], "ef_construction": [3, 4], "quant": ["float32"]})
        assert value == IndexArgs(m=[1, 2], ef_construction=[3, 4], quant=["float32"])

    def test_index_args_extra(self):
        value = IndexArgs.from_dict({"m": [1, 2], "ef_construction": [3, 4], "quant": ["float32"], "other": [5, 6]})
        assert value == IndexArgs(m=[1, 2], ef_construction=[3, 4], quant=["float32"], kwargs={"other": [5, 6]})

    def test_index_args_expand(self):
        args = IndexArgs(m=[1, 2], ef_construction=[3, 4], quant=["float32"], kwargs={"foo": [5, 6]})
        expanded = args.expand()
        assert len(expanded) == 8

    def test_search_args_simple(self):
        value = SearchArgs.from_dict({"ef_search": [1, 2], "filter_selectivity": [3, 4]})
        assert value == SearchArgs(ef_search=[1, 2], filter_selectivity=[3, 4])

    def test_search_args_extra(self):
        value = SearchArgs.from_dict({"ef_search": [1, 2], "filter_selectivity": [3, 4], "other": [5, 6]})
        assert value == SearchArgs(ef_search=[1, 2], filter_selectivity=[3, 4], kwargs={"other": [5, 6]})

    def test_search_args_expand(self):
        args = SearchArgs(ef_search=[1, 2], filter_selectivity=[3, 4], kwargs={"foo": [5, 6]})
        expanded = args.expand()
        assert len(expanded) == 8
