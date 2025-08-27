from hseb.core.stats import recall


class TestRecall:
    def test_same(self):
        assert recall(approx=[1, 2, 3], exact=[1, 2, 3], k=3) == 1.0

    def test_zero(self):
        assert recall(approx=[1, 2, 3], exact=[4, 5, 6], k=3) == 0.0

    def test_cutoff(self):
        assert recall(approx=[1, 2, 3, 4, 5, 6], exact=[1, 2, 3, 7, 8, 9], k=3) == 1.0
