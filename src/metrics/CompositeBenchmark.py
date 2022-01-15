from src.metrics.BaseBenchmark import BaseBenchmark
from src.model.SegmentedPointCloud import SegmentedPointCloud


class CompositeBenchmark(BaseBenchmark):
    def __init__(self, benchmarks: list):
        self.benchmarks = benchmarks

    def execute(self, cloud_predicted: SegmentedPointCloud, cloud_gt: SegmentedPointCloud):
        result = "Benchmark results:\n"
        for benchmark in self.benchmarks:
            benchmark_result = benchmark.execute(cloud_predicted, cloud_gt)
            result += str(benchmark_result) + "\n"
