import time
from flow_graph import FlowGraph, FunctionNode
from threading import Event

def time_stage(func, data, n_runs=50):
    """Time a single stage in isolation. Returns avg ms."""
    # warmup
    for _ in range(10):
        func(data)

    # measure
    t0 = time.perf_counter()
    for _ in range(n_runs):
        func(data)
    elapsed = time.perf_counter() - t0

    return elapsed / n_runs * 1000


def make_sink(collected, done, n):
    def sink(x):
        collected.append(x)
        if len(collected) >= n:
            done.set()
    return sink


def test_concurrency(func, inputs, configs=None):
    if configs is None:
        configs = [(1, 1), (2, 2), (4, 4), (6, 6), (10, 10)]

    n = len(inputs)
    results = []

    for workers, cc in configs:
        collected = []
        done = Event()

        g = FlowGraph(num_workers=workers)
        stage = FunctionNode('stage', func, concurrency=cc)
        sink_node = FunctionNode('sink', make_sink(collected, done, n), concurrency=cc)
        g.add_edge(stage, sink_node)

        t0 = time.perf_counter()
        for inp in inputs:
            g.try_put(stage, inp)
        done.wait(timeout=30)
        elapsed = time.perf_counter() - t0

        g.shutdown()
        time.sleep(0.1)

        results.append((workers, cc, elapsed))

    return results