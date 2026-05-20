from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import numpy as np


class FlowGraph:
    """DAG of FunctionNodes with typed input/output.

    Each node's func: _out = func(_in).
    Output of one node becomes input of the next.
    Last node (no successors) returns None and handles
    its own output internally.

    Features:
      - Per-node concurrency limits
      - Data locality: same thread runs successor
      - Pending queue when node is at capacity

    In C++: maps to tbb::flow::graph with make_edge.
    """

    def __init__(self, num_workers=2):
        self.pool = ThreadPoolExecutor(max_workers=num_workers)

    def add_edge(self, src, dst):
        """Connect src's output to dst's input."""
        src.successors.append(dst)

    def try_put(self, node, _in):
        """Submit work to a node. 

        In C++: maps to input_node.try_put() or direct
        function_node.try_put().
        """
        self._try_claim_node(node, _in)

    def _try_claim_node(self, node, _in):
        """Try to claim capacity on a node. If full, queue for later."""
        with node._lock:
            if node._active < node.concurrency:
                node._active += 1
                self.pool.submit(self._execute_node, node, _in)
            else:
                node._pending.put(_in)

    def _execute_node(self, node, _in):
        """Runs on a POOL THREAD. Execute the node, then handle continuation.

        After the node finishes, try to continue to the successor
        directly on this thread (data locality). If the successor
        is full, queue the context and return to the pool.
        """
        # run the node's function
        _out = node.stage(_in)

        # last node: call done_callback instead of dropping result
        if not node.successors:
            self._release_node(node)
            if node._done_callback and _out is not None:
                node._done_callback(_out)
            return

        #  try to claim successor for data locality
        successor_claimed = False
        succ = node.successors[0]
        with succ._lock:
            if succ._active < succ.concurrency:
                succ._active += 1
                successor_claimed = True

        # release current node, pop pending if any
        # submit it to the pool (a different thread will pick it up).
        self._release_node(node)

        if successor_claimed:
            # DATA LOCALITY: run successor on THIS thread.
            # no pool submission, no queue. data stays in cache.
            self._execute_node(node.successors[0], _out)
        elif node.successors:
            # successor was full, queue for later.
            # when successor finishes its current work, _release_node
            # will pick this up from pending.
            self._try_claim_node(node.successors[0], _out)

    def _release_node(self, node):
        """Release one unit of capacity. If pending work exists, submit it."""
        with node._lock:
            node._active -= 1
            if not node._pending.empty() and node._active < node.concurrency:
                _in = node._pending.get()
                node._active += 1
                self.pool.submit(self._execute_node, node, _in)

    def shutdown(self):
        self.pool.shutdown(wait=False)



class FunctionNode:
    """A processing stage in the flow graph.

    Wraps a callable: _out = func(_in).
    The last node in a chain returns None — it handles
    its own side effects (print, store, send message, etc.)

    In C++: maps to tbb::flow::function_node<In, Out>.
    """

    def __init__(self, name, stage, concurrency=1, done_callback=None):
        self.name = name
        self.stage = stage
        self.func = getattr(stage, 'func', None)
        self.concurrency = concurrency
        self.successors = []
        self._active = 0
        self._lock = threading.Lock()
        self._pending = queue.Queue()
        self._done_callback = done_callback # callback for last node to handle output

    def __repr__(self):
        return f'FunctionNode({self.name}, concurrency={self.concurrency})'



class SlotPool:
    """Pre-allocated working memory for pipeline stages.

    Avoids per-message allocation. N slots bound memory and provide
    backpressure: acquire() returns None when all slots are busy,
    causing the actor to queue incoming messages until a slot is released.
    """
    
    def __init__(self, n_slots, slot_factory):
        self.slots = [slot_factory() for _ in range(n_slots)]
        self._free = list(range(n_slots))

    def acquire(self):
        if self._free:
            return self._free.pop(0)
        return None

    def release(self, slot_idx):
        self.slots[slot_idx].reset()
        self._free.append(slot_idx)

    def make_stage(self, func):
        """Create pipeline stage closure. Attaches pure_func for config access."""
        pool = self
        def stage(token):
            slot = pool.slots[token.slot]
            func(slot)
            return token
        stage.func = func 
        return stage    


class ResizableSlotPool(SlotPool):
    """Safe resize: free slots immediately, in-flight on release."""

    def __init__(self, n_slots, slot_factory):
        super().__init__(n_slots, slot_factory)
        self._slot_size = len(self.slots[0].data)
        self._pending_resize = set()

    def set_size(self, new_size):
        self._slot_size = new_size
        for i in self._free:
            self.slots[i].data = np.empty(new_size, dtype=complex)
        all_slots = set(range(len(self.slots)))
        self._pending_resize = all_slots - set(self._free)

    def release(self, slot_idx):
        self.slots[slot_idx].reset()
        if slot_idx in self._pending_resize:
            self.slots[slot_idx].data = np.empty(self._slot_size, dtype=complex)
            self._pending_resize.discard(slot_idx)
        self._free.append(slot_idx)
