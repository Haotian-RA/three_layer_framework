import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from msg_type import DataReady, PipelineDone, Config, Token


class ActorSystem:
    """A lightweight actor scheduler."""
    
    def __init__(self, num_workers=2):
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self.actors = {}                        
        self.task_queue = queue.Queue()          
        self.states = ['idle', 'in-queue', 'processing']
        self._running = True
        self._start_workers()


    # ---- actor registration ----

    def create_actor(self, actor):
        self.actors[actor.name] = {
            'actor': actor,
            'state': self.states[0]             
        }

    # ---- messaging ----

    def send_message(self, target_name, message):
        target = self.actors.get(target_name)
        if not target:
            print(f'unknown actor: {target_name}')
            return

        target['actor'].store_message(message)          # step 1: store first
        if target['state'] == 'idle':                    # step 2: then check
            self.task_queue.put(target['actor'])
            self._advance_state(target_name)             # idle -> in-queue

    # ---- scheduling internals ----

    def _start_workers(self):
        """Launch worker threads that pull actors from the task queue."""
        for _ in range(self.executor._max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()

    def _worker_loop(self):
        """Main loop for each worker thread."""
        while self._running:
            try:
                actor = self.task_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if not self._running:
                break
            self._advance_state(actor.name)
            future = self.executor.submit(actor)
            future.add_done_callback(
                lambda _, name=actor.name: self._on_actor_done(name)
            )

    def _on_actor_done(self, actor_name):
        """Called when an actor finishes processing."""
        self._advance_state(actor_name)                  # processing -> idle
        if not self.actors[actor_name]['actor'].mailbox.empty():
            self.task_queue.put(self.actors[actor_name]['actor'])
            self._advance_state(actor_name)              # idle -> in-queue

    def _advance_state(self, actor_name):
        """Cycle actor state: idle -> in-queue -> processing -> idle."""
        current = self.actors[actor_name]['state']
        next_idx = (self.states.index(current) + 1) % len(self.states)
        self.actors[actor_name]['state'] = self.states[next_idx]

    def shutdown(self):
        self._running = False
        self.executor.shutdown(wait=False)



class Actor:
    """Base actor with mailbox and behavior switching (become pattern)."""
 
    def __init__(self, name, system):
        self.name = name
        self.system = system
        self.mailbox = queue.Queue()
        self.behavior = self._default_behavior

    def store_message(self, message):
        self.mailbox.put(message)
 
    def __call__(self):
        while not self.mailbox.empty():
            message = self.mailbox.get()
            self.behavior(message)
 
    def become(self, new_behavior):
        self.behavior = new_behavior
 
    def _default_behavior(self, message):
        raise NotImplementedError(
            f'{self.name}: no behavior set for message {message}')    



class PipelinedActor(Actor):
    """Base class for actors that dispatch work to a FlowGraph pipeline.

    Handles: slot pool, queue, backpressure, dispatch, completion.
    Subclass implements: _fill_slot(), on_result(), on_config().
    """

    def __init__(self, name, system, pool, graph, first_node):
        super().__init__(name, system)
        self._pool = pool
        self._graph = graph
        self._queue = []
        self._in_flight = 0
        self._first_node = first_node

    def _default_behavior(self, msg):
        if isinstance(msg, DataReady):
            self._queue.append(msg)
            self._try_dispatch()

        elif isinstance(msg, PipelineDone):
            self.on_result(msg)           # subclass reads slot BEFORE release
            self._pool.release(msg.slot)
            self._in_flight -= 1
            self._try_dispatch()

        elif isinstance(msg, Config):
            self.on_config(msg)

    def _try_dispatch(self):
        while self._queue:
            slot_idx = self._pool.acquire()
            if slot_idx is None:
                break                     # all slots busy, wait for pipeline_done
            msg = self._queue.pop(0)
            self._fill_slot(slot_idx, msg)
            self._in_flight += 1
            token = Token(slot=slot_idx, tag=msg.tag)
            self._graph.try_put(self._first_node, token)

    def _fill_slot(self, slot_idx, msg):
        """Read data from source, write into pool slot."""
        raise NotImplementedError

    def on_result(self, msg):
        """Process pipeline result. Slot is still valid here."""
        pass

    def on_config(self, msg):
        """Handle config message."""
        pass