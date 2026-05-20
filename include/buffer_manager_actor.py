from actor_system import Actor
from protected_circular_buffer import CircularBuffer
import threading
import time
from msg_type import BufferRead, BufferConsume, BufferRelease, DataReady, DataExhausted



class BufferManagerActor(Actor):
    """Manages ring buffer: ingestion, protection, release.
    
    All buffer mutations (protect, release, consume) happen on this
    actor's thread — no race conditions. Task actors only call
    buf.read_protected() which is read-only and thread-safe.
    
    Pending mechanisms:
      BufferRead:    queued if data not written yet, retried on timer
      BufferConsume: queued if data not written yet, retried on timer
      DataExhausted: only sent when ingestion done AND data will never arrive
    """

    def __init__(self, system, data, buffer_size, batch_size,
                 ingest_delay=0.001):
        super().__init__('buffer_manager', system)
        self.buf = CircularBuffer(buffer_size)
        self.data = data
        self.batch_size = batch_size
        self.ingestion_thread = None
        self.running = False
        self.ingest_delay = ingest_delay
        self.pending_queue = []
        self._pending_consume = None
        self._data_exhausted = False
        self._check_scheduled = False

    def _default_behavior(self, msg):

        # ---- lifecycle ----
        if msg == 'start':
            self.running = True
            self.ingestion_thread = threading.Thread(
                target=self._ingest, daemon=True)
            self.ingestion_thread.start()
            time.sleep(0.01)
            self.system.send_message('controller', 'start')

        elif msg == 'stop':
            self.running = False

        elif msg == 'reset':
            self._reset()

        elif msg == 'check_pending':
            self._check_scheduled = False
            self._check_pending()

        # ---- read request ----
        elif isinstance(msg, BufferRead):
            self._handle_read(msg)

        # ---- consume ----
        elif isinstance(msg, BufferConsume):
            self._handle_consume(msg)

        # ---- release from task actors ----
        elif isinstance(msg, BufferRelease):
            if not self.buf.release(msg.pid):
                print(f'[BufMgr] WARNING: unknown pid {msg.pid}')

    # ---- read handling ----

    def _handle_read(self, msg):
        if self._data_exhausted:
            return

        pid = self.buf.protect(msg.offset, msg.length)
        abs_end = self.buf._protections[pid]['abs_start'] + msg.length

        if abs_end <= self.buf._abs_write:
            self.system.send_message(msg.dest, DataReady(
                pid=pid, tag=msg.tag))
        else:
            self.pending_queue.append({
                'pid': pid,
                'abs_end': abs_end,
                'dest': msg.dest,
                'tag': msg.tag
            })
            self._schedule_check()

    # ---- consume handling ----

    def _handle_consume(self, msg):
        if self._data_exhausted:
            return

        target = self.buf._abs_read + msg.length
        if target <= self.buf._abs_write:
            self.buf.consume(msg.length)
        else:
            # data not written yet — queue and retry
            self._pending_consume = msg
            self._schedule_check()

    # ---- pending check (reads + consume) ----

    def _check_pending(self):
        # pending consume first — reads may depend on _abs_read
        if self._pending_consume is not None:
            msg = self._pending_consume
            target = self.buf._abs_read + msg.length
            if target <= self.buf._abs_write:
                self.buf.consume(msg.length)
                self._pending_consume = None
            elif not self.running:
                # ingestion done, data will never arrive
                self._data_exhausted = True
                self._flush_pending()
                self.system.send_message('controller', DataExhausted())
                return

        # pending reads
        still_pending = []
        for entry in self.pending_queue:
            if entry['abs_end'] <= self.buf._abs_write:
                self.system.send_message(entry['dest'], DataReady(
                    pid=entry['pid'], tag=entry['tag']))
            else:
                still_pending.append(entry)
        self.pending_queue = still_pending

        # reschedule if anything still pending
        if self.pending_queue or self._pending_consume is not None:
            self._schedule_check()

    def _flush_pending(self):
        """Release all pending protections on data exhausted."""
        for entry in self.pending_queue:
            self.buf.release(entry['pid'])
        self.pending_queue.clear()
        self._pending_consume = None

    # ---- timer ----

    def _schedule_check(self):
        if not self._check_scheduled:
            self._check_scheduled = True
            threading.Timer(self.ingest_delay, lambda:
                self.system.send_message('buffer_manager', 'check_pending')
            ).start()

    # ---- ingestion thread ----

    def _ingest(self):
        """Background thread: write samples into ring buffer in batches."""
        end = 0
        while self.running and end < len(self.data):
            start = end
            end += self.batch_size
            chunk = self.data[start:end]
            while self.running and self.buf.free_space < self.batch_size:
                time.sleep(0.001)
            self.buf.write(chunk)
            time.sleep(self.ingest_delay)
        self.running = False

    def _reset(self):
        self.running = False
        if self.ingestion_thread:
            self.ingestion_thread.join()
        self.buf.reset()
