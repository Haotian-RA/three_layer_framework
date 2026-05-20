import numpy as np


class CircularBuffer:
    """Circular buffer for complex IQ samples with protection.

    Three logical pointers tracked as ABSOLUTE positions (monotonically
    increasing, never wrap). Physical positions computed via modulo.
    """

    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = np.empty(buffer_size, dtype=complex)

        self._abs_write = 0
        self._abs_read = 0
        self._abs_protect = 0

        self._protections = {}   
        self._next_protection_id = 0
    
    def _physical(self, abs_pos):
        return abs_pos % self.buffer_size
    
    @property
    def available(self):
        return self._abs_write - self._abs_read

    @property
    def free_space(self):
        if self._protections:
            oldest = min(self._abs_protect, self._abs_read)
        else:
            oldest = self._abs_read
        return self.buffer_size - (self._abs_write - oldest)
    

    # ---- write side (called by ingestion thread) ----

    def write(self, data):
        n = len(data)
        if n > self.free_space:
            return False
        # get physical position in the buffer
        phys = self._physical(self._abs_write)
        first_part = min(n, self.buffer_size - phys)
        # before boundary
        self.buffer[phys:phys + first_part] = data[:first_part]
        # if after boundary
        if first_part < n:
            self.buffer[:n - first_part] = data[first_part:]
        # absolute position
        self._abs_write += n
        return True
    
    # ---- read side ----

    def read(self, offset, length):
        if offset + length > self.available:
            return None
        phys = self._physical(self._abs_read + offset)
        return self._read_physical(phys, length)

    def _read_physical(self, phys_start, length):
        first_part = min(length, self.buffer_size - phys_start)
        # before boundary
        result = self.buffer[phys_start:phys_start + first_part]
        # if after boundary
        if first_part < length:
            result = np.concatenate((result, self.buffer[:length - first_part]))
        return result
    
    def consume(self, length):
        """Advance _abs_read by length. No check protections."""
        if length > self.available:
            return False
        self._abs_read += length
 
        if not self._protections:
            self._abs_protect = self._abs_read
        return True
    
    # ---- protection side (eager) ----

    def protect(self, offset, length):
        """Mark a region as protected. Return protection id (pid)"""
        abs_start = self._abs_read + offset
        pid = self._next_protection_id
        self._next_protection_id += 1
 
        self._protections[pid] = {
            'abs_start': abs_start,
            'length': length
        }
 
        self._update_abs_protect()
        return pid
    
    def read_protected(self, pid):
        """Read data using a pid."""
        prot = self._protections.get(pid)
        if prot is None:
            print(f'WARNING: unknown protection ID {pid}')
            return None
 
        phys_start = self._physical(prot['abs_start'])
        return self._read_physical(phys_start, prot['length'])

    def _update_abs_protect(self):
        """Recompute abs_protect as the minimum of all active protections."""
        if self._protections:
            self._abs_protect = min(
                p['abs_start'] for p in self._protections.values())
        else:
            self._abs_protect = self._abs_read
    
    def release(self, pid):
        """Release a protection. Return True if released."""
        if pid not in self._protections:
            print(f'WARNING: unknown protection ID {pid}')
            return False
 
        del self._protections[pid]
        self._update_abs_protect()
        return True

    
    def reset(self):
        self.buffer.fill(0)
        self._abs_write = 0
        self._abs_read = 0
        self._abs_protect = 0
        self._protections.clear()
        self._next_protection_id = 0