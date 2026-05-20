from dataclasses import dataclass
import numpy as np


@dataclass
class DataReady:
    pid: int
    tag: dict

@dataclass
class PipelineDone:
    slot: int
    tag: dict

@dataclass
class Config:
    params: dict

@dataclass
class Token:
    slot: int
    tag: dict

@dataclass
class CellFound:
    N_id: int          
    f_d: float         
    F: int           
    pss_local_index: int  
    tag: dict    

@dataclass
class NoCell:
    tag: dict      


@dataclass
class MIBDecoded:
    dl_bw: int
    n_ant: int
    sfn: int
    phich_dur: str
    phich_res: str
    tag: dict

@dataclass
class MIBFail:
    tag: dict


@dataclass
class SIB1Decoded:
    sib1_bytes: np.ndarray
    tag: dict

@dataclass
class SIB1Fail:
    tag: dict


# ---- Buffer Manager messages ----

@dataclass
class BufferRead:
    offset: int
    length: int
    dest: str       
    tag: dict

@dataclass
class BufferRelease:
    pid: int

@dataclass
class BufferConsume:
    length: int

@dataclass
class DataExhausted:
    pass 


