from dataclasses import dataclass, field
import re
from tqdm.auto import tqdm
from rich import print, print_json, inspect
from rich.console import Console
from rich.json import JSON

console = Console()
log = console.log
print = log


@dataclass
class Subtitle:
    words: str
    start: float
    end: float


@dataclass
class WordGroup:
    source: str
    target: str = ""
    start: float = 0
    end: float = 0
    end_of_sentence: bool = False
    reference: Subtitle = None
    filler: bool = False

sent_enders = re.compile(r'.+[\.\!\?\;\)]+\"*$')
