from dataclasses import dataclass, field
from pandas import DataFrame
from typing import List

@dataclass
class PCAConfigurations:
    data: DataFrame
    endogen_variable: str
    ignore: List[str] = field(default_factory=list)