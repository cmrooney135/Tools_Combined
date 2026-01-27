
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import pandas as pd



@dataclass
class Test:
    
    """Represents a single test run on a cable."""
    test_type: str          # e.g. "CONTINUITY", "LEAKAGE", "DCR"
    result: str             # e.g. "PASS", "FAIL", "NA"
    operator: str
    retry_count: int = 0

    test_date: Optional[str] = None
    test_time: Optional[str] = None

    data: Optional[pd.DataFrame] = None
    failure_data: Optional[pd.DataFrame] = None
    
    def __post_init__(self) -> None:
        self.test_type = (self.test_type or "").lower()
        self.result = (self.result or "").upper()

    def has_data(self) -> bool:
        return self.data is not None and not self.data.empty

    def has_failures(self) -> bool:
        return self.failure_data is not None and not self.failure_data.empty
