
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import pandas as pd
from abc import ABC, abstractmethod

from Test import Test


# Cable.py
from dataclasses import dataclass, field
from typing import List, Dict
from Test import Test

def _fresh_tests_dict() -> Dict[str, List[Test]]:
    # Keep keys *lowercase* — add_test/get_tests lower-case the key
    return {
        "continuity": [],
        "inv_continuity": [],
        "dcr": [],
        "inv_dcr": [],
        "leakage": [],
        "leakage_1s": [],
        "si": [],   # <-- lower-case 'si'
    }

@dataclass
class Cable:
    serial_number: str
    length: int
    TESTS: Dict[str, List[Test]] = field(default_factory=_fresh_tests_dict)

    def _ensure_tests_dict(self) -> None:
        """
        Migrate/initialize TESTS in all legacy states:
          - missing attr
          - None
          - mistakenly a list of Test
          - wrong/mixed-case keys
          - missing standard buckets
        """
        # 1) TESTS missing or None → create fresh
        if not hasattr(self, "TESTS") or self.TESTS is None:
            self.TESTS = _fresh_tests_dict()
            return

        # 2) Legacy state where TESTS was a list of Test objects
        if isinstance(self.TESTS, list):
            new_bucket = _fresh_tests_dict()
            for t in self.TESTS:
                try:
                    key = getattr(t, "test_type", "") or ""
                    key = key.lower() if isinstance(key, str) else "unknown"
                    new_bucket.setdefault(key, []).append(t)
                except Exception:
                    pass
            self.TESTS = new_bucket

        # 3) Normalize dict keys to lowercase
        if not isinstance(self.TESTS, dict):
            self.TESTS = _fresh_tests_dict()
            return

        self.TESTS = {str(k).lower(): v for k, v in self.TESTS.items()}

        # 4) Ensure required buckets exist
        for required in _fresh_tests_dict().keys():
            self.TESTS.setdefault(required, [])

    def add_test(self, test: Test) -> None:
        self._ensure_tests_dict()
        key = (getattr(test, "test_type", "") or "").lower()
        if key not in self.TESTS:
            self.TESTS[key] = []
        self.TESTS[key].append(test)

    def get_tests(self, test_type: str) -> List[Test]:
        self._ensure_tests_dict()
        return self.TESTS.get((test_type or "").lower(), [])

    def latest_test(self, test_type: str):
        ts = self.get_tests(test_type)
        return ts[-1] if ts else None

    def has_data(self, test_type: str, *, use_latest: bool = True) -> bool:
        if use_latest:
            t = self.latest_test(test_type)
            return (t is not None) and t.has_data()
        return any(t.has_data() for t in self.get_tests(test_type))

    # -----------------
    # Abstract API
    # -----------------
    @abstractmethod
    def extract_channel(self, text: str) -> Tuple[str, ...]:
        """Extract channel identifiers from a string."""
        raise NotImplementedError
    @abstractmethod
    def build_ordered_arrays(self, df):
        """put the channels in the correct order"""
        raise NotImplementedError
    @abstractmethod
    def build_bucket_arrays(self, df):
        """build the bucket arrays for heatmap"""
        raise NotImplementedError
    @abstractmethod
    def make_defect_heatmap(self,df):
        """create the bucket heatmap from a test"""
        raise NotImplementedError
    @abstractmethod
    def make_analog_heatmap(self,df):
        """make the heatmap from a given group of data"""
        raise NotImplementedError
    @abstractmethod 
    def overlay_failures(self, df):
        """overlay failures on given data"""
        raise NotImplementedError
    
    
    def __init__(self, serial_number: str, length: int):
        self.serial_number = serial_number
        self.length = length

    @property
    def type(self) -> str:
        raise NotImplementedError

    # -----------------
    # Common logic
    # -----------------


    def tests_by_type(self, test_type: Test.TestType) -> List[Test]:
        return [t for t in self.TESTS if t.type == test_type]

    def last_result_for(self, test_type: Test.TestType) -> Optional[Test.TestResult]:
        filtered = [t for t in self.TESTS if t.type == test_type]
        return filtered[-1].result if filtered else None


