
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Tuple
import pandas as pd
from abc import ABC, abstractmethod

from Test import Test

@dataclass
class Cable:
    serial_number: str
    length: int

    # Dict: "continuity" -> [Test, Test, ...]
    TESTS: Dict[str, List[Test]] = field(default_factory=lambda: {
        "continuity": [],
        "inv_continuity": [],
        "dcr": [],
        "inv_dcr": [],
        "leakage": [],
        "leakage_1s": [],
    })

    # --- Defensive migration (handles legacy code that set TESTS = [] accidentally) ---
    def _ensure_tests_dict(self) -> None:
        if isinstance(self.TESTS, list):
            # Convert legacy list of Test into dict-of-lists by test_type
            new_bucket: Dict[str, List[Test]] = {
                "continuity": [],
                "inv_continuity": [],
                "dcr": [],
                "inv_dcr": [],
                "leakage": [],
                "leakage_1s": [],
            }
            for t in self.TESTS:
                try:
                    key = t.test_type.lower()
                    new_bucket.setdefault(key, []).append(t)
                except Exception:
                    # Skip anything that isn't a Test or lacks test_type
                    pass
            self.TESTS = new_bucket

    def add_test(self, test: Test) -> None:
        self._ensure_tests_dict()
        key = test.test_type.lower()
        if key not in self.TESTS:
            self.TESTS[key] = []
        self.TESTS[key].append(test)

    def get_tests(self, test_type: str) -> List[Test]:
        self._ensure_tests_dict()
        return self.TESTS.get(test_type.lower(), [])

    def latest_test(self, test_type: str) -> Optional[Test]:
        tests = self.get_tests(test_type)
        return tests[-1] if tests else None

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
    def add_test(self, test: Test) -> None:
        self.TESTS.append(test)

    def tests_by_type(self, test_type: Test.TestType) -> List[Test]:
        return [t for t in self.TESTS if t.type == test_type]

    def last_result_for(self, test_type: Test.TestType) -> Optional[Test.TestResult]:
        filtered = [t for t in self.TESTS if t.type == test_type]
        return filtered[-1].result if filtered else None


