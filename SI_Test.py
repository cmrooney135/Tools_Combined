# SI_Test.py
from dataclasses import dataclass, field
from typing import Optional, Dict
import pandas as pd
from Test import Test

@dataclass(init=False, repr=True)
class SI_Test(Test):
    # ----- Declare ALL fields here so dataclass repr shows them -----
    # Base/metadata
    test_type: str = "si"
    result: str = "NA"
    operator: str = ""
    retry_count: int = 0
    test_date: Optional[str] = None   # anchor (Zo time only)
    test_time: Optional[str] = None
    data: Optional[pd.DataFrame] = None          # (unused for SI; keep for base compatibility)
    failure_data: Optional[pd.DataFrame] = None

    # SI specific
    test_end: str = ""                # "P1" or "P2"
    SI_type: str = ""                 # last seen ("zo"/"skew")

    # NOTE: elsewhere you treat zo_data as a dict of DataFrames; reflect that here.
    zo_data: Optional[Dict[str, pd.DataFrame]] = None

    skew_data: Optional[pd.DataFrame] = None

    # IMPORTANT: define the dataclass field *here* with default_factory,
    # and put repr=False here (not in __init__ params).
    traces: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False)

    def __init__(
        self,
        *,
        test_type: str = "si",
        result: str = "NA",
        operator: str = "",
        retry_count: int = 0,
        test_date: Optional[str] = None,
        test_time: Optional[str] = None,
        data: Optional[pd.DataFrame] = None,
        failure_data: Optional[pd.DataFrame] = None,
        # SI-specific
        test_end: str = "",
        SI_type: str = "",
        zo_data: Optional[Dict[str, pd.DataFrame]] = None,
        skew_data: Optional[pd.DataFrame] = None,
        traces: Optional[Dict[str, pd.DataFrame]] = None,
    ):
        # Initialize base
        super().__init__(
            test_type=test_type,
            result=result,
            operator=operator,
            retry_count=retry_count,
            test_date=test_date,
            test_time=test_time,
            data=data,
            failure_data=failure_data,
        )

        # Assign SI-specific fields
        self.test_end = test_end
        self.SI_type = SI_type
        self.zo_data = zo_data
        self.skew_data = skew_data

        # Ensure we never end up with a Field/None here
        self.traces = {} if traces is None else dict(traces)

        # If you want to keep some normalization step, do it here (since __post_init__ won't be auto-called)
        # Example: coerce wrong zo_data shapes to None instead of blowing up later.
        if self.zo_data is not None and not isinstance(self.zo_data, dict):
            self.zo_data = None

    # With init=False and a custom __init__, __post_init__ is not auto-invoked.
    # You can delete this method, or call it manually from __init__ if you still want it.
    def __post_init__(self):
        if not isinstance(getattr(self, "traces", None), dict):
            self.traces = {}
