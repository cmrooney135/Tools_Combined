
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
    zo_data: Optional[pd.DataFrame] = None
    skew_data: Optional[pd.DataFrame] = None
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
        zo_data: Optional[pd.DataFrame] = None,
        skew_data: Optional[pd.DataFrame] = None,
        traces: Dict[str, pd.DataFrame] = field(default_factory=dict, repr=False),

        
    ):
        # Call base initializer
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
        # Assign SI-specific declared fields (so repr will show them)
        self.test_end = test_end
        self.SI_type = SI_type
        self.zo_data = zo_data
        self.skew_data = skew_data
        self.traces = traces

    def __post_init__(self):
        # Backfill for old instances or deserialized objects
        if not hasattr(self, "traces") or self.traces is None:
            self.traces = {}
