from typing import List
from pydantic import BaseModel, Field


class DiseaseProcedureExtraction(BaseModel):
    diseases: List = Field(description="Diseases or Medical procedures extracted from Abstract.")
    # procedures: Optional[List] = Field(description="Medical procedures extracted from Abstract.")

class Disease(BaseModel):
    diseases: List[str] = Field(description="Diseases mentioned in text or referred through procedures.")
    is_carcinogenic: List[bool] = Field(description="If the disease is a cancer disease or not.")

template1 = """<|user|>
You are an expert in analyzing medical documents. Extract diseases/procedures from:
Abstract: {abstract}

Return ONLY a JSON array like: ["disease1", "procedure2", ...]

{format_instructions}<|end|>
<assistant>
"""

template2 = """<|user|>
You are an oncologist. For these terms: {disease}
1. List ONLY disease names
2. For EACH disease, state if it's cancer-related (True/False)

Example response:
{{
  "diseases": ["melanoma", "leukemia"],
  "is_carcinogenic": true
}}

{format_instructions}<|end|>
<assistant>
"""