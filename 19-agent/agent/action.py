from typing import Any, Optional, Dict

from pydantic import BaseModel, Field


class Action(BaseModel):
    name: str = Field(..., description="The name of the tool.")
    args: Dict[str, Any] | None = Field(None, description="The arguments to the tool.")

    def __str__(self):
        ret = f"Action(name={self.name}"
        if self.args:
            for k, v in self.args.items():
                ret += f", {k}={v}"
        ret += ")"
        return ret
