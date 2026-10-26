"""Silver-layer contracts. Everything that reaches silver must validate against these.

The agent in Phase 3 is held to the same models: its extraction code passes only if its
output validates here AND its own generated tests pass.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from trial_signal.config import HER2_DRUGS


class Trial(BaseModel):
    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    brief_title: str
    overall_status: str
    phases: list[str] = []
    enrollment: int | None = Field(default=None, ge=0)
    start_date: str | None = None
    completion_date: str | None = None
    conditions: list[str] = []
    interventions: list[str] = []
    eligibility_criteria: str | None = None  # free text — Phase 3 agent target


class ArmAdverseEvent(BaseModel):
    """One MedDRA term × one arm (event group) of one trial."""

    nct_id: str = Field(pattern=r"^NCT\d{8}$")
    group_id: str
    group_title: str
    seriousness: Literal["serious", "other"]
    term: str
    organ_system: str | None = None
    source_vocabulary: str | None = None
    num_affected: int = Field(ge=0)
    num_at_risk: int = Field(ge=0)
    num_events: int | None = Field(default=None, ge=0)

    @field_validator("term")
    @classmethod
    def term_not_blank(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("blank adverse event term")
        return v

    @model_validator(mode="after")
    def affected_le_at_risk(self) -> "ArmAdverseEvent":
        if self.num_at_risk and self.num_affected > self.num_at_risk:
            raise ValueError(f"{self.nct_id}/{self.group_id}/{self.term}: affected > at risk")
        return self

    @property
    def rate(self) -> float | None:
        return self.num_affected / self.num_at_risk if self.num_at_risk else None


# --- Phase 3 agent output contracts ---------------------------------------------------------
# The agent's extract(study) must return a dict that validates against one of these.


class EligibilityExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    her2_definition: str | None = None
    requires_prior_trastuzumab: bool | None = None
    allows_brain_mets: bool | None = None
    min_age: int | None = Field(default=None, ge=0, le=120)


class DrugMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    generic: str | None

    @field_validator("generic")
    @classmethod
    def generic_in_scope(cls, v: str | None) -> str | None:
        if v is not None and v not in HER2_DRUGS:
            raise ValueError(f"{v!r} is not one of the study's generic names")
        return v


class DrugNormalization(BaseModel):
    model_config = ConfigDict(extra="forbid")

    interventions: list[DrugMapping]
