"""Auditable functional-family slot templates.

The templates distinguish a complete family assembly from a valid binary
subassembly.  This prevents shaft+hub or housing+bearing pairs from being
silently treated as the complete target assembly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlotSpec:
    role: str
    minimum: int
    maximum: int
    required_for_complete: bool
    candidate_cap: int


@dataclass(frozen=True)
class RelationSpec:
    roles: tuple[str, str]
    accepted_relation_types: frozenset[str]
    minimum_physical_evidence: int = 1
    required_for_complete: bool = True


@dataclass(frozen=True)
class FamilyTemplate:
    family: str
    center_roles: frozenset[str]
    slots: tuple[SlotSpec, ...]
    relations: tuple[RelationSpec, ...]
    minimum_size: int
    maximum_size: int
    description: str

    @property
    def complete_roles(self) -> frozenset[str]:
        return frozenset(
            slot.role for slot in self.slots if slot.required_for_complete
        )


CYLINDRICAL = frozenset({"clearance", "coaxial", "pocket_mate"})
PLANAR = frozenset({"planar_mate", "planar_align"})


TEMPLATES = (
    FamilyTemplate(
        family="cover_base",
        center_roles=frozenset({"base", "cover"}),
        slots=(
            SlotSpec("base", 1, 1, True, 4),
            SlotSpec("cover", 1, 1, True, 4),
            SlotSpec("locating_pin", 1, 2, True, 5),
        ),
        relations=(
            RelationSpec(("base", "cover"), PLANAR, 1, True),
            RelationSpec(("base", "locating_pin"), CYLINDRICAL, 1, True),
            RelationSpec(("cover", "locating_pin"), CYLINDRICAL, 1, True),
        ),
        minimum_size=3,
        maximum_size=4,
        description="registered cover on a base with at least one locating element",
    ),
    FamilyTemplate(
        family="shaft_hub_key",
        center_roles=frozenset({"shaft", "hub"}),
        slots=(
            SlotSpec("shaft", 1, 1, True, 4),
            SlotSpec("hub", 1, 1, True, 4),
            SlotSpec("key", 1, 1, True, 5),
            SlotSpec("axial_retainer", 0, 1, False, 4),
        ),
        relations=(
            RelationSpec(("shaft", "hub"), CYLINDRICAL, 2, True),
            RelationSpec(("shaft", "key"), PLANAR, 1, True),
            RelationSpec(("hub", "key"), PLANAR, 1, True),
            RelationSpec(("shaft", "axial_retainer"), CYLINDRICAL, 1, False),
            RelationSpec(("hub", "axial_retainer"), PLANAR, 1, False),
        ),
        minimum_size=3,
        maximum_size=4,
        description="keyed shaft and hub torque-transfer assembly",
    ),
    FamilyTemplate(
        family="bearing_housing",
        center_roles=frozenset({"housing", "shaft"}),
        slots=(
            SlotSpec("housing", 1, 1, True, 4),
            SlotSpec("bearing", 1, 1, True, 5),
            SlotSpec("shaft", 1, 1, True, 4),
            SlotSpec("end_cover", 1, 1, True, 5),
            SlotSpec("bearing_retainer", 0, 1, False, 4),
        ),
        relations=(
            RelationSpec(("housing", "bearing"), CYLINDRICAL, 2, True),
            RelationSpec(("bearing", "shaft"), CYLINDRICAL, 2, True),
            RelationSpec(("housing", "end_cover"), PLANAR, 1, True),
            RelationSpec(("housing", "bearing_retainer"), PLANAR, 1, False),
        ),
        minimum_size=4,
        maximum_size=5,
        description="housing, bearing, shaft, and registered end-cover assembly",
    ),
)


TEMPLATE_BY_FAMILY = {template.family: template for template in TEMPLATES}
