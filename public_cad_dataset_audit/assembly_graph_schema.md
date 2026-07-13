# Unified Assembly Graph Schema

Version: `1.0.0`

The unit of a node is a **part occurrence-body instance**, not merely a unique
body definition. Repeated occurrences of one body therefore receive different
`part_id` values. This is required because a repeated fastener can participate
in different relations and have a different transform.

## Top-level object

```json
{
  "schema_version": "1.0.0",
  "assembly_id": "stable source assembly id",
  "source_dataset": "fusion360_gallery_assembly | automate | linkify",
  "source_record_path": "path or source URI",
  "source_record_sha256": "optional local-record checksum",
  "units": {"length": "cm", "angle": "radian"},
  "parts": [],
  "positive_part_pair_edges": [],
  "negative_part_pair_edges": [],
  "failure_reasons": [],
  "unavailable_fields": [],
  "quality": {
    "status": "usable | partial | insufficient | failed",
    "failure_reasons": [],
    "unavailable_fields": []
  }
}
```

Every output must contain `failure_reasons` and `unavailable_fields`, including
successful outputs, where they may be empty lists.

## Part node

```json
{
  "part_id": "occurrence_uuid_body_uuid",
  "body_id": "source body definition id",
  "occurrence_id": "source occurrence id or null for root bodies",
  "part_name": "source name if available",
  "visible": true,
  "transform": [[1, 0, 0, 0], [0, 1, 0, 0],
                [0, 0, 1, 0], [0, 0, 0, 1]],
  "geometry": {
    "path": "body.step",
    "format": "step | smt | x_t | obj",
    "available": true,
    "candidates": [],
    "unavailable_reason": null
  }
}
```

`geometry.path` points to geometry; geometry is not embedded in graph JSON.
Source transforms and units must be retained without implicit conversion.

## Positive part-pair edge

A positive pair has at least one recorded joint, as-built joint, mate, or
contact. Multiple source relations between the same unordered pair are
aggregated, while their individual records remain under `relations`.

```json
{
  "edge_id": "positive_000001",
  "part_pair": ["part_a", "part_b"],
  "relation_types": ["RigidJointType", "contact"],
  "relation_kinds": ["joint", "contact"],
  "source_dataset": "fusion360_gallery_assembly",
  "relations": [
    {
      "source_relation_id": "source id",
      "relation_kind": "joint | as_built_joint | contact | mate",
      "relation_type": "source-specific subtype",
      "part_pair_mapping_status": "mapped | unmapped",
      "interface_entities": []
    }
  ],
  "failure_reasons": [],
  "unavailable_fields": []
}
```

## Interface entity reference

```json
{
  "part_id": "part_a",
  "mapping_status": "mapped | unmapped | unavailable",
  "failure_reason": null,
  "entity_type": "BRepFace | BRepEdge | BRepVertex | null",
  "entity_id": "BRepFace:6",
  "body_id": "source body id",
  "occurrence_id": "source occurrence id",
  "topology_index": 6,
  "topology_index_available": true,
  "geometry_file_available": true,
  "local_geometry": {
    "surface_type": "CylinderSurfaceType",
    "point_on_entity": {},
    "bounding_box": {}
  },
  "unavailable_fields": []
}
```

An index being present means it is structurally addressable in the source
dataset. It does **not** prove that a neutral STEP importer preserves the same
topological numbering. Verification against `.smt` or indexed `.obj` geometry
must be reported separately.

## Negative part-pair edge

```json
{
  "edge_id": "negative_000001",
  "part_pair": ["part_a", "part_c"],
  "relation_type": "none_observed",
  "source_dataset": "fusion360_gallery_assembly",
  "negative_definition": "No recorded positive relation in this assembly.",
  "failure_reasons": [],
  "unavailable_fields": ["proof_of_physical_non_interaction"]
}
```

A negative is a **closed-world dataset negative**, not proof that the two parts
can never mate. Train/validation splits must be made by assembly or source
document before negative sampling to avoid geometry and author leakage.

## Dataset mappings

### Fusion 360 Gallery Assembly

- part node: visible occurrence-body instance;
- positive: `joints`, `as_built_joints`, or `contacts`;
- interface: source body plus B-Rep face/edge index and optional local data;
- geometry: `.smt` is the topology-index ground truth; `.step` is neutral B-Rep;
- known limitation: published contacts may be missing or inaccurate.

### AutoMate

- part node: occurrence referencing a unique part id;
- positive: mate occurrence pair;
- relation type: one of the source mate types;
- interface: mate coordinate frames are available, but direct face/edge ids are
  not present in the published parquet/assembly schema;
- geometry: `.x_t` and, where conversion succeeded, `.step`.

### Linkify

- retains the Fusion occurrence-body convention;
- replaces/augments contacts with recomputed face indices, contact area,
  contact volume, and local contact point-cloud files;
- should be treated as a corrected interface derivative, with its own
  provenance and version, rather than silently overwriting source labels.
