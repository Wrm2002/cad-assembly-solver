"""Generate JSON Schema files from the versioned Pydantic contracts."""

from pathlib import Path

from contracts import write_json_schemas


if __name__ == "__main__":
    for schema_path in write_json_schemas(Path(__file__).parent / "schemas"):
        print(schema_path)
