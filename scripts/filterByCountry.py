import sys
import json
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python filter_spain.py /absolute/path/to/gdpr-export.json")
        sys.exit(2)

    input_path = Path(sys.argv[1])
    output_path = input_path.with_name("gdpr-export-spain.json")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("Error: input JSON must be a list of objects")
        sys.exit(3)

    # Filtrar solo los registros españoles
    spain_records = [
        rec for rec in data
        if rec.get("content", {}).get("decider_type") == "dpa"
        and rec.get("content", {}).get("jurisdiction") == "Spain"
        and rec.get("content", {}).get("jurisdiction_abbr") == "ES"
    ]

    # Guardar resultado
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(spain_records, f, ensure_ascii=False, indent=2)

    print(f"✅ Filtrados {len(spain_records)} casos de España")
    print(f"📄 Guardado en: {output_path}")

if __name__ == "__main__":
    main()
