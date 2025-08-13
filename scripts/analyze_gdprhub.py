import sys
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal, InvalidOperation


def parse_decimal(amount_text: str) -> Decimal:
	try:
		return Decimal(str(amount_text))
	except (InvalidOperation, ValueError, TypeError):
		return Decimal(0)


def normalize_article(ref: dict) -> str:
	article = str(ref.get("article", "?")).strip()
	paragraph = str(ref.get("paragraph", "")).strip()
	subparagraph = str(ref.get("subparagraph", "")).strip()
	if paragraph and subparagraph:
		return f"Art {article}({paragraph})({subparagraph})"
	if paragraph:
		return f"Art {article}({paragraph})"
	return f"Art {article}"


def parse_iso_date(date_value) -> datetime | None:
	if not date_value:
		return None
	if isinstance(date_value, dict) and "$date" in date_value:
		date_value = date_value["$date"]
	try:
		return datetime.fromisoformat(str(date_value).replace("Z", "+00:00"))
	except Exception:
		return None


def main() -> int:
	if len(sys.argv) != 2:
		print("Usage: uv run python analyze_gdprhub.py /absolute/path/to/gdprhub-export.json", file=sys.stderr)
		return 2

	json_path = sys.argv[1]
	with open(json_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	if not isinstance(data, list):
		print("Input JSON must be an array of records", file=sys.stderr)
		return 3

	total_records = len(data)
	decider_type_counts: Counter[str] = Counter()
	jurisdiction_counts: Counter[str] = Counter()
	decision_type_counts: Counter[str] = Counter()
	outcome_counts: Counter[str] = Counter()
	appeal_status_counts: Counter[str] = Counter()
	source_language_counts: Counter[str] = Counter()
	gdpr_article_counts: Counter[str] = Counter()
	jurisdiction_abbr_counts: Counter[str] = Counter()

	unique_party_names: set[str] = set()
	first_date: datetime | None = None
	last_date: datetime | None = None

	# Fines
	fines_by_currency: defaultdict[str, Decimal] = defaultdict(Decimal)
	num_records_with_fines = 0
	min_fine: tuple[Decimal, str] | None = None
	max_fine: tuple[Decimal, str] | None = None

	missing_summary = 0
	missing_facts = 0

	for rec in data:
		content = rec.get("content", {}) or {}
		wiki = rec.get("wiki", {}) or {}

		decider_type = content.get("decider_type")
		if decider_type:
			decider_type_counts[decider_type] += 1

		jurisdiction = content.get("jurisdiction")
		if jurisdiction:
			jurisdiction_counts[jurisdiction] += 1

		decision_type = content.get("decision_type")
		if decision_type:
			decision_type_counts[decision_type] += 1

		outcome = content.get("outcome")
		if outcome:
			outcome_counts[outcome] += 1

		abbr = content.get("jurisdiction_abbr")
		if abbr:
			jurisdiction_abbr_counts[abbr] += 1

		appeal_status = (content.get("appeal_to") or {}).get("status")
		if appeal_status:
			appeal_status_counts[appeal_status] += 1

		for src in content.get("sources", []) or []:
			lang = src.get("language")
			if lang:
				source_language_counts[lang] += 1

		for party in content.get("parties", []) or []:
			name = party.get("name")
			if name:
				unique_party_names.add(name)

		applied = (content.get("applied_laws") or {}).get("gdpr", []) or []
		for ref in applied:
			gdpr_article_counts[normalize_article(ref)] += 1

		text = content.get("text") or {}
		if not (text.get("summary") or "summary" in text):
			missing_summary += 1
		if not (text.get("facts") or "facts" in text):
			missing_facts += 1

		date_published = parse_iso_date(content.get("date_published"))
		if date_published:
			if first_date is None or date_published < first_date:
				first_date = date_published
			if last_date is None or date_published > last_date:
				last_date = date_published

		fine = content.get("fine") or {}
		amount_text = fine.get("amount")
		currency = fine.get("currency")
		amount = parse_decimal(amount_text) if amount_text is not None and currency else None
		if amount is not None and currency:
			num_records_with_fines += 1
			fines_by_currency[currency] += amount
			case_name = content.get("case_nr_name") or wiki.get("title") or "N/A"
			if min_fine is None or amount < min_fine[0]:
				min_fine = (amount, case_name)
			if max_fine is None or amount > max_fine[0]:
				max_fine = (amount, case_name)

	print("GDPRHub export analysis")
	print("======================")
	print(f"Total records: {total_records}")
	if first_date or last_date:
		fd = first_date.date().isoformat() if first_date else "?"
		ld = last_date.date().isoformat() if last_date else "?"
		print(f"Date range: {fd} … {ld}")

	def print_top(counter: Counter[str], title: str, limit: int = 10) -> None:
		if not counter:
			return
		print(f"\n{title} (top {limit}):")
		for key, count in counter.most_common(limit):
			print(f"- {key}: {count}")

	print_top(jurisdiction_counts, "Jurisdictions")
	print_top(decider_type_counts, "Decider types")
	print_top(decision_type_counts, "Decision types")
	print_top(outcome_counts, "Outcomes")
	print_top(jurisdiction_abbr_counts, "Jurisdiction abbreviations")
	print_top(source_language_counts, "Source languages")
	print_top(gdpr_article_counts, "GDPR article references")

	print("\nFines:")
	print(f"- Records with fines: {num_records_with_fines}")
	if fines_by_currency:
		for cur, total in sorted(fines_by_currency.items()):
			print(f"- Total fines in {cur}: {total}")
	if min_fine:
		print(f"- Min fine: {min_fine[0]} ({min_fine[1]})")
	if max_fine:
		print(f"- Max fine: {max_fine[0]} ({max_fine[1]})")

	print("\nData completeness:")
	print(f"- Missing summaries: {missing_summary} ({missing_summary/total_records:.1%})")
	print(f"- Missing facts: {missing_facts} ({missing_facts/total_records:.1%})")
	print(f"- Unique party names: {len(unique_party_names)}")

	return 0


if __name__ == "__main__":
	sys.exit(main())


