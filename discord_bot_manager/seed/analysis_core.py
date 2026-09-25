"""Shared, Discord-independent analysis primitives for dream journal reports."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import math
import re
from statistics import mean, median
from typing import Any, Callable, Iterable, Sequence


DATE_FORMATS = ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y")
MISSING = None


def parse_report_date(value: Any) -> date | None:
    if not isinstance(value, str):
        return None
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(value.strip(), fmt).date()
        except ValueError:
            continue
    return None


def parse_year(value: Any, default: int) -> int | None:
    """Accept YYYY or YY and map the latter to the current century."""
    if value is None or str(value).strip() == "":
        return default
    try:
        year = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    if 0 <= year < 100:
        century = (default // 100) * 100
        year += century
        if year > default + 1:
            year -= 100
    return year if 1900 <= year <= 2200 else None


def finite_number(value: Any, minimum: float | None = None, maximum: float | None = None) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    if minimum is not None and result < minimum:
        return None
    if maximum is not None and result > maximum:
        return None
    return result


def nonnegative_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        return None
    return int(text)


def scale_10(value: Any) -> float | None:
    """Normalize decimal or fractional input to a 0-10 scale."""
    if value is None:
        return None
    text = str(value).strip()
    if "/" in text:
        pieces = text.split("/")
        if len(pieces) != 2:
            return None
        numerator = finite_number(pieces[0])
        denominator = finite_number(pieces[1])
        if numerator is None or denominator is None or denominator <= 0:
            return None
        if numerator < 0 or numerator > denominator:
            return None
        return numerator / denominator * 10
    return finite_number(text, 0, 10)


def first_present(report: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in report and report[key] not in (None, ""):
            return report[key]
    return None


def technique_parts(value: Any) -> tuple[str, ...]:
    """Lightly normalize comma-separated free-form techniques without a vocabulary."""
    if value is None:
        return ()
    parts: list[str] = []
    for part in str(value).split(","):
        normalized = re.sub(r"\s+", " ", part.strip().casefold())
        if normalized and normalized not in {"none", "no technique", "no tech"} and normalized not in parts:
            parts.append(normalized)
    return tuple(parts)


@dataclass(frozen=True)
class NightRecord:
    user_id: str
    report_date: date
    dream_count: int
    quality_10: float | None
    lucid_count: int
    wbtb_count: int
    sleep_hours: float | None
    focus_10: float | None
    journal_minutes: float | None
    technique_text: str
    technique_components: tuple[str, ...]
    notes: str

    @property
    def lucid_night(self) -> bool:
        return self.lucid_count > 0

    @property
    def wbtb_attempted(self) -> bool:
        return self.wbtb_count > 0

    @property
    def technique_combination(self) -> str:
        return ", ".join(sorted(self.technique_components))


@dataclass(frozen=True)
class DataIssue:
    kind: str
    message: str
    report_index: int | None = None
    report_date: date | None = None


@dataclass
class CanonicalDataset:
    user_id: str
    records: list[NightRecord] = field(default_factory=list)
    issues: list[DataIssue] = field(default_factory=list)
    duplicate_dates: set[date] = field(default_factory=set)
    raw_report_count: int = 0

    def for_year(self, year: int) -> list[NightRecord]:
        return [record for record in self.records if record.report_date.year == year]


def canonicalize_reports(user_id: str, raw_reports: Iterable[dict[str, Any]]) -> CanonicalDataset:
    raw_list = list(raw_reports)
    result = CanonicalDataset(user_id=str(user_id), raw_report_count=len(raw_list))
    date_counts: Counter[date] = Counter()
    parsed_dates: list[date | None] = []

    for report in raw_list:
        parsed = parse_report_date(report.get("date")) if isinstance(report, dict) else None
        parsed_dates.append(parsed)
        if parsed is not None:
            date_counts[parsed] += 1
    result.duplicate_dates = {day for day, count in date_counts.items() if count > 1}

    for index, (report, report_date) in enumerate(zip(raw_list, parsed_dates)):
        if not isinstance(report, dict):
            result.issues.append(DataIssue("invalid_report", "Report is not an object.", index))
            continue
        if report_date is None:
            result.issues.append(DataIssue("invalid_date", "Date is missing or invalid.", index))
            continue
        if report_date in result.duplicate_dates:
            result.issues.append(DataIssue(
                "duplicate_date",
                f"Multiple reports exist for {report_date:%d.%m.%Y}; excluded until corrected.",
                index,
                report_date,
            ))
            continue

        dreams = nonnegative_int(report.get("dreams"))
        lucids = nonnegative_int(report.get("lucid"))
        wbtb = nonnegative_int(report.get("wbtb"))
        if dreams is None or lucids is None or wbtb is None:
            result.issues.append(DataIssue(
                "invalid_required_metric",
                f"Dreams, lucid, or WBTB is invalid for {report_date:%d.%m.%Y}; excluded.",
                index,
                report_date,
            ))
            continue

        quality_raw = report.get("quality")
        quality = scale_10(quality_raw)
        if quality_raw not in (None, "") and quality is None:
            result.issues.append(DataIssue("invalid_quality", "Quality is invalid and treated as missing.", index, report_date))

        sleep_raw = first_present(report, "sleep_time", "sleep time")
        sleep = finite_number(sleep_raw, 0, 24)
        if sleep_raw not in (None, "") and sleep is None:
            result.issues.append(DataIssue("invalid_sleep", "Sleep time is invalid and treated as missing.", index, report_date))

        focus_raw = report.get("focus")
        focus = scale_10(focus_raw)
        if focus_raw not in (None, "") and focus is None:
            result.issues.append(DataIssue("invalid_focus", "Focus is invalid and treated as missing.", index, report_date))

        journal_raw = first_present(report, "journal_time", "journal time", "ournal_time")
        journal = finite_number(journal_raw, 0, 24 * 60)
        if journal_raw not in (None, "") and journal is None:
            result.issues.append(DataIssue("invalid_journal_time", "Journal time is invalid and treated as missing.", index, report_date))

        technique_raw = first_present(report, "technique", "techniques", "technicue")
        technique_text = str(technique_raw).strip() if technique_raw is not None else ""
        notes_raw = first_present(report, "notes", "note")

        result.records.append(NightRecord(
            user_id=str(user_id),
            report_date=report_date,
            dream_count=dreams,
            quality_10=quality,
            lucid_count=lucids,
            wbtb_count=wbtb,
            sleep_hours=sleep,
            focus_10=focus,
            journal_minutes=journal,
            technique_text=technique_text,
            technique_components=technique_parts(technique_raw),
            notes=str(notes_raw) if notes_raw is not None else "",
        ))

    result.records.sort(key=lambda record: record.report_date)
    return result


def metric_value(record: NightRecord, metric: str) -> float | None:
    aliases = {
        "dreams": float(record.dream_count),
        "recall": float(record.dream_count),
        "quality": record.quality_10,
        "lucid": float(record.lucid_count),
        "lucids": float(record.lucid_count),
        "lucid_rate": float(record.lucid_night),
        "wbtb": float(record.wbtb_count),
        "sleep": record.sleep_hours,
        "focus": record.focus_10,
        "journal": record.journal_minutes,
        "journaltime": record.journal_minutes,
    }
    return aliases.get(metric.casefold())


def records_between(records: Iterable[NightRecord], start: date, end: date) -> list[NightRecord]:
    return [record for record in records if start <= record.report_date <= end]


def adjacent_windows(records: Iterable[NightRecord], anchor: date) -> tuple[list[NightRecord], list[NightRecord]]:
    current_start = anchor - timedelta(days=6)
    previous_start = anchor - timedelta(days=13)
    return (
        records_between(records, current_start, anchor),
        records_between(records, previous_start, current_start - timedelta(days=1)),
    )


def period_coverage(records: Sequence[NightRecord], start: date, end: date) -> tuple[int, int, float]:
    eligible = max(0, (end - start).days + 1)
    reported = len({record.report_date for record in records if start <= record.report_date <= end})
    return reported, eligible, (reported / eligible if eligible else 0.0)


def _mean_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return mean(present) if present else None


def _median_present(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return median(present) if present else None


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float] | None:
    if total <= 0:
        return None
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z * z / (4 * total)) / total) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def summarize_records(records: Sequence[NightRecord]) -> dict[str, Any]:
    lucid_nights = sum(record.lucid_night for record in records)
    return {
        "n": len(records),
        "dream_total": sum(record.dream_count for record in records),
        "dream_mean": _mean_present(float(record.dream_count) for record in records),
        "dream_median": _median_present(float(record.dream_count) for record in records),
        "zero_recall_rate": _mean_present(float(record.dream_count == 0) for record in records),
        "quality_mean": _mean_present(record.quality_10 for record in records),
        "quality_median": _median_present(record.quality_10 for record in records),
        "lucid_count": sum(record.lucid_count for record in records),
        "lucid_nights": lucid_nights,
        "lucid_rate": lucid_nights / len(records) if records else None,
        "lucid_rate_interval": wilson_interval(lucid_nights, len(records)),
        "wbtb_mean": _mean_present(float(record.wbtb_count) for record in records),
        "sleep_mean": _mean_present(record.sleep_hours for record in records),
        "sleep_median": _median_present(record.sleep_hours for record in records),
        "sleep_n": sum(record.sleep_hours is not None for record in records),
        "focus_mean": _mean_present(record.focus_10 for record in records),
        "focus_n": sum(record.focus_10 is not None for record in records),
        "journal_mean": _mean_present(record.journal_minutes for record in records),
        "journal_median": _median_present(record.journal_minutes for record in records),
        "journal_total": sum(record.journal_minutes or 0 for record in records),
        "journal_n": sum(record.journal_minutes is not None for record in records),
    }


def daily_values(
    records: Sequence[NightRecord],
    start: date,
    end: date,
    metric: str,
) -> tuple[list[date], list[float]]:
    """Return reported dates and values; unreported dates are deliberately omitted."""
    values: dict[date, float] = {}
    for record in records:
        if start <= record.report_date <= end:
            value = metric_value(record, metric)
            if value is not None:
                values[record.report_date] = value
    ordered = sorted(values)
    return ordered, [values[day] for day in ordered]


def confidence_label(n: int, coverage: float | None = None) -> str:
    if n < 5:
        return "insufficient"
    if n < 10 or (coverage is not None and coverage < 0.5):
        return "preliminary"
    if n < 30 or (coverage is not None and coverage < 0.75):
        return "moderate"
    return "strong observational"


def percentile(values: Sequence[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def rank_values(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(indexed):
        end = index
        while end + 1 < len(indexed) and indexed[end + 1][1] == indexed[index][1]:
            end += 1
        average_rank = (index + end) / 2 + 1
        for cursor in range(index, end + 1):
            ranks[indexed[cursor][0]] = average_rank
        index = end + 1
    return ranks


def pearson(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    x_mean, y_mean = mean(x_values), mean(y_values)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    x_var = sum((x - x_mean) ** 2 for x in x_values)
    y_var = sum((y - y_mean) ** 2 for y in y_values)
    denominator = math.sqrt(x_var * y_var)
    return numerator / denominator if denominator else None


def spearman(x_values: Sequence[float], y_values: Sequence[float]) -> float | None:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return None
    return pearson(rank_values(x_values), rank_values(y_values))


def consecutive_pairs(records: Sequence[NightRecord]) -> list[tuple[NightRecord, NightRecord]]:
    ordered = sorted(records, key=lambda record: record.report_date)
    return [
        (current, following)
        for current, following in zip(ordered, ordered[1:])
        if following.report_date == current.report_date + timedelta(days=1)
    ]


def technique_groups(records: Sequence[NightRecord], components: bool = False) -> dict[str, list[NightRecord]]:
    groups: defaultdict[str, list[NightRecord]] = defaultdict(list)
    for record in records:
        names = record.technique_components if components else ((record.technique_combination,) if record.technique_combination else ())
        for name in names:
            groups[name].append(record)
    return dict(groups)


def shrunk_lucid_rate(successes: int, attempts: int, baseline_rate: float, prior_strength: float = 8.0) -> float:
    return (successes + baseline_rate * prior_strength) / (attempts + prior_strength)


def technique_effects(records: Sequence[NightRecord], components: bool = False) -> list[dict[str, Any]]:
    baseline_rate = sum(record.lucid_night for record in records) / len(records) if records else 0.0
    effects: list[dict[str, Any]] = []
    for name, group in technique_groups(records, components=components).items():
        successes = sum(record.lucid_night for record in group)
        raw_rate = successes / len(group)
        effects.append({
            "name": name,
            "n": len(group),
            "successes": successes,
            "raw_rate": raw_rate,
            "interval": wilson_interval(successes, len(group)),
            "shrunk_rate": shrunk_lucid_rate(successes, len(group), baseline_rate),
            "uplift": raw_rate - baseline_rate,
            "baseline_rate": baseline_rate,
            "records": group,
        })
    effects.sort(key=lambda item: (item["n"] >= 5, item["shrunk_rate"], item["n"]), reverse=True)
    return effects


def keyword_groups(records: Sequence[NightRecord], keyword: str) -> tuple[list[NightRecord], list[NightRecord]]:
    normalized = " ".join(keyword.casefold().split())
    if not normalized:
        return [], list(records)
    pattern = re.compile(rf"(?<!\w){re.escape(normalized)}(?!\w)", re.IGNORECASE)
    matching, other = [], []
    for record in records:
        normalized_notes = " ".join(record.notes.split())
        (matching if pattern.search(normalized_notes) else other).append(record)
    return matching, other


def lucid_gap_stats(records: Sequence[NightRecord], today: date) -> dict[str, Any] | None:
    lucid_dates = sorted({record.report_date for record in records if record.lucid_night})
    if len(lucid_dates) < 2:
        return None
    gaps = [(following - current).days for current, following in zip(lucid_dates, lucid_dates[1:])]
    recent = gaps[-3:] if len(gaps) >= 3 else gaps
    historical = gaps[:-3] if len(gaps) >= 6 else gaps
    historical_median = median(historical)
    recent_median = median(recent)
    tolerance = max(1.0, historical_median * 0.1)
    if len(gaps) < 3:
        trend = "insufficient evidence"
    elif recent_median < historical_median - tolerance:
        trend = "shorter"
    elif recent_median > historical_median + tolerance:
        trend = "longer"
    else:
        trend = "stable"
    q1, q3 = percentile([float(gap) for gap in gaps], 0.25), percentile([float(gap) for gap in gaps], 0.75)
    return {
        "lucid_dates": lucid_dates,
        "gaps": gaps,
        "n_lucid_nights": len(lucid_dates),
        "n_intervals": len(gaps),
        "mean": mean(gaps),
        "median": median(gaps),
        "q1": q1,
        "q3": q3,
        "minimum": min(gaps),
        "maximum": max(gaps),
        "current_gap": max(0, (today - lucid_dates[-1]).days),
        "recent_median": recent_median,
        "historical_median": historical_median,
        "trend": trend,
    }


def streak_lengths(days: Iterable[date]) -> tuple[int, int]:
    unique = sorted(set(days))
    if not unique:
        return 0, 0
    longest = run = 1
    for previous, current in zip(unique, unique[1:]):
        run = run + 1 if current == previous + timedelta(days=1) else 1
        longest = max(longest, run)
    current_run = 1
    for index in range(len(unique) - 1, 0, -1):
        if unique[index] == unique[index - 1] + timedelta(days=1):
            current_run += 1
        else:
            break
    return current_run, longest


def group_user_summaries(datasets: Iterable[CanonicalDataset], start: date, end: date) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for dataset in datasets:
        records = records_between(dataset.records, start, end)
        if not records:
            continue
        summary = summarize_records(records)
        reported, eligible, coverage = period_coverage(records, start, end)
        summary.update({"user_id": dataset.user_id, "reported": reported, "eligible": eligible, "coverage": coverage})
        summaries.append(summary)
    return summaries


def aggregate_user_metric(user_summaries: Sequence[dict[str, Any]], key: str) -> dict[str, float | int | None]:
    values = [summary[key] for summary in user_summaries if summary.get(key) is not None]
    return {
        "n_users": len(values),
        "mean": mean(values) if values else None,
        "median": median(values) if values else None,
        "q1": percentile(values, 0.25) if values else None,
        "q3": percentile(values, 0.75) if values else None,
    }


def binned_summary(
    records: Sequence[NightRecord],
    predictor: Callable[[NightRecord], float | None],
    outcome: Callable[[NightRecord], float | None],
    bins: Sequence[tuple[str, float, float]],
) -> list[dict[str, Any]]:
    result = []
    for label, lower, upper in bins:
        values = [
            outcome(record)
            for record in records
            if predictor(record) is not None and lower <= predictor(record) < upper and outcome(record) is not None
        ]
        result.append({"label": label, "n": len(values), "mean": mean(values) if values else None})
    return result
