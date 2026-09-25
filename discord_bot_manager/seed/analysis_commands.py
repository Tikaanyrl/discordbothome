"""Discord command layer for the normalized dream-journal analysis system."""

from __future__ import annotations

import asyncio
import calendar
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
import math
from statistics import mean, median
from typing import Any, Callable

import discord
from discord.ext import commands
import matplotlib.pyplot as plt
import numpy as np

import analysis_core as ac


METRIC_LABELS = {
    "dreams": "Dream recall",
    "quality": "Quality (/10)",
    "lucid": "Lucid dreams",
    "lucid_rate": "Lucid-night rate",
    "wbtb": "WBTB attempts",
    "sleep": "Sleep (hours)",
    "focus": "Focus (/10)",
    "journal": "Journal time (minutes)",
}


def fmt(value: float | None, digits: int = 1, suffix: str = "") -> str:
    return "—" if value is None else f"{value:.{digits}f}{suffix}"


def pct(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value * 100:.{digits}f}%"


def change(current: float | None, previous: float | None, digits: int = 1) -> str:
    if current is None or previous is None:
        return "—"
    return f"{current - previous:+.{digits}f}"


def month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, calendar.monthrange(year, month)[1])


class AnalysisCommands(commands.Cog):
    def __init__(
        self,
        bot: commands.Bot,
        active_year: int,
        source_provider: Callable[[int], dict[str, list[dict[str, Any]]] | None],
        local_today: Callable[[str], date],
        output_path: Callable[..., str],
        send_long: Callable[..., Any],
        send_file: Callable[..., Any],
    ):
        self.bot = bot
        self.active_year = active_year
        self.source_provider = source_provider
        self.local_today = local_today
        self.output_path = output_path
        self.send_long = send_long
        self.send_file = send_file

    async def _target_year(
        self,
        ctx: commands.Context,
        first: str | None = None,
        second: str | None = None,
        fixed_year: int | None = None,
    ) -> tuple[discord.User, int] | None:
        target = ctx.author
        year_token = second
        if first:
            possible_year = ac.parse_year(first, self.active_year)
            if possible_year is not None and (first.isdigit() and len(first) in (2, 4)):
                year_token = first
            else:
                try:
                    target = await commands.UserConverter().convert(ctx, first)
                except commands.BadArgument:
                    await ctx.send("Could not find that user. Mention a user or provide a two- or four-digit year.")
                    return None
        year = fixed_year if fixed_year is not None else ac.parse_year(year_token, self.active_year)
        if year is None:
            await ctx.send("Invalid year. Use a value such as `25` or `2025`.")
            return None
        if self.source_provider(year) is None:
            await ctx.send(f"No data file is configured for {year}.")
            return None
        return target, year

    def _dataset(self, user_id: str, year: int) -> ac.CanonicalDataset:
        source = self.source_provider(year) or {}
        return ac.canonicalize_reports(user_id, source.get(str(user_id), []))

    def _combined_dataset(self, user_id: str, years: set[int]) -> ac.CanonicalDataset:
        raw: list[dict[str, Any]] = []
        for year in sorted(years):
            source = self.source_provider(year)
            if source is not None:
                raw.extend(source.get(str(user_id), []))
        return ac.canonicalize_reports(user_id, raw)

    def _all_datasets(self, year: int) -> list[ac.CanonicalDataset]:
        source = self.source_provider(year) or {}
        return [ac.canonicalize_reports(user_id, reports) for user_id, reports in source.items()]

    def _all_datasets_between(self, start: date, end: date) -> list[ac.CanonicalDataset]:
        years = set(range(start.year, end.year + 1)); user_ids: set[str] = set()
        for year in years:
            source = self.source_provider(year)
            if source is not None:
                user_ids.update(source)
        return [self._combined_dataset(user_id, years) for user_id in user_ids]

    async def _require_records(
        self,
        ctx: commands.Context,
        target: discord.User,
        year: int,
    ) -> tuple[ac.CanonicalDataset, list[ac.NightRecord]] | None:
        dataset = self._dataset(str(target.id), year)
        records = dataset.for_year(year)
        if not records:
            await ctx.send(f"No usable reports found for {target.display_name} in {year}.")
            return None
        return dataset, records

    async def _send_figure(self, ctx: commands.Context, fig: plt.Figure, prefix: str) -> None:
        path = self.output_path(prefix)
        try:
            await asyncio.to_thread(fig.savefig, path, dpi=140, bbox_inches="tight")
        finally:
            plt.close(fig)
        await self.send_file(ctx, path)

    @staticmethod
    def _window_arrays(records: list[ac.NightRecord], start: date, metric: str) -> list[float]:
        lookup = {record.report_date: ac.metric_value(record, metric) for record in records}
        return [lookup.get(start + timedelta(days=index), np.nan) for index in range(7)]

    async def _trend(self, ctx: commands.Context, target: discord.User, year: int) -> None:
        required = await self._require_records(ctx, target, year)
        if not required:
            return
        dataset, records = required
        anchor = min(self.local_today(str(target.id)), date(year, 12, 31)) if year == self.active_year else date(year, 12, 31)
        if year == self.active_year:
            baseline_start = anchor - timedelta(days=34)
            dataset = self._combined_dataset(str(target.id), set(range(baseline_start.year, anchor.year + 1)))
            records = dataset.records
        current, previous = ac.adjacent_windows(records, anchor)
        if not current and not previous:
            await ctx.send("No usable reports were found in the two adjacent seven-day periods.")
            return
        current_start, previous_start = anchor - timedelta(days=6), anchor - timedelta(days=13)
        current_summary, previous_summary = ac.summarize_records(current), ac.summarize_records(previous)
        current_cov = ac.period_coverage(current, current_start, anchor)
        previous_cov = ac.period_coverage(previous, previous_start, current_start - timedelta(days=1))
        baseline_start = anchor - timedelta(days=34)
        baseline_records = ac.records_between(records, baseline_start, anchor - timedelta(days=7))
        baseline = ac.summarize_records(baseline_records)

        text = (
            f"📊 **Trend for {target.display_name}** — {current_start:%d.%m}–{anchor:%d.%m.%Y}\n"
            f"Coverage: **{current_cov[0]}/7** nights (previous: **{previous_cov[0]}/7**)\n\n"
            f"• Recall/night: **{fmt(current_summary['dream_mean'])}** ({change(current_summary['dream_mean'], previous_summary['dream_mean'])}; 28-day baseline {fmt(baseline['dream_mean'])})\n"
            f"• Quality: **{fmt(current_summary['quality_mean'])}/10** ({change(current_summary['quality_mean'], previous_summary['quality_mean'])}; baseline {fmt(baseline['quality_mean'])})\n"
            f"• Lucid nights: **{current_summary['lucid_nights']}/{current_summary['n']}** ({pct(current_summary['lucid_rate'])}; baseline {pct(baseline['lucid_rate'])})\n"
            f"• Total lucid dreams: **{current_summary['lucid_count']}** ({current_summary['lucid_count'] - previous_summary['lucid_count']:+d})\n"
            f"• WBTB/night: **{fmt(current_summary['wbtb_mean'])}** ({change(current_summary['wbtb_mean'], previous_summary['wbtb_mean'])})\n"
            f"• Sleep: **{fmt(current_summary['sleep_mean'])} h** | Focus: **{fmt(current_summary['focus_mean'])}/10** | Journal: **{fmt(current_summary['journal_mean'])} min**\n\n"
            f"Evidence: **{ac.confidence_label(current_summary['n'] + previous_summary['n'], (current_cov[2] + previous_cov[2]) / 2)}**. Missing nights are not counted as zeroes."
        )
        if dataset.duplicate_dates:
            text += f"\n⚠️ {len(dataset.duplicate_dates)} duplicate date(s) were excluded."
        await self.send_long(ctx, text)

        metrics = ("dreams", "quality", "lucid", "wbtb")
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        x = np.arange(7)
        labels = [(current_start + timedelta(days=i)).strftime("%a") for i in range(7)]
        for ax, metric in zip(axes.flat, metrics):
            ax.plot(x, self._window_arrays(current, current_start, metric), marker="o", label="Current 7 days")
            ax.plot(x, self._window_arrays(previous, previous_start, metric), marker="o", linestyle="--", label="Previous 7 days")
            baseline_value = baseline["dream_mean" if metric == "dreams" else "quality_mean" if metric == "quality" else "wbtb_mean" if metric == "wbtb" else "lucid_rate"]
            if baseline_value is not None:
                if metric == "lucid":
                    baseline_value = baseline["lucid_count"] / baseline["n"] if baseline["n"] else None
                if baseline_value is not None:
                    ax.axhline(baseline_value, color="gray", linestyle=":", label="28-day baseline")
            ax.set_title(METRIC_LABELS[metric])
            ax.set_xticks(x, labels)
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
        fig.suptitle(f"Recent trend — {target.display_name}")
        fig.tight_layout()
        await self._send_figure(ctx, fig, f"trend_{target.id}_{year}")

    @commands.command(name="trend")
    async def trend(self, ctx: commands.Context) -> None:
        await self._trend(ctx, ctx.author, self.active_year)

    @commands.command(name="score")
    async def score(self, ctx: commands.Context) -> None:
        dataset = self._dataset(str(ctx.author.id), self.active_year)
        records = dataset.for_year(self.active_year)
        anchor = self.local_today(str(ctx.author.id))
        current, previous = ac.adjacent_windows(records, anchor)
        current_summary, previous_summary = ac.summarize_records(current), ac.summarize_records(previous)
        await ctx.send(
            f"**Recall score** — current 7 days: **{current_summary['dream_total']}** dreams over {current_summary['n']} reported nights "
            f"({fmt(current_summary['dream_mean'])}/night); previous: **{previous_summary['dream_total']}** over {previous_summary['n']} "
            f"({fmt(previous_summary['dream_mean'])}/night). Full context: `!trend`."
        )

    async def _overview(self, ctx: commands.Context, target: discord.User, year: int) -> None:
        required = await self._require_records(ctx, target, year)
        if not required:
            return
        dataset, records = required
        summary = ac.summarize_records(records)
        start, end = date(year, 1, 1), min(date(year, 12, 31), self.local_today(str(target.id))) if year == self.active_year else date(year, 12, 31)
        coverage = ac.period_coverage(records, start, end)
        effects = ac.technique_effects(records, components=True)
        supported = [item for item in effects if item["n"] >= 5]
        gaps = ac.lucid_gap_stats(records, end)
        current_streak, longest_streak = ac.streak_lengths(record.report_date for record in records)
        top_frequency = Counter(component for record in records for component in record.technique_components).most_common(1)
        best = supported[0] if supported else None
        text = (
            f"🌟 **Dream Journal Overview — {target.display_name}, {year}**\n"
            f"Coverage: **{coverage[0]}/{coverage[1]} nights ({coverage[2] * 100:.1f}%)**\n\n"
            f"• Dreams: **{summary['dream_total']} total**, {fmt(summary['dream_mean'])}/reported night (median {fmt(summary['dream_median'])})\n"
            f"• Quality: **{fmt(summary['quality_mean'])}/10** (median {fmt(summary['quality_median'])})\n"
            f"• Lucidity: **{summary['lucid_nights']} lucid nights ({pct(summary['lucid_rate'])})**, {summary['lucid_count']} lucid dreams\n"
            f"• Reporting streak: **{current_streak} current**, **{longest_streak} longest**\n"
        )
        if top_frequency:
            text += f"• Most used technique: **{top_frequency[0][0]}** ({top_frequency[0][1]} nights)\n"
        if best:
            text += f"• Best-supported technique association: **{best['name']}**, {pct(best['raw_rate'])} lucid nights over {best['n']} uses (baseline {pct(best['baseline_rate'])})\n"
        if gaps:
            text += f"• Lucid gaps: median **{gaps['median']:.1f} days**, current **{gaps['current_gap']} days**, recent pattern **{gaps['trend']}**\n"
        text += f"\nEvidence: **{ac.confidence_label(summary['n'], coverage[2])}**."
        if dataset.issues:
            text += f" Data audit found {len(dataset.issues)} issue(s); use `!data_quality {year % 100:02d}`."
        await self.send_long(ctx, text)

        monthly: defaultdict[int, list[ac.NightRecord]] = defaultdict(list)
        weekday: defaultdict[int, list[ac.NightRecord]] = defaultdict(list)
        for record in records:
            monthly[record.report_date.month].append(record)
            weekday[record.report_date.weekday()].append(record)
        months = sorted(monthly)
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes[0, 0].plot(months, [ac.summarize_records(monthly[m])["dream_mean"] for m in months], marker="o")
        axes[0, 0].set_title("Monthly recall/night")
        axes[0, 1].plot(months, [ac.summarize_records(monthly[m])["lucid_rate"] * 100 for m in months], marker="o", color="green")
        axes[0, 1].set_title("Monthly lucid-night rate (%)")
        names = [item["name"] for item in supported[:6]][::-1]
        rates = [item["shrunk_rate"] * 100 for item in supported[:6]][::-1]
        axes[1, 0].barh(names, rates, color="tab:purple")
        axes[1, 0].set_title("Technique rates (small-sample adjusted)")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        axes[1, 1].bar(days, [ac.summarize_records(weekday[d])["lucid_rate"] * 100 if weekday[d] else 0 for d in range(7)], color="tab:green")
        axes[1, 1].set_title("Lucid-night rate by weekday")
        for ax in axes.flat:
            ax.grid(alpha=.2, axis="y")
        fig.suptitle(f"Overview — {target.display_name}, {year}")
        fig.tight_layout()
        await self._send_figure(ctx, fig, f"overview_{target.id}_{year}")

    @commands.command(name="overview")
    async def overview(self, ctx: commands.Context, first: str | None = None, second: str | None = None) -> None:
        parsed = await self._target_year(ctx, first, second)
        if parsed:
            await self._overview(ctx, *parsed)

    @commands.command(name="overview_25", aliases=["overview_previous"])
    async def overview_25(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        await self._overview(ctx, user or ctx.author, 2025)

    @commands.command(name="baseline")
    async def baseline(self, ctx: commands.Context, metric: str | None = None) -> None:
        anchor = self.local_today(str(ctx.author.id))
        records = self._combined_dataset(str(ctx.author.id), set(range((anchor - timedelta(days=89)).year, anchor.year + 1))).records
        if not records:
            await ctx.send("No usable reports found.")
            return
        windows = (7, 28, 90)
        summaries = {days: ac.summarize_records(ac.records_between(records, anchor - timedelta(days=days - 1), anchor)) for days in windows}
        selected = metric.casefold() if metric else None
        allowed = {"dreams", "quality", "lucid", "wbtb", "sleep", "focus", "journal"}
        if selected and selected not in allowed:
            await ctx.send(f"Unknown metric. Choose: {', '.join(sorted(allowed))}.")
            return
        metric_keys = {
            "dreams": "dream_mean", "quality": "quality_mean", "lucid": "lucid_rate",
            "wbtb": "wbtb_mean", "sleep": "sleep_mean", "focus": "focus_mean", "journal": "journal_mean",
        }
        shown = [selected] if selected else list(metric_keys)
        lines = ["**Personal baseline**"]
        for name in shown:
            key = metric_keys[name]
            values = [summaries[days][key] for days in windows]
            formatter = pct if name == "lucid" else fmt
            lines.append(f"• {METRIC_LABELS.get(name, name.title())}: 7d **{formatter(values[0])}** | 28d **{formatter(values[1])}** | 90d **{formatter(values[2])}**")
        lines.append("_Reported nights only; missing nights are not zero._")
        await self.send_long(ctx, "\n".join(lines))
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(shown)); width = .24
        for offset, days in enumerate(windows):
            values = [summaries[days][metric_keys[name]] or 0 for name in shown]
            if "lucid" in shown:
                values = [value * 100 if name == "lucid" else value for name, value in zip(shown, values)]
            ax.bar(x + (offset - 1) * width, values, width, label=f"{days} days")
        ax.set_xticks(x, [METRIC_LABELS.get(name, name).split(" (")[0] for name in shown], rotation=25)
        ax.set_title("Personal rolling baselines")
        ax.legend(); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"baseline_{ctx.author.id}")

    @commands.command(name="month")
    async def month(self, ctx: commands.Context, month_token: str | None = None) -> None:
        today = self.local_today(str(ctx.author.id))
        if month_token:
            match = re_month(month_token, today.year)
            if match is None:
                await ctx.send("Use `MM.YY` or `MM.YYYY`.")
                return
            month, year = match
        else:
            month, year = today.month, today.year
        if self.source_provider(year) is None:
            await ctx.send(f"No data file is configured for {year}.")
            return
        records = self._dataset(str(ctx.author.id), year).for_year(year)
        start, end = month_bounds(year, month)
        selected = ac.records_between(records, start, end)
        if not selected:
            await ctx.send(f"No usable reports found for {start:%B %Y}.")
            return
        summary = ac.summarize_records(selected)
        coverage_end = min(end, today) if year == today.year and month == today.month else end
        coverage = ac.period_coverage(selected, start, coverage_end)
        previous_end = start - timedelta(days=1); previous_start = date(previous_end.year, previous_end.month, 1)
        previous = ac.summarize_records(ac.records_between(self._dataset(str(ctx.author.id), previous_end.year).records if self.source_provider(previous_end.year) else [], previous_start, previous_end))
        techniques = Counter(component for record in selected for component in record.technique_components)
        text = (
            f"📅 **{start:%B %Y}** — coverage **{coverage[0]}/{coverage[1]} ({coverage[2] * 100:.1f}%)**\n"
            f"• Recall: **{summary['dream_total']} total**, {fmt(summary['dream_mean'])}/night ({change(summary['dream_mean'], previous['dream_mean'])} vs previous month)\n"
            f"• Quality: **{fmt(summary['quality_mean'])}/10** (median {fmt(summary['quality_median'])})\n"
            f"• Lucidity: **{summary['lucid_nights']} lucid nights ({pct(summary['lucid_rate'])})**, {summary['lucid_count']} lucid dreams\n"
            f"• Sleep: **{fmt(summary['sleep_mean'])} h** ({summary['sleep_n']}/{summary['n']} nights) | Focus: **{fmt(summary['focus_mean'])}/10**\n"
            f"• WBTB: **{fmt(summary['wbtb_mean'])}/night** | Journal: **{fmt(summary['journal_mean'])} min/session**\n"
            f"• Most used: **{techniques.most_common(1)[0][0] if techniques else 'none'}** ({techniques.most_common(1)[0][1] if techniques else 0} nights)"
        )
        await self.send_long(ctx, text)
        weeks: defaultdict[int, list[ac.NightRecord]] = defaultdict(list)
        for record in selected:
            weeks[(record.report_date.day - 1) // 7 + 1].append(record)
        labels = sorted(weeks)
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        for ax, key, title in zip(axes.flat, ("dream_mean", "quality_mean", "lucid_rate", "wbtb_mean"), ("Recall/night", "Quality", "Lucid-night rate", "WBTB/night")):
            values = [ac.summarize_records(weeks[w])[key] or 0 for w in labels]
            if key == "lucid_rate": values = [value * 100 for value in values]
            ax.plot(labels, values, marker="o"); ax.set_title(title); ax.set_xlabel("Week of month"); ax.grid(alpha=.25)
        fig.suptitle(f"Weekly view — {start:%B %Y}"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"month_{ctx.author.id}_{year}_{month}")

    async def _group(self, ctx: commands.Context, start: date, end: date, title: str) -> None:
        datasets = self._all_datasets_between(start, end)
        summaries = ac.group_user_summaries(datasets, start, end)
        if not summaries:
            await ctx.send("No usable group reports found for that period.")
            return
        metrics = ("dream_mean", "quality_mean", "lucid_rate", "wbtb_mean")
        aggregates = {key: ac.aggregate_user_metric(summaries, key) for key in metrics}
        total_reports = sum(item["n"] for item in summaries)
        text = (
            f"🌍 **{title}**\nParticipants: **{len(summaries)} users**, **{total_reports} reports**\n\n"
            f"• Median user recall: **{fmt(aggregates['dream_mean']['median'])}/night** (mean {fmt(aggregates['dream_mean']['mean'])})\n"
            f"• Median user quality: **{fmt(aggregates['quality_mean']['median'])}/10**\n"
            f"• Median user lucid-night rate: **{pct(aggregates['lucid_rate']['median'])}**\n"
            f"• Median user WBTB: **{fmt(aggregates['wbtb_mean']['median'])}/night**\n"
            "_Each user is summarized first, so frequent reporters do not dominate the averages._"
        )
        await self.send_long(ctx, text)
        fig, ax = plt.subplots(figsize=(9, 6))
        names = ["Recall", "Quality", "Lucid rate %", "WBTB"]
        values = [aggregates[key]["median"] or 0 for key in metrics]
        values[2] *= 100
        ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:green", "tab:purple"])
        ax.set_title(title); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"group_{start}_{end}")

    @commands.command(name="group")
    async def group(self, ctx: commands.Context) -> None:
        end = date.today(); start = end - timedelta(days=6)
        await self._group(ctx, start, end, "Group — latest 7 days")

    @commands.command(name="month_group")
    async def month_group(self, ctx: commands.Context, month_token: str | None = None) -> None:
        today = date.today(); parsed = re_month(month_token, today.year) if month_token else (today.month, today.year)
        if parsed is None:
            await ctx.send("Use `MM.YY` or `MM.YYYY`."); return
        month, year = parsed
        if self.source_provider(year) is None:
            await ctx.send(f"No data file is configured for {year}."); return
        start, end = month_bounds(year, month)
        await self._group(ctx, start, end, f"Group — {start:%B %Y}")

    @commands.command(name="heatmap")
    async def heatmap(self, ctx: commands.Context, year_token: str | None = None, metric: str = "lucid") -> None:
        year = ac.parse_year(year_token, self.active_year)
        if year is None or self.source_provider(year) is None:
            await ctx.send("Invalid or unavailable year."); return
        records = self._dataset(str(ctx.author.id), year).for_year(year)
        allowed = {"lucid", "dreams", "quality", "wbtb", "sleep", "focus", "journal"}
        metric = metric.casefold()
        if metric not in allowed:
            await ctx.send(f"Unknown metric. Choose: {', '.join(sorted(allowed))}."); return
        first = date(year, 1, 1); grid_start = first - timedelta(days=first.weekday())
        last = date(year, 12, 31); weeks = ((last - grid_start).days // 7) + 1
        matrix = np.full((7, weeks), np.nan)
        for record in records:
            value = ac.metric_value(record, metric)
            if value is not None:
                offset = (record.report_date - grid_start).days
                matrix[offset % 7, offset // 7] = value
        fig, ax = plt.subplots(figsize=(20, 5))
        cmap = plt.get_cmap("Greens").copy(); cmap.set_bad("#e5e7eb")
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap)
        ax.set_yticks(range(7), ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        month_ticks, month_labels = [], []
        for month in range(1, 13):
            month_start = date(year, month, 1)
            month_ticks.append((month_start - grid_start).days // 7); month_labels.append(month_start.strftime("%b"))
        ax.set_xticks(month_ticks, month_labels)
        ax.set_title(f"{METRIC_LABELS.get(metric, metric.title())} heatmap — {year} (gray = unreported/missing)")
        fig.colorbar(image, ax=ax, label=METRIC_LABELS.get(metric, metric.title()))
        fig.tight_layout(); await self._send_figure(ctx, fig, f"heatmap_{ctx.author.id}_{year}_{metric}")

    @commands.command(name="day_of_week")
    async def day_of_week(self, ctx: commands.Context, metric: str | None = None) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        if not records: await ctx.send("No usable reports found."); return
        grouped = {day: [record for record in records if record.report_date.weekday() == day] for day in range(7)}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        configs = {"dreams": "dream_mean", "quality": "quality_mean", "lucid": "lucid_rate", "focus": "focus_mean"}
        if metric:
            metric = metric.casefold()
            if metric not in configs: await ctx.send("Choose dreams, quality, lucid, or focus."); return
            selected = [metric]
        else: selected = list(configs)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8)); axes_flat = list(axes.flat)
        lines = ["**Day-of-week analysis**"]
        for index, name in enumerate(selected):
            values = [ac.summarize_records(grouped[day])[configs[name]] or 0 for day in range(7)]
            if name == "lucid": values = [value * 100 for value in values]
            ax = axes_flat[index]; ax.bar(days, values); ax.set_title(METRIC_LABELS[name]); ax.grid(alpha=.2, axis="y")
            best = int(np.argmax(values)); lines.append(f"• {METRIC_LABELS[name]}: highest observed on **{days[best]}** ({fmt(values[best])}, n={len(grouped[best])})")
        for ax in axes_flat[len(selected):]: ax.axis("off")
        lines.append("_Weekday differences are observational and may reflect routines or reporting patterns._")
        await self.send_long(ctx, "\n".join(lines)); fig.tight_layout()
        await self._send_figure(ctx, fig, f"weekday_{ctx.author.id}")

    @commands.command(name="journaltime", aliases=["journal"])
    async def journaltime(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        usable = [record for record in records if record.journal_minutes is not None]
        if not usable: await ctx.send("No usable journal-time data found."); return
        summary = ac.summarize_records(records)
        current, longest = ac.streak_lengths(record.report_date for record in usable if (record.journal_minutes or 0) > 0)
        text = (
            f"📓 **Journal time**\n• Total: **{summary['journal_total']:.1f} minutes** ({summary['journal_total']/60:.1f} hours)\n"
            f"• Sessions recorded: **{summary['journal_n']}/{summary['n']}**\n• Mean: **{fmt(summary['journal_mean'])} min** | Median: **{fmt(summary['journal_median'])} min**\n"
            f"• Positive-time streak: **{current} current**, **{longest} longest**\n_Zero, missing, and unreported nights remain distinct._"
        )
        await self.send_long(ctx, text)
        months: defaultdict[tuple[int, int], list[float]] = defaultdict(list)
        for record in usable: months[(record.report_date.year, record.report_date.month)].append(record.journal_minutes or 0)
        keys = sorted(months)[-12:]; labels = [f"{m:02d}.{str(y)[2:]}" for y, m in keys]
        fig, ax = plt.subplots(figsize=(11, 5)); ax.bar(labels, [median(months[key]) for key in keys])
        ax.set_title("Median journal minutes by month"); ax.tick_params(axis="x", rotation=35); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"journaltime_{ctx.author.id}")

    async def _journal_impact_user(self, ctx: commands.Context, target: discord.User, year: int) -> None:
        records = self._dataset(str(target.id), year).for_year(year)
        same = [record for record in records if record.journal_minutes is not None]
        pairs = [(current, following) for current, following in ac.consecutive_pairs(records) if current.journal_minutes is not None]
        if len(same) < 2: await ctx.send("At least two journal-time records are required."); return
        datasets = [
            ([r.journal_minutes for r in same], [float(r.dream_count) for r in same], "Same-report recall"),
            ([r.journal_minutes for r in same if r.quality_10 is not None], [r.quality_10 for r in same if r.quality_10 is not None], "Same-report quality"),
            ([a.journal_minutes for a, _ in pairs], [float(b.dream_count) for _, b in pairs], "Next-night recall"),
            ([a.journal_minutes for a, b in pairs if b.quality_10 is not None], [b.quality_10 for a, b in pairs if b.quality_10 is not None], "Next-night quality"),
        ]
        lines = [f"**Journal-time impact — {target.display_name}**"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for ax, (x, y, title) in zip(axes.flat, datasets):
            rho = ac.spearman(x, y)
            lines.append(f"• {title}: ρ **{fmt(rho, 2)}**, n={len(x)}" + (" (preliminary)" if len(x) < 10 else ""))
            ax.scatter(x, y, alpha=.65); ax.set_title(f"{title} (n={len(x)}, ρ={fmt(rho, 2)})"); ax.set_xlabel("Journal minutes"); ax.grid(alpha=.2)
        lines.append("_Same-report relationships may reflect that remembering more creates more material to journal. Next-night pairs use only consecutive dates._")
        await self.send_long(ctx, "\n".join(lines)); fig.tight_layout()
        await self._send_figure(ctx, fig, f"journal_impact_{target.id}_{year}")

    @commands.command(name="journal_impact", aliases=["journaltime_impact"])
    async def journal_impact(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        await self._journal_impact_user(ctx, user or ctx.author, self.active_year)

    @commands.command(name="journal_impact_all", aliases=["journaltime_all"])
    async def journal_impact_all(self, ctx: commands.Context) -> None:
        effects = []
        pair_count = 0
        for dataset in self._all_datasets(self.active_year):
            pairs = [(a, b) for a, b in ac.consecutive_pairs(dataset.records) if a.journal_minutes is not None]
            x = [a.journal_minutes for a, _ in pairs]; y = [float(b.dream_count) for _, b in pairs]
            rho = ac.spearman(x, y)
            if rho is not None and len(pairs) >= 3:
                effects.append(rho); pair_count += len(pairs)
        if not effects: await ctx.send("Not enough within-user consecutive pairs."); return
        await ctx.send(f"**Group journal impact** — median within-user next-night recall correlation: **ρ={median(effects):.2f}**, {len(effects)} users, {pair_count} pairs. Users are analyzed separately before pooling.")
        fig, ax = plt.subplots(figsize=(8, 5)); ax.hist(effects, bins=np.linspace(-1, 1, 11), color="tab:blue", alpha=.8); ax.axvline(median(effects), color="red", linestyle="--"); ax.set_title("Within-user journal-time effects"); ax.set_xlabel("Spearman ρ"); fig.tight_layout()
        await self._send_figure(ctx, fig, "journal_impact_all")

    @commands.command(name="correlate")
    async def correlate(self, ctx: commands.Context, *, keyword: str) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        matching, other = ac.keyword_groups(records, keyword)
        if not matching: await ctx.send(f"No whole-word or phrase matches found for `{keyword}`."); return
        left, right = ac.summarize_records(matching), ac.summarize_records(other)
        text = (
            f"**Keyword comparison: `{keyword}`**\n"
            f"• Matching notes (n={left['n']}): recall **{fmt(left['dream_mean'])}**, quality **{fmt(left['quality_mean'])}**, lucid rate **{pct(left['lucid_rate'])}**\n"
            f"• Other notes (n={right['n']}): recall **{fmt(right['dream_mean'])}**, quality **{fmt(right['quality_mean'])}**, lucid rate **{pct(right['lucid_rate'])}**\n"
            f"• Differences: recall **{change(left['dream_mean'], right['dream_mean'])}**, quality **{change(left['quality_mean'], right['quality_mean'])}**, lucid rate **{pct((left['lucid_rate'] or 0) - (right['lucid_rate'] or 0))}**\n"
            f"Evidence: **{ac.confidence_label(min(left['n'], right['n']))}**. Keyword presence may coincide with sleep, techniques, motivation, or time trends."
        )
        await self.send_long(ctx, text)
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        labels = [f"Contains\n{keyword}", "Other"]
        for ax, key, title, multiplier in zip(
            axes,
            ("dream_mean", "quality_mean", "lucid_rate"),
            ("Recall/night", "Quality", "Lucid-night rate (%)"),
            (1, 1, 100),
        ):
            ax.bar(labels, [(left[key] or 0) * multiplier, (right[key] or 0) * multiplier], color=["tab:blue", "gray"])
            ax.set_title(title); ax.grid(alpha=.2, axis="y")
        fig.suptitle(f"Keyword comparison — {keyword}"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"keyword_{ctx.author.id}")

    async def _effectiveness(self, ctx: commands.Context, technique: str | None, year: int) -> None:
        records = self._dataset(str(ctx.author.id), year).for_year(year)
        if not records: await ctx.send(f"No usable reports found for {year}."); return
        component_effects = ac.technique_effects(records, components=True)
        combination_effects = ac.technique_effects(records, components=False)
        if technique:
            normalized = ac.technique_parts(technique)
            if len(normalized) > 1:
                name = ", ".join(sorted(normalized))
                effects = [item for item in combination_effects if item["name"] == name]
            else:
                name = normalized[0] if normalized else technique.casefold().strip()
                effects = [item for item in component_effects if item["name"] == name]
            if not effects: await ctx.send(f"No uses found for `{technique}`."); return
            shown = effects
        else:
            shown = component_effects[:10]
        lines = [f"**Technique effectiveness — {year}**", f"Personal baseline lucid-night rate: **{pct(ac.summarize_records(records)['lucid_rate'])}**"]
        if not technique:
            lines.append("\n**Individual technique components**")
        for index, item in enumerate(shown, 1):
            status = "established" if item["n"] >= 10 else "preliminary" if item["n"] >= 5 else "insufficient"
            lines.append(f"{index}. **{item['name']}** — {pct(item['raw_rate'])} raw, {pct(item['shrunk_rate'])} adjusted, n={item['n']} ({status})")
        if not technique:
            supported_combinations = [item for item in combination_effects if len(ac.technique_parts(item["name"])) > 1 and item["n"] >= 5][:5]
            if supported_combinations:
                lines.append("\n**Exact combinations**")
                for item in supported_combinations:
                    lines.append(f"• **{item['name']}** — {pct(item['raw_rate'])} raw, {pct(item['shrunk_rate'])} adjusted, n={item['n']}")
        lines.append("_A logged technique indicates reported use, not proven completion. Focus is analyzed separately and small samples are pulled toward baseline._")
        await self.send_long(ctx, "\n".join(lines))
        fig, ax = plt.subplots(figsize=(10, max(4, len(shown) * .55)))
        names = [item["name"] for item in shown][::-1]; values = [item["shrunk_rate"] * 100 for item in shown][::-1]
        ax.barh(names, values); ax.axvline(ac.summarize_records(records)["lucid_rate"] * 100, color="red", linestyle="--", label="Personal baseline"); ax.set_xlabel("Adjusted lucid-night rate (%)"); ax.legend(); fig.tight_layout()
        await self._send_figure(ctx, fig, f"effectiveness_{ctx.author.id}_{year}")

    @commands.command(name="effectiveness")
    async def effectiveness(self, ctx: commands.Context, *, query: str | None = None) -> None:
        technique, year = split_query_year(query, self.active_year)
        if self.source_provider(year) is None: await ctx.send(f"No data file is configured for {year}."); return
        await self._effectiveness(ctx, technique, year)

    async def _wbtb(self, ctx: commands.Context, target: discord.User, year: int) -> None:
        records = self._dataset(str(target.id), year).for_year(year)
        groups = {"No WBTB": [r for r in records if not r.wbtb_attempted], "WBTB": [r for r in records if r.wbtb_attempted]}
        stats = {name: ac.summarize_records(group) for name, group in groups.items()}
        if not all(stats[name]["n"] for name in stats): await ctx.send("Need both WBTB and non-WBTB nights."); return
        text = [f"**WBTB impact — {target.display_name}, {year}**"]
        for name in groups:
            item = stats[name]; text.append(f"• {name} (n={item['n']}): lucid rate **{pct(item['lucid_rate'])}**, recall **{fmt(item['dream_mean'])}**, quality **{fmt(item['quality_mean'])}**, sleep **{fmt(item['sleep_mean'])} h**")
        text.append("_Observational: WBTB nights may also differ in sleep, technique, and motivation._")
        await self.send_long(ctx, "\n".join(text))
        fig, axes = plt.subplots(1, 3, figsize=(12, 4)); names = list(groups)
        for ax, key, title in zip(axes, ("lucid_rate", "dream_mean", "quality_mean"), ("Lucid-night rate (%)", "Recall/night", "Quality")):
            values = [stats[name][key] or 0 for name in names]; values = [v * 100 for v in values] if key == "lucid_rate" else values
            ax.bar(names, values, color=["gray", "tab:orange"]); ax.set_title(title); ax.grid(alpha=.2, axis="y")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"wbtb_{target.id}_{year}")

    @commands.command(name="wbtb_impact")
    async def wbtb_impact(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        await self._wbtb(ctx, user or ctx.author, self.active_year)

    @commands.command(name="conditions", aliases=["lucid_factors"])
    async def conditions(self, ctx: commands.Context, outcome: str = "lucid") -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        if len(records) < 5: await ctx.send("At least five usable reports are required."); return
        outcome = outcome.casefold()
        if outcome not in {"lucid", "dreams", "quality"}: await ctx.send("Choose lucid, dreams, or quality."); return
        outcome_fn = (lambda r: float(r.lucid_night)) if outcome == "lucid" else (lambda r: float(r.dream_count)) if outcome == "dreams" else (lambda r: r.quality_10)
        factors = {
            "WBTB": [("No", lambda r: not r.wbtb_attempted), ("Yes", lambda r: r.wbtb_attempted)],
            "Sleep": [("<7h", lambda r: r.sleep_hours is not None and r.sleep_hours < 7), ("7–9h", lambda r: r.sleep_hours is not None and 7 <= r.sleep_hours <= 9), (">9h", lambda r: r.sleep_hours is not None and r.sleep_hours > 9)],
            "Focus": [("<5", lambda r: r.focus_10 is not None and r.focus_10 < 5), ("5–7", lambda r: r.focus_10 is not None and 5 <= r.focus_10 < 8), ("8–10", lambda r: r.focus_10 is not None and r.focus_10 >= 8)],
            "Weekend": [("Weekday", lambda r: r.report_date.weekday() < 5), ("Weekend", lambda r: r.report_date.weekday() >= 5)],
        }
        fig, axes = plt.subplots(2, 2, figsize=(12, 8)); lines = [f"**Conditions associated with {outcome}**"]
        for ax, (factor, categories) in zip(axes.flat, factors.items()):
            labels, values, counts = [], [], []
            for label, predicate in categories:
                usable = [outcome_fn(r) for r in records if predicate(r) and outcome_fn(r) is not None]
                labels.append(label); values.append(mean(usable) if usable else 0); counts.append(len(usable))
            plotted = [v * 100 for v in values] if outcome == "lucid" else values
            ax.bar(labels, plotted); ax.set_title(factor); ax.set_xticks(range(len(labels)), [f"{label}\n(n={n})" for label, n in zip(labels, counts)]); ax.grid(alpha=.2, axis="y")
            if values: lines.append(f"• {factor}: highest observed **{labels[int(np.argmax(values))]}** ({pct(max(values)) if outcome == 'lucid' else fmt(max(values))})")
        lines.append("_These are unadjusted observational comparisons; factors overlap._")
        await self.send_long(ctx, "\n".join(lines)); fig.tight_layout()
        await self._send_figure(ctx, fig, f"conditions_{ctx.author.id}_{outcome}")

    @commands.command(name="sleep_impact")
    async def sleep_impact(self, ctx: commands.Context) -> None:
        records = [r for r in self._dataset(str(ctx.author.id), self.active_year).records if r.sleep_hours is not None]
        if len(records) < 10: await ctx.send("At least ten reports with sleep duration are required."); return
        bins = [("<6", 0, 6), ("6–7", 6, 7), ("7–8", 7, 8), ("8–9", 8, 9), ("9+", 9, 25)]
        recall = ac.binned_summary(records, lambda r: r.sleep_hours, lambda r: float(r.dream_count), bins)
        lucid = ac.binned_summary(records, lambda r: r.sleep_hours, lambda r: float(r.lucid_night), bins)
        lines = ["**Sleep impact**"]
        for rec, luc in zip(recall, lucid): lines.append(f"• {rec['label']}h (n={rec['n']}): recall **{fmt(rec['mean'])}**, lucid rate **{pct(luc['mean'])}**")
        lines.append("_Self-reported observational relationship; WBTB and technique use may differ between sleep bands._")
        await self.send_long(ctx, "\n".join(lines))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4)); labels = [item["label"] for item in recall]
        axes[0].plot(labels, [item["mean"] or 0 for item in recall], marker="o"); axes[0].set_title("Recall by sleep duration")
        axes[1].plot(labels, [(item["mean"] or 0) * 100 for item in lucid], marker="o", color="green"); axes[1].set_title("Lucid-night rate by sleep duration")
        for ax in axes: ax.grid(alpha=.2); ax.set_xlabel("Hours")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"sleep_impact_{ctx.author.id}")

    @commands.command(name="lagged_effects")
    async def lagged_effects(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year); pairs = ac.consecutive_pairs(records)
        if len(pairs) < 5: await ctx.send("At least five consecutive-night pairs are required."); return
        predictors = {"Journal time": lambda r: r.journal_minutes, "Focus": lambda r: r.focus_10, "Sleep": lambda r: r.sleep_hours, "WBTB": lambda r: float(r.wbtb_count), "Prior recall": lambda r: float(r.dream_count)}
        outcomes = {"Next recall": lambda r: float(r.dream_count), "Next quality": lambda r: r.quality_10, "Next lucid": lambda r: float(r.lucid_night)}
        matrix = np.full((len(predictors), len(outcomes)), np.nan); counts = np.zeros_like(matrix)
        for i, predictor in enumerate(predictors.values()):
            for j, outcome_fn in enumerate(outcomes.values()):
                usable = [(predictor(a), outcome_fn(b)) for a, b in pairs if predictor(a) is not None and outcome_fn(b) is not None]
                matrix[i, j] = ac.spearman([x for x, _ in usable], [y for _, y in usable]) if len(usable) >= 2 else np.nan; counts[i, j] = len(usable)
        await ctx.send(f"**Lagged effects** — {len(pairs)} consecutive pairs. Values are Spearman correlations; cells under n=10 are preliminary.")
        fig, ax = plt.subplots(figsize=(8, 6)); image = ax.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(outcomes)), outcomes); ax.set_yticks(range(len(predictors)), predictors)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]): ax.text(j, i, f"{fmt(matrix[i,j],2)}\nn={int(counts[i,j])}", ha="center", va="center")
        fig.colorbar(image, ax=ax, label="Spearman ρ"); ax.set_title("Date D → date D+1 associations"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"lagged_{ctx.author.id}")

    @commands.command(name="streaks")
    async def streaks(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        if not records: await ctx.send("No usable reports found."); return
        categories = {
            "Reporting": [r.report_date for r in records],
            "Technique practice": [r.report_date for r in records if r.technique_components],
            "WBTB": [r.report_date for r in records if r.wbtb_attempted],
            "Lucid nights": [r.report_date for r in records if r.lucid_night],
            "Recall ≥3": [r.report_date for r in records if r.dream_count >= 3],
        }
        lines = ["**Streaks**"]
        current_values, longest_values = [], []
        for label, days in categories.items():
            current, longest = ac.streak_lengths(days); current_values.append(current); longest_values.append(longest); lines.append(f"• {label}: **{current} latest**, **{longest} longest**")
        await self.send_long(ctx, "\n".join(lines))
        fig, ax = plt.subplots(figsize=(10, 5)); x = np.arange(len(categories)); width = .36
        ax.bar(x - width/2, current_values, width, label="Latest streak"); ax.bar(x + width/2, longest_values, width, label="Longest streak")
        ax.set_xticks(x, categories.keys(), rotation=25); ax.set_ylabel("Consecutive nights"); ax.legend(); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"streaks_{ctx.author.id}")

    @commands.command(name="lucid_gaps")
    async def lucid_gaps(self, ctx: commands.Context, first: str | None = None, second: str | None = None) -> None:
        parsed = await self._target_year(ctx, first, second)
        if not parsed: return
        target, year = parsed; records = self._dataset(str(target.id), year).for_year(year)
        today = min(self.local_today(str(target.id)), date(year, 12, 31)) if year == self.active_year else date(year, 12, 31)
        stats = ac.lucid_gap_stats(records, today)
        if not stats: await ctx.send("At least two unique lucid nights are required."); return
        text = (
            f"**Lucid gaps — {target.display_name}, {year}**\n• Lucid nights: **{stats['n_lucid_nights']}** | completed intervals: **{stats['n_intervals']}**\n"
            f"• Median: **{stats['median']:.1f} days** | mean: **{stats['mean']:.1f}** | IQR: **{stats['q1']:.1f}–{stats['q3']:.1f}**\n"
            f"• Shortest: **{stats['minimum']}** | longest: **{stats['maximum']}** | current unfinished gap: **{stats['current_gap']}**\n"
            f"• Recent pattern: **{stats['trend']}** (recent median {stats['recent_median']:.1f}, historical {stats['historical_median']:.1f})"
        )
        await self.send_long(ctx, text)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5)); dates = stats["lucid_dates"][1:]
        axes[0].plot(dates, stats["gaps"], marker="o"); axes[0].axhline(stats["median"], linestyle="--", color="red", label="Median"); axes[0].set_title("Completed gaps"); axes[0].legend(); axes[0].tick_params(axis="x", rotation=35)
        axes[1].hist(stats["gaps"], bins="auto", color="tab:green", alpha=.8); axes[1].axvline(stats["median"], linestyle="--", color="red"); axes[1].set_title("Gap distribution")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"lucid_gaps_{target.id}_{year}")

    @commands.command(name="lucid_probability")
    async def lucid_probability(self, ctx: commands.Context, days: int | None = None) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year); today = self.local_today(str(ctx.author.id)); stats = ac.lucid_gap_stats(records, today)
        if not stats: await ctx.send("At least two unique lucid nights are required."); return
        horizons = [days] if days is not None else [7, 14, 30]
        if any(h <= 0 or h > 365 for h in horizons): await ctx.send("Days must be between 1 and 365."); return
        current = stats["current_gap"]; eligible = [gap for gap in stats["gaps"] if gap >= current]
        lines = [f"**Historical lucid probability context** — current gap {current} days"]
        rates = []
        for horizon in horizons:
            successes = sum(gap <= current + horizon for gap in eligible); rate = successes / len(eligible) if eligible else None
            rates.append(rate)
            lines.append(f"• Within next {horizon} days: **{pct(rate)}** ({successes}/{len(eligible)} comparable completed gaps)")
        lines.append("_Historical recurrence estimate, not a prediction guarantee._"); await self.send_long(ctx, "\n".join(lines))
        fig, ax = plt.subplots(figsize=(8, 4)); ax.bar([str(h) for h in horizons], [(rate or 0) * 100 for rate in rates], color="tab:green")
        ax.set_xlabel("Next N days"); ax.set_ylabel("Historical recurrence estimate (%)"); ax.set_ylim(0, 100); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"lucid_probability_{ctx.author.id}")

    @commands.command(name="momentum")
    async def momentum(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year); pairs = ac.consecutive_pairs(records)
        after_lucid = [b for a, b in pairs if a.lucid_night]; after_other = [b for a, b in pairs if not a.lucid_night]
        if len(after_lucid) < 3 or len(after_other) < 3: await ctx.send("Need at least three next-night pairs in both groups."); return
        left, right = ac.summarize_records(after_lucid), ac.summarize_records(after_other)
        await ctx.send(f"**Momentum** — after lucid nights (n={left['n']}): recall **{fmt(left['dream_mean'])}**, lucid rate **{pct(left['lucid_rate'])}**; after other nights (n={right['n']}): recall **{fmt(right['dream_mean'])}**, lucid rate **{pct(right['lucid_rate'])}**. Consecutive dates only; observational.")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4)); labels = ["After lucid", "After other"]
        axes[0].bar(labels, [left["dream_mean"] or 0, right["dream_mean"] or 0]); axes[0].set_title("Next-night recall")
        axes[1].bar(labels, [(left["lucid_rate"] or 0) * 100, (right["lucid_rate"] or 0) * 100], color="tab:green"); axes[1].set_title("Next-night lucid rate (%)")
        for ax in axes: ax.grid(alpha=.2, axis="y")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"momentum_{ctx.author.id}")

    @commands.command(name="interactions")
    async def interactions(self, ctx: commands.Context, factor1: str, factor2: str | None = None) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        definitions = {
            "wbtb": lambda r: "yes" if r.wbtb_attempted else "no",
            "sleep": lambda r: None if r.sleep_hours is None else "<7h" if r.sleep_hours < 7 else "7–9h" if r.sleep_hours <= 9 else ">9h",
            "focus": lambda r: None if r.focus_10 is None else "<5" if r.focus_10 < 5 else "5–7" if r.focus_10 < 8 else "8–10",
            "technique": lambda r: r.technique_combination or "none",
        }
        factor1 = factor1.casefold(); requested = [factor2.casefold()] if factor2 else [name for name in ("wbtb", "sleep", "focus", "technique") if name != factor1]
        if factor1 not in definitions or any(name not in definitions for name in requested): await ctx.send("Factors: wbtb, sleep, focus, technique."); return
        lines = [f"**Interactions for {factor1}**"]; plot_labels, plot_values = [], []
        for second in requested:
            cells: defaultdict[tuple[str, str], list[ac.NightRecord]] = defaultdict(list)
            for record in records:
                a, b = definitions[factor1](record), definitions[second](record)
                if a is not None and b is not None: cells[(a, b)].append(record)
            supported = [(key, group) for key, group in cells.items() if len(group) >= 5]
            if not supported: lines.append(f"• {factor1} × {second}: insufficient cells (need n≥5)"); continue
            best_key, best_group = max(supported, key=lambda item: ac.summarize_records(item[1])["lucid_rate"] or 0)
            best_rate = ac.summarize_records(best_group)["lucid_rate"] or 0
            lines.append(f"• {factor1} × {second}: highest supported cell **{best_key[0]} + {best_key[1]}**, {pct(best_rate)} (n={len(best_group)})")
            plot_labels.append(f"{second}\n{best_key[0]} + {best_key[1]}"); plot_values.append(best_rate * 100)
        lines.append("_Exploratory comparisons; requesting many interactions increases the chance of unstable patterns._"); await self.send_long(ctx, "\n".join(lines))
        if plot_labels:
            fig, ax = plt.subplots(figsize=(max(7, len(plot_labels) * 2.2), 4)); ax.bar(plot_labels, plot_values, color="tab:purple"); ax.set_ylabel("Lucid-night rate (%)"); ax.set_title("Highest supported interaction cells"); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
            await self._send_figure(ctx, fig, f"interactions_{ctx.author.id}")

    @commands.command(name="matched_nights")
    async def matched_nights(self, ctx: commands.Context, *, query: str) -> None:
        records = self._dataset(str(ctx.author.id), self.active_year).for_year(self.active_year)
        component = ac.technique_parts(query)
        treated = [r for r in records if component and component[0] in r.technique_components]
        if not treated: treated, _ = ac.keyword_groups(records, query)
        treated_ids = {r.report_date for r in treated}; controls = [r for r in records if r.report_date not in treated_ids]
        if len(treated) < 3 or len(controls) < 3: await ctx.send("Need at least three matching and three comparison nights."); return
        pairs = []
        for item in treated:
            def distance(other: ac.NightRecord) -> float:
                score = abs(item.wbtb_count - other.wbtb_count) * 2 + abs(item.report_date.weekday() - other.report_date.weekday()) / 6
                if item.sleep_hours is not None and other.sleep_hours is not None: score += abs(item.sleep_hours - other.sleep_hours)
                if item.focus_10 is not None and other.focus_10 is not None: score += abs(item.focus_10 - other.focus_10) / 2
                return score
            pairs.append((item, min(controls, key=distance)))
        treated_summary = ac.summarize_records([a for a, _ in pairs]); control_summary = ac.summarize_records([b for _, b in pairs])
        await ctx.send(f"**Matched nights: `{query}`** — {len(pairs)} pairs. Matching nights: recall **{fmt(treated_summary['dream_mean'])}**, lucid rate **{pct(treated_summary['lucid_rate'])}**; comparisons: recall **{fmt(control_summary['dream_mean'])}**, lucid rate **{pct(control_summary['lucid_rate'])}**. Matching considers WBTB, weekday, sleep, and focus when available; unrecorded factors remain.")
        fig, axes = plt.subplots(1, 2, figsize=(9, 4)); labels = ["Matching", "Comparison"]
        axes[0].bar(labels, [treated_summary["dream_mean"] or 0, control_summary["dream_mean"] or 0]); axes[0].set_title("Recall/night")
        axes[1].bar(labels, [(treated_summary["lucid_rate"] or 0) * 100, (control_summary["lucid_rate"] or 0) * 100], color="tab:green"); axes[1].set_title("Lucid-night rate (%)")
        for ax in axes: ax.grid(alpha=.2, axis="y")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"matched_{ctx.author.id}")

    @commands.command(name="data_quality")
    async def data_quality(self, ctx: commands.Context, year_token: str | None = None) -> None:
        year = ac.parse_year(year_token, self.active_year)
        if year is None or self.source_provider(year) is None: await ctx.send("Invalid or unavailable year."); return
        dataset = self._dataset(str(ctx.author.id), year); records = dataset.for_year(year); counts = Counter(issue.kind for issue in dataset.issues)
        summary = ac.summarize_records(records)
        lines = [f"**Data quality — {year}**", f"• Raw reports: **{dataset.raw_report_count}** | usable unique nights: **{len(records)}**", f"• Duplicate dates excluded: **{len(dataset.duplicate_dates)}**"]
        for kind, count in counts.most_common(): lines.append(f"• {kind.replace('_', ' ').title()}: **{count}**")
        lines.extend([f"• Sleep coverage: **{summary['sleep_n']}/{summary['n']}**", f"• Focus coverage: **{summary['focus_n']}/{summary['n']}**", f"• Journal-time coverage: **{summary['journal_n']}/{summary['n']}**"])
        readiness = {"Trend": min(summary["n"], 14), "Conditions": summary["n"], "Techniques": max((item["n"] for item in ac.technique_effects(records, True)), default=0), "Lagged effects": len(ac.consecutive_pairs(records))}
        lines.append("\n**Readiness**")
        for name, n in readiness.items(): lines.append(f"• {name}: **{ac.confidence_label(n)}** (n={n})")
        if dataset.duplicate_dates: lines.append("⚠️ Correct duplicate dates with `!edit` or `!delete`; new duplicate submissions are now warned immediately.")
        await self.send_long(ctx, "\n".join(lines))
        fig, ax = plt.subplots(figsize=(8, 4)); names = ["Sleep", "Focus", "Journal time"]; values = [(summary["sleep_n"] / summary["n"] * 100) if summary["n"] else 0, (summary["focus_n"] / summary["n"] * 100) if summary["n"] else 0, (summary["journal_n"] / summary["n"] * 100) if summary["n"] else 0]
        ax.bar(names, values, color=["tab:blue", "tab:orange", "tab:green"]); ax.set_ylim(0, 100); ax.set_ylabel("Usable field coverage (%)"); ax.grid(alpha=.2, axis="y"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"data_quality_{ctx.author.id}_{year}")

    @commands.command(name="dreams")
    async def dreams(self, ctx: commands.Context, period: str | None = None) -> None:
        year = ac.parse_year(period, self.active_year) if period and period.isdigit() else self.active_year
        if year is None or self.source_provider(year) is None: await ctx.send("Invalid or unavailable year."); return
        records = self._dataset(str(ctx.author.id), year).for_year(year); summary = ac.summarize_records(records)
        await ctx.send(f"**Dream recall — {year}**: {summary['dream_total']} total over {summary['n']} reported nights ({fmt(summary['dream_mean'])}/night; {pct(summary['zero_recall_rate'])} zero-recall nights).")

    @commands.command(name="personal_recap")
    async def personal_recap(self, ctx: commands.Context, year_token: str | None = None) -> None:
        year = ac.parse_year(year_token, self.active_year)
        if year is None or self.source_provider(year) is None: await ctx.send("Invalid or unavailable year."); return
        await self._overview(ctx, ctx.author, year)

    @commands.command(name="final_group")
    async def final_group(self, ctx: commands.Context, year_token: str | None = None) -> None:
        year = ac.parse_year(year_token, self.active_year)
        if year is None or self.source_provider(year) is None: await ctx.send("Invalid or unavailable year."); return
        await self._group(ctx, date(year, 1, 1), date(year, 12, 31), f"Group recap — {year}")

    @commands.command(name="lucid_history")
    async def lucid_history(self, ctx: commands.Context, limit: int = 10) -> None:
        limit = max(1, min(limit, 50)); records = [r for r in self._dataset(str(ctx.author.id), self.active_year).records if r.lucid_night]
        if not records: await ctx.send("No lucid nights found."); return
        lines = [f"**Most recent {min(limit, len(records))} lucid nights**"]
        for record in records[-limit:][::-1]: lines.append(f"• **{record.report_date:%d.%m.%Y}** — {record.lucid_count} lucid, recall {record.dream_count}, quality {fmt(record.quality_10)}, technique {record.technique_text or 'none'}")
        await self.send_long(ctx, "\n".join(lines))

    @commands.command(name="dreams_25", aliases=["dreams_previous"])
    async def dreams_25(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), 2025).for_year(2025); summary = ac.summarize_records(records)
        await ctx.send(f"**Dream recall — 2025**: {summary['dream_total']} total over {summary['n']} reported nights ({fmt(summary['dream_mean'])}/night; {pct(summary['zero_recall_rate'])} zero-recall nights).")

    @commands.command(name="personal_recap_25", aliases=["personal_recap_previous"])
    async def personal_recap_25(self, ctx: commands.Context) -> None:
        await self._overview(ctx, ctx.author, 2025)

    @commands.command(name="final_group_25", aliases=["final_group_previous"])
    async def final_group_25(self, ctx: commands.Context) -> None:
        await self._group(ctx, date(2025, 1, 1), date(2025, 12, 31), "Group recap — 2025")

    @commands.command(name="journaltime_25", aliases=["journaltime_previous"])
    async def journaltime_25(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        await self._journal_impact_user(ctx, user or ctx.author, 2025)

    @commands.command(name="correlate_25", aliases=["correlate_previous"])
    async def correlate_25(self, ctx: commands.Context, *, keyword: str) -> None:
        records = self._dataset(str(ctx.author.id), 2025).for_year(2025); matching, other = ac.keyword_groups(records, keyword)
        if not matching: await ctx.send(f"No whole-word or phrase matches found for `{keyword}` in 2025."); return
        left, right = ac.summarize_records(matching), ac.summarize_records(other)
        await ctx.send(f"**2025 keyword comparison: `{keyword}`** — matching (n={left['n']}): recall {fmt(left['dream_mean'])}, quality {fmt(left['quality_mean'])}, lucid rate {pct(left['lucid_rate'])}; other (n={right['n']}): recall {fmt(right['dream_mean'])}, quality {fmt(right['quality_mean'])}, lucid rate {pct(right['lucid_rate'])}.")

    @commands.command(name="day_of_week_25", aliases=["day_of_week_previous"])
    async def day_of_week_25(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), 2025).for_year(2025)
        if not records: await ctx.send("No usable 2025 reports found."); return
        grouped = {day: [record for record in records if record.report_date.weekday() == day] for day in range(7)}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        configs = (("dream_mean", "Recall"), ("quality_mean", "Quality"), ("lucid_rate", "Lucid-night rate"), ("focus_mean", "Focus"))
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (key, title) in zip(axes.flat, configs):
            values = [ac.summarize_records(grouped[day])[key] or 0 for day in range(7)]
            if key == "lucid_rate": values = [value * 100 for value in values]
            ax.bar(days, values); ax.set_title(f"{title} — 2025"); ax.set_xticks(range(7), [f"{name}\nn={len(grouped[i])}" for i, name in enumerate(days)]); ax.grid(alpha=.2, axis="y")
        fig.tight_layout(); await self._send_figure(ctx, fig, f"weekday_{ctx.author.id}_2025")

    @commands.command(name="lucid_factors_25", aliases=["lucid_factors_previous"])
    async def lucid_factors_25(self, ctx: commands.Context) -> None:
        records = self._dataset(str(ctx.author.id), 2025).for_year(2025)
        if len(records) < 5: await ctx.send("At least five usable 2025 reports are required."); return
        factors = {
            "WBTB": [("No", lambda r: not r.wbtb_attempted), ("Yes", lambda r: r.wbtb_attempted)],
            "Sleep": [("<7h", lambda r: r.sleep_hours is not None and r.sleep_hours < 7), ("7–9h", lambda r: r.sleep_hours is not None and 7 <= r.sleep_hours <= 9), (">9h", lambda r: r.sleep_hours is not None and r.sleep_hours > 9)],
            "Focus": [("<5", lambda r: r.focus_10 is not None and r.focus_10 < 5), ("5–7", lambda r: r.focus_10 is not None and 5 <= r.focus_10 < 8), ("8–10", lambda r: r.focus_10 is not None and r.focus_10 >= 8)],
            "Weekend": [("Weekday", lambda r: r.report_date.weekday() < 5), ("Weekend", lambda r: r.report_date.weekday() >= 5)],
        }
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, (factor, categories) in zip(axes.flat, factors.items()):
            values, labels = [], []
            for label, predicate in categories:
                group = [r for r in records if predicate(r)]; values.append(ac.summarize_records(group)["lucid_rate"] or 0); labels.append(f"{label}\nn={len(group)}")
            ax.bar(labels, [value * 100 for value in values]); ax.set_title(factor); ax.grid(alpha=.2, axis="y")
        fig.suptitle("Conditions associated with lucidity — 2025"); fig.tight_layout()
        await self._send_figure(ctx, fig, f"conditions_{ctx.author.id}_2025")

    @commands.command(name="month_group_25", aliases=["month_group_previous"])
    async def month_group_25(self, ctx: commands.Context, month_token: str | None = None) -> None:
        try: month = int(month_token) if month_token else date.today().month
        except ValueError: await ctx.send("Use a month number from 1 to 12."); return
        if not 1 <= month <= 12: await ctx.send("Use a month number from 1 to 12."); return
        start, end = month_bounds(2025, month); await self._group(ctx, start, end, f"Group — {start:%B 2025}")

    @commands.command(name="month_25", aliases=["month_previous"])
    async def month_25(self, ctx: commands.Context, month_token: str | None = None) -> None:
        token = f"{int(month_token):02d}.25" if month_token and month_token.isdigit() else month_token or f"{date.today().month:02d}.25"
        await self.month.callback(self, ctx, token)

    @commands.command(name="wbtb_impact_25", aliases=["wbtb_impact_previous"])
    async def wbtb_impact_25(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        await self._wbtb(ctx, user or ctx.author, 2025)

    @commands.command(name="lucid_gaps_25", aliases=["lucid_gaps_previous"])
    async def lucid_gaps_25(self, ctx: commands.Context, user: discord.User | None = None) -> None:
        target = user or ctx.author; records = self._dataset(str(target.id), 2025).for_year(2025); stats = ac.lucid_gap_stats(records, date(2025, 12, 31))
        if not stats: await ctx.send("At least two unique lucid nights are required."); return
        await ctx.send(f"**2025 lucid gaps — {target.display_name}**: median {stats['median']:.1f} days, IQR {stats['q1']:.1f}–{stats['q3']:.1f}, {stats['n_lucid_nights']} lucid nights. Use `!lucid_gaps {target.mention} 25` for the chart.")

    @commands.command(name="effectiveness_25", aliases=["effectiveness_previous"])
    async def effectiveness_25(self, ctx: commands.Context, *, technique: str | None = None) -> None:
        await self._effectiveness(ctx, technique, 2025)


def re_month(token: str, default_year: int) -> tuple[int, int] | None:
    try:
        month_text, year_text = token.strip().split(".", 1)
        month = int(month_text); year = ac.parse_year(year_text, default_year)
        if year is None or not 1 <= month <= 12: return None
        return month, year
    except (AttributeError, TypeError, ValueError):
        return None


def split_query_year(query: str | None, default_year: int) -> tuple[str | None, int]:
    if not query: return None, default_year
    pieces = query.split(); possible = pieces[-1]
    if possible.isdigit() and len(possible) in (2, 4):
        year = ac.parse_year(possible, default_year)
        if year is not None: return " ".join(pieces[:-1]) or None, year
    return query, default_year
