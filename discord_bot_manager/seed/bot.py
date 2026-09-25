import os
import json
import re
import math
import asyncio
import signal
import tempfile
import logging
import discord
from discord import app_commands
from discord.ext import commands, tasks
import calendar
from collections import defaultdict
from dotenv import load_dotenv
from datetime import datetime, timedelta, date, timezone
import pytz  # For timezone support
import matplotlib
matplotlib.use('Agg')  # Important for server environments
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from analysis_commands import AnalysisCommands

# Load environment variables relative to this script, not the launch directory.
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get('BOT_STORAGE_DIR', CODE_DIR)
os.makedirs(BASE_DIR, exist_ok=True)
load_dotenv(os.path.join(CODE_DIR, '.env'))
DISCORD_TOKEN = os.getenv('DISCORD_TOKEN')

# Initialize bot
intents = discord.Intents.default()
intents.message_content = True
intents.members = True
bot = commands.Bot(
    command_prefix='!',
    intents=intents,
    allowed_mentions=discord.AllowedMentions.none(),
)
_commands_synced = False
_plot_lock = asyncio.Lock()
_command_cooldowns = commands.CooldownMapping.from_cooldown(3, 10.0, commands.BucketType.user)
PLOT_COMMANDS = {
    'trend', 'group', 'month', 'month_group', 'wbtb_impact', 'lucid_factors', 'conditions',
    'overview', 'heatmap', 'day_of_week', 'journaltime', 'journal_impact',
    'journal_impact_all', 'journaltime_all', 'baseline', 'sleep_impact', 'lagged_effects',
    'lucid_gaps', 'overview_25', 'wbtb_impact_25', 'lucid_gaps_25', 'month_25',
    'effectiveness', 'effectiveness_25', 'journaltime_25', 'day_of_week_25',
    'lucid_factors_25', 'month_group_25', 'personal_recap', 'personal_recap_25',
    'final_group', 'final_group_25', 'correlate', 'streaks', 'lucid_probability',
    'momentum', 'interactions', 'matched_nights', 'data_quality',
}


@bot.before_invoke
async def apply_command_safety(ctx):
    bucket = _command_cooldowns.get_bucket(ctx.message)
    retry_after = bucket.update_rate_limit()
    if retry_after:
        raise commands.CommandOnCooldown(bucket, retry_after, commands.BucketType.user)
    if ctx.command and ctx.command.name in PLOT_COMMANDS:
        await _plot_lock.acquire()
        ctx._holds_plot_lock = True


@bot.after_invoke
async def release_command_resources(ctx):
    if getattr(ctx, '_holds_plot_lock', False) and _plot_lock.locked():
        _plot_lock.release()
        ctx._holds_plot_lock = False


@bot.event
async def on_command_error(ctx, error):
    error = getattr(error, 'original', error)
    if isinstance(error, commands.CommandNotFound):
        return
    if isinstance(error, commands.CommandOnCooldown):
        await ctx.send(f"Please wait {error.retry_after:.1f} seconds before using another command.")
        return
    if isinstance(error, commands.MissingPermissions):
        await ctx.send("You do not have permission to use that command.")
        return
    if isinstance(error, commands.NoPrivateMessage):
        await ctx.send("That command can only be used in a server.")
        return
    if isinstance(error, commands.MissingRequiredArgument):
        await ctx.send(f"Missing required argument: `{error.param.name}`. Use `!help` for usage.")
        return
    if isinstance(error, commands.BadArgument):
        await ctx.send("One of the command arguments is invalid. Use `!help` for usage.")
        return
    await send_internal_error(ctx, error)

# Define file to store data
ACTIVE_REPORT_YEAR = datetime.now(timezone.utc).year
DATA_FILE_26 = os.path.join(BASE_DIR, f"user_reports_{ACTIVE_REPORT_YEAR % 100:02d}.json")
DATA_FILE_25 = os.path.join(BASE_DIR, "user_reports_25.json")
PREFERENCES_FILE = os.path.join(BASE_DIR, "user_preferences.json")


def load_json_file(path, default):
    """Load JSON without silently discarding malformed or unreadable data."""
    if not os.path.exists(path):
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Could not load {os.path.basename(path)}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise RuntimeError(f"{os.path.basename(path)} must contain a JSON object at the top level.")
    return loaded


def atomic_write_json(path, data):
    """Durably replace a JSON file after a complete successful write."""
    directory = os.path.dirname(path) or BASE_DIR
    fd, temporary_file = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_file, path)
    finally:
        if os.path.exists(temporary_file):
            os.remove(temporary_file)


def unique_output_path(prefix, suffix=".png"):
    """Reserve a unique output path beside the bot for concurrent commands."""
    fd, path = tempfile.mkstemp(prefix=f"{prefix}_", suffix=suffix, dir=BASE_DIR)
    os.close(fd)
    return path

# Load data without mutating any file during startup.
user_reports = load_json_file(DATA_FILE_26, {})
user_reports_25 = load_json_file(DATA_FILE_25, {})
user_preferences = load_json_file(PREFERENCES_FILE, {})

def find_users_with_missing_reports(user_ids=None, now_utc=None):
    """Find users missing a report on their own current local date."""
    now_utc = now_utc or datetime.now(timezone.utc)
    missing_users = []

    candidate_ids = [str(user_id) for user_id in (user_ids if user_ids is not None else user_reports.keys())]
    for user_id in candidate_ids:
        reports = user_reports.get(user_id, [])
        prefs = user_preferences.get(user_id, {})
        try:
            local_date = now_utc.astimezone(pytz.timezone(prefs.get("timezone", "UTC"))).date()
        except (pytz.UnknownTimeZoneError, AttributeError):
            local_date = now_utc.date()
        reported_today = False
        for report in reports:
            parsed = parse_report_date(report.get("date", ""))
            if parsed and parsed.date() == local_date:
                reported_today = True
            if reported_today:
                break
        if not reported_today:
            missing_users.append(user_id)
            
    return missing_users

def save_preferences():
    """Save user preferences to file."""
    atomic_write_json(PREFERENCES_FILE, user_preferences)

# Allowed user IDs for the !inactive command
ALLOWED_INACTIVE_USERS = [
    1058450155445170296,
    809449651664191582,
    727104213561114678
]

def normalize_technique(tech_str):
    """Normalize technique string so order doesn't matter.
    
    'mild, ssild' and 'ssild, mild' will both become 'mild, ssild'.
    """
    if not tech_str or tech_str.lower().strip() == 'none':
        return 'none'
    techniques = [t.strip().lower() for t in tech_str.split(',')]
    return ', '.join(sorted(techniques))

def get_sleep_time(report):
    """Get sleep_time from a report, checking both 'sleep_time' and 'sleep time' keys."""
    value = report.get("sleep_time") or report.get("sleep time")
    if value:
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    return None

def is_valid_float(value):
    """Check if a value is a valid float (including integers)."""
    try:
        float(value)
        return True
    except (ValueError, TypeError):
        return False

# Common timezones for the reminder slash command
COMMON_TIMEZONES = [
    "Europe/London", "Europe/Paris", "Europe/Berlin", "Europe/Moscow",
    "America/New_York", "America/Chicago", "America/Denver", "America/Los_Angeles",
    "Asia/Tokyo", "Asia/Shanghai", "Asia/Kolkata", "Australia/Sydney",
    "UTC"
]

# Daily reminder task - runs every minute to check user-specific reminder times
@tasks.loop(minutes=1.0)
async def daily_reminder():
    channel_id = 1452731841797820578  # REPLACE WITH YOUR CHANNEL ID
    channel = bot.get_channel(channel_id)
    if not channel:
        try:
            channel = await bot.fetch_channel(channel_id)
        except (discord.NotFound, discord.Forbidden, discord.HTTPException):
            return
    
    now_utc = datetime.now(pytz.UTC)
    
    guild = getattr(channel, "guild", None)
    candidate_ids = [member.id for member in guild.members if not member.bot] if guild else user_reports.keys()
    missing_user_ids = find_users_with_missing_reports(candidate_ids, now_utc)
    
    users_to_ping = []
    for user_id in missing_user_ids:
        prefs = user_preferences.get(user_id, {})
        
        # Skip opted-out users
        if prefs.get("opted_out", False):
            continue
        
        # Get user's preferred time and timezone (default: 19:00 UTC)
        reminder_time = prefs.get("time", "19:00")
        user_timezone = prefs.get("timezone", "UTC")
        
        try:
            tz = pytz.timezone(user_timezone)
            user_now = now_utc.astimezone(tz)
            # Parse the reminder time
            hour, minute = map(int, reminder_time.split(':'))
            
            # Check if it's the right time (within the minute window)
            if user_now.hour == hour and user_now.minute == minute:
                local_date_key = user_now.strftime("%Y-%m-%d")
                if prefs.get("last_reminder_date") != local_date_key:
                    users_to_ping.append((user_id, local_date_key))
        except (pytz.UnknownTimeZoneError, ValueError, TypeError):
            # If timezone parsing fails, fall back to default behavior
            if now_utc.hour == 19 and now_utc.minute == 0:
                date_key = now_utc.strftime("%Y-%m-%d")
                if prefs.get("last_reminder_date") != date_key:
                    users_to_ping.append((user_id, date_key))
    
    if users_to_ping:
        mentions = " ".join([f"<@{user_id}>" for user_id, _ in users_to_ping])
        try:
            await channel.send(
                f"🌙 Don't forget to submit your dream report today! {mentions}",
                allowed_mentions=discord.AllowedMentions(users=True),
            )
        except (discord.Forbidden, discord.HTTPException) as exc:
            logging.error("Failed to send daily reminders", exc_info=(type(exc), exc, exc.__traceback__))
            return
        for user_id, date_key in users_to_ping:
            user_preferences.setdefault(user_id, {})["last_reminder_date"] = date_key
        try:
            save_preferences()
        except OSError as exc:
            logging.error("Failed to save reminder state", exc_info=(type(exc), exc, exc.__traceback__))

@tasks.loop(seconds=10)
async def manager_heartbeat():
    status_path = os.getenv("BOT_STATUS_FILE")
    if status_path:
        atomic_write_json(status_path, {
            "connected": bot.is_ready(),
            "updated": datetime.now(timezone.utc).timestamp(),
        })


@bot.event
async def on_ready():
    global _commands_synced
    if not manager_heartbeat.is_running():
        manager_heartbeat.start()
    if not daily_reminder.is_running():
        daily_reminder.start()
    # Sync slash commands with Discord
    if not _commands_synced:
        try:
            synced = await bot.tree.sync()
            _commands_synced = True
            print(f'Synced {len(synced)} slash command(s)')
        except discord.HTTPException as e:
            print(f'Failed to sync slash commands: {e}')
    print(f'{bot.user.name} has connected to Discord and is ready!')

async def send_long_message(ctx, content, allowed_mentions=None):
    """Splits a long message into chunks of 2000 characters and sends them."""
    if len(content) <= 2000:
        await ctx.send(content, allowed_mentions=allowed_mentions)
        return
    
    # Split by lines to avoid cutting in the middle of a line if possible
    lines = content.split('\n')
    current_chunk = ""
    for line in lines:
        if len(current_chunk) + len(line) + 1 > 2000:
            if current_chunk:
                await ctx.send(current_chunk, allowed_mentions=allowed_mentions)
                current_chunk = ""
            
            # If a single line is > 2000, split it by characters
            if len(line) > 2000:
                for i in range(0, len(line), 2000):
                    await ctx.send(line[i:i+2000], allowed_mentions=allowed_mentions)
                continue
        
        if current_chunk:
            current_chunk += '\n' + line
        else:
            current_chunk = line
            
    if current_chunk:
        await ctx.send(current_chunk, allowed_mentions=allowed_mentions)


async def send_internal_error(ctx, exc):
    """Log diagnostic detail server-side without exposing it in Discord."""
    plt.close('all')
    logging.error(
        "Command %s failed",
        getattr(getattr(ctx, "command", None), "qualified_name", "unknown"),
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    await ctx.send("An unexpected error occurred. The details were written to the server log.")


async def send_generated_file(ctx, path):
    """Send a temporary artifact and always remove it afterwards."""
    try:
        await ctx.send(file=discord.File(path))
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

@bot.tree.command(name="defaults", description="Set or view your default preferences for reports.")
@app_commands.describe(technique="Set your default technique (or 'none' to explicitly set no technique).")
async def defaults(interaction: discord.Interaction, technique: str = None):
    user_id = str(interaction.user.id)
    
    if user_id not in user_preferences:
        user_preferences[user_id] = {}
        
    changes_made = False
    
    if technique:
        # If user explicitly types "None", we can store it as None or handle it. 
        # But per requirements, "default here would be none" implies if they don't set it, it's None.
        # If they use this command to set it to something, it's that something.
        # If they want to clear it, they might type "None".
        if technique.lower() == "none":
             # We can either remove the key or set it to "None". 
             # Let's set it to "None" to be explicit, or remove it so it falls back to "None" logic?
             # User said: "default here would be none".
             # If I set it to "None", it works.
             user_preferences[user_id]["default_technique"] = "none"
        else:
            user_preferences[user_id]["default_technique"] = technique
        changes_made = True

    if changes_made:
        save_preferences()
        await interaction.response.send_message(f"✅ Your defaults have been updated.\nCurrent Default Technique: {user_preferences[user_id].get('default_technique', 'none')}")
    else:
        # Just viewing
        current_tech = user_preferences[user_id].get("default_technique", "none")
        await interaction.response.send_message(f"ℹ️ **Your Current Defaults:**\nTechnique: {current_tech}\n\nUse `/defaults technique:Name` to change it.")

REPORT_REQUIRED_FIELDS = ["dreams", "quality", "wbtb", "lucid"]
REPORT_ALLOWED_FIELDS = {
    "date", "dreams", "quality", "wbtb", "lucid", "technique", "notes",
    "sleep_time", "focus", "journal_time"
}
MAX_REPORT_RANGE_DAYS = 366


def parse_report_date(value):
    """Parse a report date in the formats accepted by the bot."""
    if not isinstance(value, str):
        return None
    for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(value.strip(), fmt)
        except ValueError:
            continue
    return None


def normalized_quality(value):
    """Return quality on a 0-10 scale, or None for invalid input."""
    text = str(value).strip()
    try:
        if '/' in text:
            parts = text.split('/')
            if len(parts) != 2:
                return None
            numerator, denominator = map(float, parts)
            if not all(math.isfinite(v) for v in (numerator, denominator)):
                return None
            if denominator <= 0 or numerator < 0 or numerator > denominator:
                return None
            return numerator / denominator * 10
        number = float(text)
        if math.isfinite(number) and 0 <= number <= 10:
            return number
    except (TypeError, ValueError, ZeroDivisionError):
        pass
    return None


def normalized_nonnegative_int(value, maximum=None):
    """Return a canonical integer or None; reject floats, signs and infinities."""
    text = str(value).strip()
    if not re.fullmatch(r"\d+", text):
        return None
    number = int(text)
    if maximum is not None and number > maximum:
        return None
    return number


def normalized_finite_float(value, minimum=0, maximum=None):
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number < minimum:
        return None
    if maximum is not None and number > maximum:
        return None
    return number


def rolling_report_windows(reports, anchor_date=None):
    """Return reports from the latest 7 calendar days and the preceding 7 days."""
    anchor_date = anchor_date or datetime.now(timezone.utc).date()
    current_start = anchor_date - timedelta(days=6)
    previous_start = anchor_date - timedelta(days=13)
    current, previous = [], []
    for report in reports:
        parsed = parse_report_date(report.get("date", ""))
        if not parsed:
            continue
        report_date = parsed.date()
        if current_start <= report_date <= anchor_date:
            current.append(report)
        elif previous_start <= report_date < current_start:
            previous.append(report)
    current.sort(key=lambda item: parse_report_date(item.get("date", "")))
    previous.sort(key=lambda item: parse_report_date(item.get("date", "")))
    return current, previous


def daily_metric_series(reports, start_date, metric, transform=int):
    """Aggregate a metric into seven calendar-day buckets, including missing days."""
    totals = {start_date + timedelta(days=i): 0 for i in range(7)}
    for report in reports:
        parsed = parse_report_date(report.get("date", ""))
        if parsed and parsed.date() in totals:
            try:
                totals[parsed.date()] += transform(report.get(metric, 0))
            except (TypeError, ValueError, ZeroDivisionError):
                continue
    return list(totals.keys()), list(totals.values())


def user_local_today(user_id, now_utc=None):
    now_utc = now_utc or datetime.now(timezone.utc)
    timezone_name = user_preferences.get(str(user_id), {}).get("timezone", "UTC")
    try:
        return now_utc.astimezone(pytz.timezone(timezone_name)).date()
    except (pytz.UnknownTimeZoneError, AttributeError):
        return now_utc.date()


def split_report_blocks(body):
    """Split pasted reports when a new Date field or !report marker begins."""
    blocks = []
    current = {}

    for raw_line in body.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.lower() == "!report":
            if current:
                blocks.append(current)
                current = {}
            continue
        if ':' not in line:
            continue

        key, _, value = line.partition(':')
        normalized_key = key.strip().lower().replace(' ', '_')
        # A second Date field starts the next pasted report. This still lets a
        # normal single report put its optional Date field at the end.
        if normalized_key == "date" and "date" in current:
            blocks.append(current)
            current = {}
        current[normalized_key] = value.strip()

    if current:
        blocks.append(current)
    return blocks


def expand_report_dates(report_data):
    """Return one report per date; date ranges are inclusive."""
    date_value = report_data.get("date")
    if not date_value:
        return [report_data], None

    range_match = re.fullmatch(r"\s*(.+?)\s*-\s*(.+?)\s*", date_value)
    if not range_match:
        parsed_date = parse_report_date(date_value)
        if not parsed_date:
            return [], "Date must be DD.MM.YY, DD.MM.YYYY, or an inclusive range of those dates."
        result = report_data.copy()
        result["date"] = parsed_date.strftime("%d.%m.%Y")
        return [result], None

    start = parse_report_date(range_match.group(1))
    end = parse_report_date(range_match.group(2))
    if not start or not end:
        return [], "Date range must look like DD.MM.YYYY-DD.MM.YYYY."
    if end < start:
        return [], "The end of a date range cannot be before its start."

    day_count = (end - start).days + 1
    if day_count > MAX_REPORT_RANGE_DAYS:
        return [], f"A date range can contain at most {MAX_REPORT_RANGE_DAYS} days."

    expanded = []
    for offset in range(day_count):
        result = report_data.copy()
        result["date"] = (start + timedelta(days=offset)).strftime("%d.%m.%Y")
        expanded.append(result)
    return expanded, None


def is_valid_report_number(value, is_quality=False, allow_float=False):
    """Validate numeric report fields."""
    if is_quality:
        return normalized_quality(value) is not None
    if allow_float:
        return normalized_finite_float(value) is not None
    return normalized_nonnegative_int(value) is not None


@bot.command(name='report')
async def report(ctx):
    try:
        content = ctx.message.content.split('\n', 1)
        if len(content) < 2:
            await ctx.send("Please provide the report in the correct format after `!report`.")
            return

        blocks = split_report_blocks(content[1])
        if not blocks:
            await ctx.send("Please provide at least one report after `!report`.")
            return

        user_id = str(ctx.author.id)
        user_prefs = user_preferences.get(user_id, {})
        reports_to_save = []
        try:
            users_today = datetime.now(pytz.timezone(user_prefs.get("timezone", "UTC"))).date()
        except (pytz.UnknownTimeZoneError, AttributeError):
            users_today = datetime.now(timezone.utc).date()

        # Validate every block before saving any part of the batch.
        for block_number, report_data in enumerate(blocks, start=1):
            unknown = sorted(set(report_data) - REPORT_ALLOWED_FIELDS)
            if unknown:
                await ctx.send(
                    f"Report {block_number} contains unknown field(s): {', '.join(unknown)}. "
                    "No reports were saved."
                )
                return
            missing = [field for field in REPORT_REQUIRED_FIELDS if field not in report_data]
            if missing:
                await ctx.send(
                    f"Report {block_number} is missing the following required fields: {', '.join(missing)}. "
                    "No reports were saved."
                )
                return

            for field in ["dreams", "lucid"]:
                value = normalized_nonnegative_int(report_data.get(field, ""))
                if value is None:
                    await ctx.send(f"In report {block_number}, '{field}' must be a non-negative integer. No reports were saved.")
                    return
                report_data[field] = str(value)

            wbtb = normalized_nonnegative_int(report_data.get("wbtb", ""))
            if wbtb is None:
                await ctx.send(f"In report {block_number}, 'wbtb' must be a non-negative integer. No reports were saved.")
                return
            report_data["wbtb"] = str(wbtb)

            if normalized_quality(report_data.get("quality", "")) is None:
                await ctx.send(
                    f"In report {block_number}, 'quality' must be from 0-10 or a valid X/Y ratio. No reports were saved."
                )
                return

            if report_data.get("sleep_time"):
                sleep_time = normalized_finite_float(report_data["sleep_time"], maximum=24)
                if sleep_time is None:
                    await ctx.send(f"In report {block_number}, 'sleep_time' must be between 0 and 24. No reports were saved.")
                    return
                report_data["sleep_time"] = str(sleep_time)

            if report_data.get("focus"):
                focus = normalized_quality(report_data["focus"])
                if focus is None:
                    await ctx.send(f"In report {block_number}, 'focus' must be from 0-10 and may be a decimal or valid X/Y ratio. No reports were saved.")
                    return
                report_data["focus"] = str(focus)

            if report_data.get("journal_time"):
                journal_time = normalized_finite_float(report_data["journal_time"])
                if journal_time is None:
                    await ctx.send(f"In report {block_number}, 'journal_time' must be a non-negative number. No reports were saved.")
                    return
                report_data["journal_time"] = str(journal_time)

            expanded_reports, date_error = expand_report_dates(report_data)
            if date_error:
                await ctx.send(f"In report {block_number}: {date_error} No reports were saved.")
                return

            wrong_years = sorted({parse_report_date(r["date"]).year for r in expanded_reports if "date" in r} - {ACTIVE_REPORT_YEAR})
            if wrong_years:
                await ctx.send(
                    f"Report dates must be in {ACTIVE_REPORT_YEAR}; got {', '.join(map(str, wrong_years))}. "
                    "No reports were saved."
                )
                return
            if any(parse_report_date(r["date"]).date() > users_today for r in expanded_reports if "date" in r):
                await ctx.send("Future report dates are not allowed. No reports were saved.")
                return

            for expanded_report in expanded_reports:
                if "date" not in expanded_report:
                    user_timezone = user_prefs.get("timezone", "UTC")
                    try:
                        expanded_report["date"] = datetime.now(pytz.timezone(user_timezone)).strftime("%d.%m.%Y")
                    except (pytz.UnknownTimeZoneError, AttributeError):
                        expanded_report["date"] = datetime.now(timezone.utc).strftime("%d.%m.%Y")
                if parse_report_date(expanded_report["date"]).year != ACTIVE_REPORT_YEAR:
                    await ctx.send(f"Report dates must be in {ACTIVE_REPORT_YEAR}. No reports were saved.")
                    return
                expanded_report.setdefault("technique", user_prefs.get("default_technique", "None"))
                expanded_report.setdefault("notes", "")
                reports_to_save.append(expanded_report)

        updated_user_reports = list(user_reports.get(user_id, [])) + reports_to_save
        updated_user_reports.sort(
            key=lambda item: parse_report_date(item.get("date", "01.01.1970")) or datetime.min
        )

        # Replace the data file only after the entire new JSON document has
        # been written successfully, preventing partially saved batches.
        updated_data = dict(user_reports)
        updated_data[user_id] = updated_user_reports
        atomic_write_json(DATA_FILE_26, updated_data)
        user_reports[user_id] = updated_user_reports

        date_counts = Counter(
            parsed.date()
            for item in updated_user_reports
            if (parsed := parse_report_date(item.get("date", ""))) is not None
        )
        duplicate_dates = sorted(day for day, count in date_counts.items() if count > 1)
        duplicate_warning = ""
        if duplicate_dates:
            formatted = ", ".join(day.strftime("%d.%m.%Y") for day in duplicate_dates)
            duplicate_warning = (
                f"\n⚠️ Duplicate report date(s): **{formatted}**. "
                "Those dates will be excluded from analysis until corrected with `!edit` or `!delete`."
            )

        if len(reports_to_save) == 1:
            saved = reports_to_save[0]
            await ctx.send(f"Your report has been saved! (Date: {saved['date']}, Technique: {saved['technique']}){duplicate_warning}")
        else:
            saved_dates = sorted(parse_report_date(item["date"]) for item in reports_to_save)
            first_date = saved_dates[0].strftime("%d.%m.%Y")
            last_date = saved_dates[-1].strftime("%d.%m.%Y")
            await ctx.send(
                f"Your {len(reports_to_save)} reports have been saved! "
                f"(Dates: {first_date} to {last_date}){duplicate_warning}"
            )

    except Exception as e:
        await send_internal_error(ctx, e)





@bot.command(name='trend')
async def trend(ctx):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        reports = user_reports[user_id]
        
        # Use calendar-day windows rather than the last 14 submitted records.
        this_week, last_week = rolling_report_windows(reports, user_local_today(user_id))
        if not this_week or not last_week:
            await ctx.send("You need reports in both the latest 7-day period and the preceding 7-day period to compare trends.")
            return
        average_week = reports

        # Date parsing
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None

        # Quality normalization
        def normalize_quality(quality_str):
            try:
                parts = str(quality_str).split('/')
                if len(parts) == 1:
                    return int(parts[0])  # Treat as /10
                numerator, denominator = map(int, parts)
                return (numerator / denominator) * 10
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                return 0  # Fallback for invalid formats

        # Summary calculation (FIXED)
        def calculate_summary(data):
            summary = {
                "dreams": 0,
                "quality": 0,
                "wbtb": 0,
                "lucid": 0,
                "sleep_time": 0,
                "focus": 0,
                "techniques": {}
            }
            sleep_time_count = 0
            focus_count = 0
    
            for report in data:
                summary["dreams"] += int(report.get("dreams", 0))
                summary["quality"] += normalize_quality(report.get("quality", "0"))
                summary["wbtb"] += int(report.get("wbtb", 0))
                summary["lucid"] += int(report.get("lucid", 0))
                
                # Track sleep_time and focus if present (uses helper to support floats and 'sleep time' variant)
                sleep_val = get_sleep_time(report)
                if sleep_val is not None:
                    summary["sleep_time"] += sleep_val
                    sleep_time_count += 1
                if report.get("focus") and str(report.get("focus")).isdigit():
                    summary["focus"] += int(report.get("focus", 0))
                    focus_count += 1
        
                # Normalize technique so order doesn't matter (e.g., 'mild, ssild' == 'ssild, mild')
                normalized_tech = normalize_technique(report.get("technique", ""))
                if normalized_tech and normalized_tech != 'none':
                    summary["techniques"][normalized_tech] = summary["techniques"].get(normalized_tech, 0) + 1

            count = len(data)
            if count > 0:
                summary["dreams"] = round(summary["dreams"] / count, 2)
                summary["quality"] = round(summary["quality"] / count, 2)
                summary["wbtb"] = round(summary["wbtb"] / count, 2)
            
            if sleep_time_count > 0:
                summary["sleep_time"] = round(summary["sleep_time"] / sleep_time_count, 1)
            if focus_count > 0:
                summary["focus"] = round(summary["focus"] / focus_count, 1)

            # Handle case with multiple techniques having same max count
            max_count = max(summary["techniques"].values(), default=0)
            most_used = [tech for tech, count in summary["techniques"].items() if count == max_count]
    
            if most_used:
                summary["most_used"] = (", ".join(most_used), max_count)
            else:
                summary["most_used"] = ("None", 0)
    
            return summary

        # Generate comparison
        this_summary = calculate_summary(this_week)
        last_summary = calculate_summary(last_week)
        overall_summary = calculate_summary(average_week)

        def format_change(current, previous):
            change = current - previous
            return f"{current:.1f} ({change:+.1f})" if previous != 0 else f"{current:.1f}"

        response = (
            "📊 **Trend Analysis**\n\n"
            "**This Week vs Last Week:**\n"
            f"• Dreams: {format_change(this_summary['dreams'], last_summary['dreams'])}\n"
            f"• Quality: {format_change(this_summary['quality'], last_summary['quality'])}/10\n"
            f"• WBTB: {format_change(this_summary['wbtb'], last_summary['wbtb'])}\n"
            f"• Lucid: {this_summary['lucid']} (+{this_summary['lucid'] - last_summary['lucid']})\n"
            f"• Sleep Time: {format_change(this_summary['sleep_time'], last_summary['sleep_time'])} hrs\n"
            f"• Focus: {format_change(this_summary['focus'], last_summary['focus'])}/10\n"
            f"• Top Technique: {this_summary['most_used'][0]} ({this_summary['most_used'][1]}x)\n\n"
            "**Overall Averages:**\n"
            f"• Dreams: {overall_summary['dreams']:.1f}/day\n"
            f"• Quality: {overall_summary['quality']:.1f}/10\n"
            f"• WBTB: {overall_summary['wbtb']:.1f}/day\n"
            f"• Total Lucid: {overall_summary['lucid']}\n"
            f"• Sleep Time: {overall_summary['sleep_time']:.1f} hrs\n"
            f"• Focus: {overall_summary['focus']:.1f}/10\n"
            f"• Most Used: {overall_summary['most_used'][0]} ({overall_summary['most_used'][1]}x)"
        )

        await send_long_message(ctx, response)

        # Process exactly seven day buckets so missing or duplicate reports do not
        # misalign the two plotted periods.
        anchor = user_local_today(user_id)
        current_start = anchor - timedelta(days=6)
        previous_start = anchor - timedelta(days=13)
        metric_transforms = {
            'dreams': int,
            'quality': lambda value: normalize_quality(value),
            'wbtb': int,
            'lucid': int,
        }
        this_week_data = {'values': {}}
        last_week_data = {'values': {}}
        for metric, transform in metric_transforms.items():
            dates, values = daily_metric_series(this_week, current_start, metric, transform)
            _, previous_values = daily_metric_series(last_week, previous_start, metric, transform)
            this_week_data['dates'] = dates
            this_week_data['values'][metric] = values
            last_week_data['values'][metric] = previous_values
        date_labels = [d.strftime('%d.%m.%y') for d in this_week_data['dates']]

        # Plotting
        plt.figure(figsize=(12, 8))
        metrics = ['dreams', 'quality', 'wbtb', 'lucid']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        x = range(len(date_labels))

        for i, metric in enumerate(metrics, 1):
            plt.subplot(2, 2, i)
            
            # Plot lines
            plt.plot(x, this_week_data['values'][metric], 
                    marker='o', color=colors[0], label='This Week')
            plt.plot(x, last_week_data['values'][metric], 
                    marker='o', color=colors[1], label='Last Week')
            
            # Baseline calculation (FIXED)
            baseline_values = [
                normalize_quality(r['quality']) if metric == 'quality' 
                else int(r[metric])
                for r in average_week
            ]
            baseline = sum(baseline_values) / len(baseline_values)
            plt.axhline(y=baseline, color=colors[2], linestyle='--', label='Baseline')
            
            plt.title(metric.capitalize())
            plt.xticks(x, date_labels, rotation=45)
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        
        # Save and send plot
        datenow = date.today()
        filename = unique_output_path(f'trend_{user_id}_{datenow}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)


    except Exception as e:
        await send_internal_error(ctx, e)



@bot.command(name='delete')
async def delete(ctx, report_date: str = None):
    try:
        user_id = str(ctx.author.id)

        # Check if user has any data
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("You don't have any reports to delete.")
            return

        # Filter reports by the specified date if provided
        if report_date:
            requested_date = parse_report_date(report_date)
            if not requested_date:
                await ctx.send("Invalid date. Use DD.MM.YY or DD.MM.YYYY.")
                return

            # Match equivalent two- and four-digit date formats.
            reports_to_delete = [
                report for report in user_reports[user_id]
                if (parse_report_date(report.get("date", ""))
                    and parse_report_date(report.get("date", "")).date() == requested_date.date())
            ]

            if not reports_to_delete:
                await ctx.send(f"No reports found for the date {report_date}.")
                return

            chosen_index = 0
            if len(reports_to_delete) > 1:
                report_list = "\n".join(
                    [f"{index + 1}: {json.dumps(report, indent=4)}" for index, report in enumerate(reports_to_delete)]
                )
                await send_long_message(
                    ctx,
                    f"Multiple reports found for {report_date}. Choose one to delete by typing its number:\n{report_list}"
                )

                def check(message):
                    return message.author == ctx.author and message.channel == ctx.channel and message.content.isdigit()

                try:
                    response = await bot.wait_for('message', timeout=180, check=check)
                except asyncio.TimeoutError:
                    await ctx.send("No response received in 3 minutes. Deletion cancelled.")
                    return
                chosen_index = int(response.content) - 1

            if 0 <= chosen_index < len(reports_to_delete):
                # Remove the selected report
                deleted_report = reports_to_delete[chosen_index]
                user_reports[user_id].remove(deleted_report)

                # Save the updated data back to the JSON file
                atomic_write_json(DATA_FILE_26, user_reports)

                # Confirm deletion
                await send_long_message(ctx,
                    f"Your report for {report_date} has been deleted. Here is the deleted report for reference:\n"
                    f"```\n{json.dumps(deleted_report, indent=4)}\n```"
                )
            else:
                await ctx.send("Invalid index selected. No report was deleted.")

        else:
            # If no date is provided, delete the chronologically most recent report.
            if user_reports[user_id]:
                most_recent_index = max(
                    range(len(user_reports[user_id])),
                    key=lambda i: parse_report_date(user_reports[user_id][i].get("date", "")) or datetime.min,
                )
                deleted_report = user_reports[user_id].pop(most_recent_index)
                atomic_write_json(DATA_FILE_26, user_reports)

                # Confirm deletion of the most recent report
                await send_long_message(ctx,
                    "Your most recent report has been deleted. Here is the deleted report for reference:\n"
                    f"```\n{json.dumps(deleted_report, indent=4)}\n```"
                )

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='group')
async def group(ctx):
    try:
        # Check if there is any data
        if not user_reports:
            await ctx.send("No reports found for the group.")
            return

        # Date parsing function
        def parse_date(date_str):
            date_str = date_str.strip()
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None

        # Get date ranges
        today = datetime.now().date()
        current_week_dates = [today - timedelta(days=i) for i in range(6, -1, -1)]  # Last 7 days
        last_week_dates = [d - timedelta(days=7) for d in current_week_dates]

        # Collect reports
        current_week_reports = []
        last_week_reports = []
        all_reports = []

        for user_id, reports in user_reports.items():
            for report in reports:
                report_date = parse_date(report['date'])
                if not report_date:
                    continue
                report_date = report_date.date()
                
                if report_date in current_week_dates:
                    current_week_reports.append((report_date, report))
                elif report_date in last_week_dates:
                    last_week_reports.append((report_date, report))
                
                all_reports.append(report)

        # Check data adequacy
        if not current_week_reports or not last_week_reports:
            await ctx.send("Need reports from both this week and last week to compare trends.")
            return

         # TEXT COMPARISON CORRECTIONS
        def calculate_summary(report_tuples):
            """Process list of (date, report) tuples"""
            summary = {
                'dreams': 0,
                'quality': 0,
                'wbtb': 0,
                'lucid': 0,
                'sleep_time': 0,
                'focus': 0,
                'techniques': defaultdict(int)
            }
            count = len(report_tuples)
            sleep_time_count = 0
            focus_count = 0
            
            if count == 0:
                return summary

            for _, report in report_tuples:
                summary['dreams'] += int(report['dreams'])
                summary['quality'] += normalized_quality(report.get('quality', 0)) or 0
                summary['wbtb'] += int(report['wbtb'])
                summary['lucid'] += int(report['lucid'])
                
                # Track sleep_time and focus if present
                if report.get('sleep_time') and str(report.get('sleep_time')).isdigit():
                    summary['sleep_time'] += int(report.get('sleep_time', 0))
                    sleep_time_count += 1
                if report.get('focus') and str(report.get('focus')).isdigit():
                    summary['focus'] += int(report.get('focus', 0))
                    focus_count += 1
                
                technique = report.get('technique', '').lower()
                if technique:
                    summary['techniques'][technique] += 1

            # Calculate averages
            summary['dreams'] = round(summary['dreams'] / count, 2)
            summary['quality'] = round(summary['quality'] / count, 2)
            summary['wbtb'] = round(summary['wbtb'] / count, 2)
            summary['lucid'] = round(summary['lucid'] / count, 2)
            
            if sleep_time_count > 0:
                summary['sleep_time'] = round(summary['sleep_time'] / sleep_time_count, 1)
            if focus_count > 0:
                summary['focus'] = round(summary['focus'] / focus_count, 1)

            # Find most used technique
            if summary['techniques']:
                most_used = max(summary['techniques'], 
                              key=summary['techniques'].get)
                summary['most_used'] = (most_used, summary['techniques'][most_used])
            else:
                summary['most_used'] = ("None", 0)

            return summary

        # Calculate summaries
        current_summary = calculate_summary(current_week_reports)
        last_summary = calculate_summary(last_week_reports)
        
        # Handle overall summary differently
        all_report_tuples = [(None, report) for report in all_reports]  # Convert format
        overall_summary = calculate_summary(all_report_tuples)

        # Format comparison message
        comparison_message = (
            f"Group Trend Analysis:\n\n"
            f"This Week vs Last Week:\n"
            f"- Dreams: {current_summary['dreams']:.2f} ({current_summary['dreams'] - last_summary['dreams']:+.2f})\n"
            f"- Quality: {current_summary['quality']:.2f} ({current_summary['quality'] - last_summary['quality']:+.2f})/10\n"
            f"- WBTB: {current_summary['wbtb']:.2f} ({current_summary['wbtb'] - last_summary['wbtb']:+.2f})\n"
            f"- Lucid Dreams: {current_summary['lucid']:.2f} ({current_summary['lucid'] - last_summary['lucid']:+.2f})\n"
            f"- Sleep Time: {current_summary['sleep_time']:.1f} ({current_summary['sleep_time'] - last_summary['sleep_time']:+.1f}) hrs\n"
            f"- Focus: {current_summary['focus']:.1f} ({current_summary['focus'] - last_summary['focus']:+.1f})/10\n"
            f"- Most Used Technique: {current_summary['most_used'][0]} ({current_summary['most_used'][1]}x)\n\n"
            f"Historical Averages:\n"
            f"- Dreams: {overall_summary['dreams']:.2f}\n"
            f"- Quality: {overall_summary['quality']:.2f}/10\n"
            f"- WBTB: {overall_summary['wbtb']:.2f}\n"
            f"- Lucid Dreams: {overall_summary['lucid']:.2f}\n"
            f"- Sleep Time: {overall_summary['sleep_time']:.1f} hrs\n"
            f"- Focus: {overall_summary['focus']:.1f}/10\n"
            f"- Most Used Technique: {overall_summary['most_used'][0]} ({overall_summary['most_used'][1]}x)"
        )

        await send_long_message(ctx, comparison_message)

        # Process data for plotting
        def prepare_metrics(reports, target_dates):
            grouped = defaultdict(list)
            for report_date, report in reports:
                grouped[report_date].append(report)
            
            metrics = {
                'dreams': [],
                'quality': [],
                'wbtb': [],
                'lucid': []
            }
            
            for report_date in target_dates:
                reports = grouped.get(report_date, [])
                if not reports:
                    for metric in metrics:
                        metrics[metric].append(float('nan'))
                    continue
                
                metrics['dreams'].append(sum(int(r['dreams']) for r in reports) / len(reports))
                metrics['quality'].append(
                    sum(normalized_quality(r.get('quality', 0)) or 0 for r in reports) / len(reports)
                )
                metrics['wbtb'].append(sum(int(r['wbtb']) for r in reports) / len(reports))
                metrics['lucid'].append(sum(int(r['lucid']) for r in reports) / len(reports))
            
            return metrics

        # Get metrics for both weeks
        current_metrics = prepare_metrics(current_week_reports, current_week_dates)
        last_metrics = prepare_metrics(
            [(d + timedelta(days=7), r) for d, r in last_week_reports],  # Offset dates to current week
            current_week_dates
        )

        # Calculate baseline
        def calculate_baseline(reports):
            total = {'dreams': 0, 'quality': 0, 'wbtb': 0, 'lucid': 0}
            count = len(reports)
            if count == 0:
                return {k: 0 for k in total}
            
            for report in reports:
                total['dreams'] += int(report['dreams'])
                total['quality'] += normalized_quality(report.get('quality', 0)) or 0
                total['wbtb'] += int(report['wbtb'])
                total['lucid'] += int(report['lucid'])
            
            return {k: v/count for k, v in total.items()}

        baseline = calculate_baseline(all_reports)

        # Create plot
        plt.figure(figsize=(12, 8))
        metrics_list = ['dreams', 'quality', 'wbtb', 'lucid']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        x = range(len(current_week_dates))
        date_labels = [d.strftime('%d.%m.%y') for d in current_week_dates]

        for i, metric in enumerate(metrics_list, 1):
            plt.subplot(2, 2, i)
            
            # Plot lines
            plt.plot(x, current_metrics[metric], 
                    marker='o', color=colors[0], label='This Week')
            plt.plot(x, last_metrics[metric], 
                    marker='o', color=colors[1], label='Last Week')
            
            # Baseline
            plt.axhline(y=baseline[metric], color=colors[2], 
                       linestyle='--', label='Baseline')
            
            # Formatting
            plt.title(metric.capitalize())
            plt.xticks(x, date_labels, rotation=45)
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        
        # Save and send plot
        datenow = date.today()
        filename = unique_output_path(f'group_trend_{datenow}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

       

    except Exception as e:
        await send_internal_error(ctx, e)



@bot.command(name='personal_recap')
async def personal_recap(ctx):
    try:
        # Get user ID
        user_id = str(ctx.author.id)

        # Check if the user has any data
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("You have no reports to recap for the year.")
            return

        # Get all reports for the user
        reports = user_reports[user_id]
        total_reports = len(reports)

        def calculate_summary(data):
            summary = {"dreams": 0, "quality": 0, "wbtb": 0, "lucid": 0, "techniques": {}}
            count = len(data)

            for report in data:
                summary["dreams"] += int(report.get("dreams", 0))
                summary["quality"] += normalized_quality(report.get("quality", 0)) or 0
                summary["wbtb"] += int(report.get("wbtb", 0))
                summary["lucid"] += int(report.get("lucid", 0))

                technique = report.get("technique", "").lower()
                if technique:
                    summary["techniques"][technique] = summary["techniques"].get(technique, 0) + 1

            # Calculate averages
            summary["quality"] = round(summary["quality"] / count, 2) if count else 0
            summary["dreams"] = round(summary["dreams"] / count, 2) if count else 0
            summary["wbtb"] = round(summary["wbtb"] / count, 2) if count else 0
            summary["lucid"] = round(summary["lucid"] / count, 2) if count else 0
            summary["total_lucid"] = sum(int(report.get("lucid", 0)) for report in data)
            summary["total_dreams"] = sum(int(report.get("dreams", 0)) for report in data)
            summary["total_wbtb"] = sum(int(report.get("wbtb", 0)) for report in data)

            # Find most used technique
            most_used_technique = max(summary["techniques"], key=summary["techniques"].get, default="None")
            most_used_technique_count = summary["techniques"].get(most_used_technique, 0)

            summary["most_used_technique"] = (most_used_technique, most_used_technique_count)
            return summary

        # Calculate yearly summary
        yearly_summary = calculate_summary(reports)

        # Prepare the recap message
        recap_message = (
            f"🌟 Your Recap 2025 🌟\n\n"
            f"Total Reports Submitted: {total_reports}\n"
            f"- Average Dreams per Report: {yearly_summary['dreams']}\n"
            f"- Average Quality per Report: {yearly_summary['quality']}/10\n"
            f"- Average WBTB Attempts: {yearly_summary['wbtb']}\n"
            f"- Average Lucid Dreams per Report: {yearly_summary['lucid']}\n"
            f"- Total Lucid Dreams: {yearly_summary['total_lucid']}\n"
            f"- Total Dreams: {yearly_summary['total_dreams']}\n"
            f"- Total WBTB Attempts: {yearly_summary['total_wbtb']}\n"
            f"- Most Used Technique: {yearly_summary['most_used_technique'][0]} "
            f"({yearly_summary['most_used_technique'][1]} times)"
        )

        await send_long_message(ctx, recap_message)

    except Exception as e:
        await send_internal_error(ctx, e)

@bot.command(name='final_group')
async def final_group(ctx):
    try:
        # Check if there is any data for the group
        if not user_reports:
            await ctx.send("No reports found for the group.")
            return

        # Aggregate all reports
        all_reports = []
        for user_id, reports in user_reports.items():
            all_reports.extend(reports)

        total_reports = len(all_reports)

        def calculate_summary(data):
            summary = {"dreams": 0, "quality": 0, "wbtb": 0, "lucid": 0, "techniques": {}}
            count = len(data)

            for report in data:
                summary["dreams"] += int(report.get("dreams", 0))
                summary["quality"] += normalized_quality(report.get("quality", 0)) or 0
                summary["wbtb"] += int(report.get("wbtb", 0))
                summary["lucid"] += int(report.get("lucid", 0))

                technique = report.get("technique", "").lower()
                if technique:
                    summary["techniques"][technique] = summary["techniques"].get(technique, 0) + 1

            # Calculate averages
            summary["quality"] = round(summary["quality"] / count, 2) if count else 0
            summary["dreams"] = round(summary["dreams"] / count, 2) if count else 0
            summary["wbtb"] = round(summary["wbtb"] / count, 2) if count else 0
            summary["lucid"] = round(summary["lucid"] / count, 2) if count else 0
            summary["total_lucid"] = sum(int(report.get("lucid", 0)) for report in data)
            summary["total_dreams"] = sum(int(report.get("dreams", 0)) for report in data)
            summary["total_wbtb"] = sum(int(report.get("wbtb", 0)) for report in data)

            # Find most used technique
            most_used_technique = max(summary["techniques"], key=summary["techniques"].get, default="None")
            most_used_technique_count = summary["techniques"].get(most_used_technique, 0)

            summary["most_used_technique"] = (most_used_technique, most_used_technique_count)
            return summary

        # Calculate summary for the entire group
        group_summary = calculate_summary(all_reports)

        # Prepare the group recap message
        group_recap_message = (
            f"🌍 Group Recap 2025 🌍\n\n"
            f"Total Reports Submitted: {total_reports}\n"
            f"- Average Dreams per Report: {group_summary['dreams']}\n"
            f"- Average Quality per Report: {group_summary['quality']}/10\n"
            f"- Average WBTB Attempts: {group_summary['wbtb']}\n"
            f"- Average Lucid Dreams per Report: {group_summary['lucid']}\n"
            f"- Total Lucid Dreams: {group_summary['total_lucid']}\n"
            f"- Total Dreams: {group_summary['total_dreams']}\n"
            f"- Total WBTB Attempts: {group_summary['total_wbtb']}\n"
            f"- Most Used Technique: {group_summary['most_used_technique'][0]} "
            f"({group_summary['most_used_technique'][1]} times)"
        )

        await send_long_message(ctx, group_recap_message)

    except Exception as e:
        await send_internal_error(ctx, e)



@bot.command(name='format')
async def format_reports(ctx):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found to format.")
            return

        def parse_date(date_str):
            # Strip any leading/trailing spaces before parsing
            date_str = date_str.strip()
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return None

        # Get and sort the user's reports by date
        reports = user_reports[user_id]
        sorted_reports = sorted(
            reports, 
            key=lambda report: parse_date(report.get("date", "01.01.1970")) or datetime.min
        )
        user_reports[user_id] = sorted_reports
        atomic_write_json(DATA_FILE_26, user_reports)

        await ctx.send(f"Your reports have been successfully sorted by date, {ctx.author.name}.")

    except Exception as e:
        await send_internal_error(ctx, e)

@bot.command(name='format_all')
@commands.has_guild_permissions(manage_guild=True)
async def format_all_reports(ctx):
    try:
        if not user_reports:
            await ctx.send("No reports found to format.")
            return

        def parse_date(date_str):
            # Strip any leading/trailing spaces before parsing
            date_str = date_str.strip()
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue
            return datetime.min  # Use the earliest possible date if parsing fails

        # Iterate through each user's reports and sort them by date
        for user_id, reports in user_reports.items():
            sorted_reports = sorted(
                reports, 
                key=lambda report: parse_date(report.get("date", "01.01.1970"))
            )
            user_reports[user_id] = sorted_reports

        atomic_write_json(DATA_FILE_26, user_reports)

        await ctx.send("All users' reports have been successfully sorted by date.")

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='list')
async def list_reports(ctx):
    user_id = str(ctx.author.id)

    if user_id not in user_reports:
        await ctx.send("You have no reports.")
        return

    # We'll group reports by month-year.
    # Each entry will hold:
    #   - "dates": a dict mapping each reported date -> number of reports that day
    #   - "dreams_total": total number of dreams for that month
    #   - "lucid_total": total number of lucid dream occurrences for that month
    #   - "month": integer month
    #   - "year": integer year
    stats_by_month = {}

    # Process each report
    for report in user_reports[user_id]:
        report_date_str = report.get("date", "").strip().replace(" ", "")
        parsed_report_date = parse_report_date(report_date_str)
        if not parsed_report_date:
            await ctx.send(f"Error parsing date: {report_date_str}. Please check your input.")
            return
        report_date = parsed_report_date.date()

        key = report_date.strftime("%B %Y")  # e.g. "March 2025"
        if key not in stats_by_month:
            stats_by_month[key] = {
                "dates": {},       # date -> count
                "lucid_total": 0,  # total lucid count for the month
                "dreams_total": 0,
                "month": report_date.month,
                "year": report_date.year
            }
        # Count the report for that day:
        stats_by_month[key]["dates"][report_date] = stats_by_month[key]["dates"].get(report_date, 0) + 1
        # Sum up the lucid dreams (assumes the "lucid" field is numeric)
        try:
            lucid = int(report.get("lucid", 0))
        except ValueError:
            lucid = 0
        stats_by_month[key]["lucid_total"] += lucid

        try:
            dreams = int(report.get("dreams", 0))
        except ValueError:
            dreams = 0
        stats_by_month[key]["dreams_total"] += dreams

    # Calculate overall (yearly) statistics across all months
    overall_total_reports = 0
    overall_unique_days = 0
    overall_duplicates = 0
    overall_lucid = 0
    overall_dreams = 0
    for data in stats_by_month.values():
        overall_total_reports += sum(data["dates"].values())
        overall_unique_days += len(data["dates"])
        overall_duplicates += sum(1 for count in data["dates"].values() if count > 1)
        overall_lucid += data["lucid_total"]
        overall_dreams += data["dreams_total"]

    message = "**Yearly Summary**\n"
    message += f"Total Reports: {overall_total_reports}\n"
    message += f"Unique Reported Days: {overall_unique_days}\n"
    message += f"Days with Duplicates: {overall_duplicates}\n"
    message += f"Total Lucid Dreams: {overall_lucid}\n"
    message += f"Total dreams: {overall_dreams}\n\n"

    # Prepare to sort the months (by year, then month)
    month_list = []
    for key, data in stats_by_month.items():
        month_list.append((data["year"], data["month"], key, data))
    month_list.sort()  # sorts by (year, month)

    today = datetime.now().date()

    # Iterate through each month (sorted)
    for (yr, mon, key, data) in month_list:
        # Determine the start and end for detailed day listing.
        # For previous months we use the full month; for the current month, list days only up to today.
        month_start = date(yr, mon, 1)
        if yr == today.year and mon == today.month:
            month_end = today
        else:
            last_day = calendar.monthrange(yr, mon)[1]
            month_end = date(yr, mon, last_day)
        # Build a list of every day from month_start to month_end.
        days_in_period = [month_start + timedelta(days=i) for i in range((month_end - month_start).days + 1)]

        # Determine missing days (i.e. days with no report)
        reported_dates = data["dates"]
        missing_days = [d for d in days_in_period if d not in reported_dates]

        # Create a summary line for the month.
        if not missing_days:
            summary_line = f"**{key}: reported every day | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        else:
            summary_line = f"**{key}: missed {len(missing_days)} day(s) | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        message += summary_line + "\n"

        # Decide whether to include detailed day-by-day info:
        # - Always show for the current month.
        # - For previous months, show details only if there are missing days or if any day has duplicates.
        show_details = False
        if yr == today.year and mon == today.month:
            show_details = True
        else:
            if missing_days or any(count > 1 for count in reported_dates.values()):
                show_details = True

        if show_details:
            for d in days_in_period:
                formatted = d.strftime("%d.%m.%y")
                if d in reported_dates:
                    status = "reported"
                    if reported_dates[d] > 1:
                        status += " Duplicate!"
                    # Determine if any report on this day was lucid.
                    # (Since we only stored a monthly total, we re-scan reports for this day.)
                    lucid_for_day = 0
                    for report in user_reports[user_id]:
                        rep_str = report.get("date", "").strip().replace(" ", "")
                        parsed_rep_date = parse_report_date(rep_str)
                        if not parsed_rep_date:
                            continue
                        rep_date = parsed_rep_date.date()
                        if rep_date == d:
                            try:
                                lucid_for_day += int(report.get("lucid", 0))
                            except ValueError:
                                continue
                    if lucid_for_day > 0:
                        status += " Lucid!"
                    message += f"  {formatted}: {status}\n"
                else:
                    message += f"  {formatted}: MISSING\n"
        message += "\n"

    await send_long_message(ctx, message)



@bot.command(name='score')
async def recall_score(ctx):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        reports = user_reports[user_id]

        this_week, last_week = rolling_report_windows(reports, user_local_today(user_id))
        if not this_week or not last_week:
            await ctx.send("You need reports in both adjacent 7-day periods to compare recall scores.")
            return

        def calculate_total_dreams(data):
            total = 0
            for report in data:
                total += int(report.get("dreams", 0))
            return total

        this_week_total = calculate_total_dreams(this_week)
        last_week_total = calculate_total_dreams(last_week)

        difference = this_week_total - last_week_total
        if difference > 0:
            difference = f"+{difference}"
        elif difference == 0:
            difference = "="

        response = (
            "**Recall score**\n"
            "**This Week vs Last Week:**\n"
            f"Total dreams this week: {this_week_total}({difference})\n"
        )

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)

bot.remove_command('help')

@bot.command(name='help')
async def help_command(ctx):
    """Displays this help message with all available commands"""
    help_message = """
**📖 Dream Journal Bot Command Guide 📖**

`!report` - Submit your daily dream report. Include these fields (* optional):
```date: [DD.MM.YY]*
dreams: [number]
quality: [X/Y or 0-10]
wbtb: [number of attempts]
lucid: [number]
technique: [name]*
notes: [text]*
sleep_time: [number, e.g. 7.5]*
focus: [0-10]*
journal_time: [number]*```
Use `date: DD.MM.YYYY-DD.MM.YYYY` to save the same report for every day in an inclusive range.
You can also paste several complete reports into one `!report` message; begin each one with `date:`.

`!edit` - Edit an existing report. Defaults to most recent, or specify a date. Use `!edit 25.02.26 24.02.26` to move a report to another date.
`!trend` - Compare the latest 7 days with the previous 7 days and your 28-day baseline.
`!baseline [metric]` - Compare your 7-, 28-, and 90-day personal baselines.
`!list` - View your reports month by month (current year).
`!overview [user] [year]` - Compact yearly dashboard; years accept `25` or `2025`.
`!group` - User-weighted group summary for the latest 7 days.
`!delete [date]` - Delete a report.
`!score` - Short recall comparison; full context is in `!trend`.
`!heatmap [year] [metric]` - Calendar heatmap; defaults to current-year lucidity.
`!day_of_week [metric]` - Weekday dashboard for recall, quality, lucidity, and focus.
`!wbtb_impact` - Impact of WBTB on recall/lucidity.
`!conditions [outcome]` - Conditions associated with lucidity by default.
`!sleep_impact` - Sleep-duration associations with recall and lucidity.
`!lagged_effects` - Date D to date D+1 associations using consecutive reports only.
`!streaks` - Reporting, technique, WBTB, lucidity, and recall streaks.
`!lucid_gaps [user] [year]` - Robust intervals between unique lucid nights.
`!lucid_probability [days]` - Historical 7/14/30-day recurrence context.
`!momentum` - Outcomes following lucid versus non-lucid nights.
`!lucid_history [limit]` - Your recent lucid nights.
`!list_month [MM.YY]` - List reports for a month.
`!list_user [user]` - List reports for a user.
`!list_month_user [MM.YY] [user]` - List reports for a month for a user.
`!month [MM.YY]` - Monthly summary and weekly visualization; defaults to current month.
`!month_group [MM.YY]` - User-weighted community month summary.
`!correlate [keyword]` - Whole-word/phrase comparison for a notes keyword.
`!effectiveness [technique] [year]` - Baseline-adjusted technique analysis.
`!interactions [factor1] [factor2]` - Guarded interactions among technique, WBTB, sleep, and focus.
`!matched_nights [technique or keyword]` - Compare similar nights with and without a condition.
`!journaltime` / `!journal` - Descriptive journal-time totals, averages, and streaks.
`!journal_impact [user]` - Same-report and valid next-night journal-time associations.
`!journal_impact_all` - Group within-user journaling association.
`!data_quality [year]` - Missing fields, duplicates, exclusions, and analysis readiness.
`!inactive` - (Admin only) Show users who haven't reported in 2+ weeks.
`!help` - Show this help message.

**🕒 Reminders & Settings**
`/reminder` - (Slash Command) Configure your daily reminder time and timezone.
`/defaults` - (Slash Command) Set or view your default technique for reports.

**📅 Historical Data (2025)**
`!list_25` / `!list_previous` - View your 2025 report list.
Most commands accept `25` or `2025` directly, for example `!overview 25` and `!effectiveness mild 25`.
Legacy `_25` and `_previous` commands remain as compatibility aliases and use the shared analysis system.
"""
    await send_long_message(ctx, help_message)


@bot.command(name='list_month')
async def list_month(ctx, month_str: str):
    """lists all reports for a specific month"""
    try:
        month, year = map(int, month_str.split('.'))
        if year < 100:
            current_year = datetime.now().year
            year += (current_year // 100) * 100
            if year > current_year + 1:
                year -= 100
        target_month = datetime(year=year, month=month, day=1)
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
        return

    user_id = str(ctx.author.id)
    if user_id not in user_reports:
        await ctx.send("You have no reports.")
        return

    reports = []
    for report in user_reports[user_id]:
        report_date_str = report.get('date', '').strip()
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
            reports.append(report)

    if not reports:
        await ctx.send(f"No reports found for {target_month.strftime('%B %Y')}.")
        return

    response = f"**Reports for {target_month.strftime('%B %Y')}:**\n"
    for report in reports:
        date_str = report.get('date', 'Unbekanntes Datum')
        dreams = report.get('dreams', '0')
        lucid = report.get('lucid', '0')
        response += f"- {date_str}: {dreams} dreams, {lucid} lucid\n"
    
    await send_long_message(ctx, response)

@bot.command(name='list_user')
async def list_user(ctx, user: discord.User):
    """Lists all reports for a specific user."""
    user_id = str(user.id)
    if user_id not in user_reports:
        await ctx.send("You have no reports.")
        return

    # We'll group reports by month-year.
    # Each entry will hold:
    #   - "dates": a dict mapping each reported date -> number of reports that day
    #   - "dreams_total": total number of dreams for that month
    #   - "lucid_total": total number of lucid dream occurrences for that month
    #   - "month": integer month
    #   - "year": integer year
    stats_by_month = {}

    # Process each report
    for report in user_reports[user_id]:
        report_date_str = report.get("date", "").strip().replace(" ", "")
        parsed_report_date = parse_report_date(report_date_str)
        if not parsed_report_date:
            await ctx.send(f"Error parsing date: {report_date_str}. Please check your input.")
            return
        report_date = parsed_report_date.date()

        key = report_date.strftime("%B %Y")  # e.g. "March 2025"
        if key not in stats_by_month:
            stats_by_month[key] = {
                "dates": {},       # date -> count
                "lucid_total": 0,  # total lucid count for the month
                "dreams_total": 0,
                "month": report_date.month,
                "year": report_date.year
            }
        # Count the report for that day:
        stats_by_month[key]["dates"][report_date] = stats_by_month[key]["dates"].get(report_date, 0) + 1
        # Sum up the lucid dreams (assumes the "lucid" field is numeric)
        try:
            lucid = int(report.get("lucid", 0))
        except ValueError:
            lucid = 0
        stats_by_month[key]["lucid_total"] += lucid

        try:
            dreams = int(report.get("dreams", 0))
        except ValueError:
            dreams = 0
        stats_by_month[key]["dreams_total"] += dreams

    # Calculate overall (yearly) statistics across all months
    overall_total_reports = 0
    overall_unique_days = 0
    overall_duplicates = 0
    overall_lucid = 0
    overall_dreams = 0
    for data in stats_by_month.values():
        overall_total_reports += sum(data["dates"].values())
        overall_unique_days += len(data["dates"])
        overall_duplicates += sum(1 for count in data["dates"].values() if count > 1)
        overall_lucid += data["lucid_total"]
        overall_dreams += data["dreams_total"]

    message = "**Yearly Summary**\n"
    message += f"Total Reports: {overall_total_reports}\n"
    message += f"Unique Reported Days: {overall_unique_days}\n"
    message += f"Days with Duplicates: {overall_duplicates}\n"
    message += f"Total Lucid Dreams: {overall_lucid}\n"
    message += f"Total dreams: {overall_dreams}\n\n"

    # Prepare to sort the months (by year, then month)
    month_list = []
    for key, data in stats_by_month.items():
        month_list.append((data["year"], data["month"], key, data))
    month_list.sort()  # sorts by (year, month)

    today = datetime.now().date()

    # Iterate through each month (sorted)
    for (yr, mon, key, data) in month_list:
        # Determine the start and end for detailed day listing.
        # For previous months we use the full month; for the current month, list days only up to today.
        month_start = date(yr, mon, 1)
        if yr == today.year and mon == today.month:
            month_end = today
        else:
            last_day = calendar.monthrange(yr, mon)[1]
            month_end = date(yr, mon, last_day)
        # Build a list of every day from month_start to month_end.
        days_in_period = [month_start + timedelta(days=i) for i in range((month_end - month_start).days + 1)]

        # Determine missing days (i.e. days with no report)
        reported_dates = data["dates"]
        missing_days = [d for d in days_in_period if d not in reported_dates]

        # Create a summary line for the month.
        if not missing_days:
            summary_line = f"**{key}: reported every day | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        else:
            summary_line = f"**{key}: missed {len(missing_days)} day(s) | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        message += summary_line + "\n"

        # Decide whether to include detailed day-by-day info:
        # - Always show for the current month.
        # - For previous months, show details only if there are missing days or if any day has duplicates.
        show_details = False
        if yr == today.year and mon == today.month:
            show_details = True
        else:
            if missing_days or any(count > 1 for count in reported_dates.values()):
                show_details = True

        if show_details:
            for d in days_in_period:
                formatted = d.strftime("%d.%m.%y")
                if d in reported_dates:
                    status = "reported"
                    if reported_dates[d] > 1:
                        status += " Duplicate!"
                    # Determine if any report on this day was lucid.
                    # (Since we only stored a monthly total, we re-scan reports for this day.)
                    lucid_for_day = 0
                    for report in user_reports[user_id]:
                        rep_str = report.get("date", "").strip().replace(" ", "")
                        parsed_rep_date = parse_report_date(rep_str)
                        if not parsed_rep_date:
                            continue
                        rep_date = parsed_rep_date.date()
                        if rep_date == d:
                            try:
                                lucid_for_day += int(report.get("lucid", 0))
                            except ValueError:
                                continue
                    if lucid_for_day > 0:
                        status += " Lucid!"
                    message += f"  {formatted}: {status}\n"
                else:
                    message += f"  {formatted}: MISSING\n"
        message += "\n"

    await send_long_message(ctx, message)



@bot.command(name='list_month_user')
async def list_month_user(ctx, month_str: str, user: discord.User):
    """Lists all reports from a user for any month"""
    try:
        month, year = map(int, month_str.split('.'))
        if year < 100:
            current_year = datetime.now().year
            year += (current_year // 100) * 100
            if year > current_year + 1:
                year -= 100
        target_month = datetime(year=year, month=month, day=1)
    except (ValueError, TypeError, ZeroDivisionError, IndexError):
        await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
        return

    user_id = str(user.id)
    if user_id not in user_reports:
        await ctx.send(f"No reports found for {user.display_name}.")
        return

    reports = []
    for report in user_reports[user_id]:
        report_date_str = report.get('date', '').strip()
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
            reports.append(report)

    if not reports:
        await ctx.send(f"No reports found for {user.display_name} im {target_month.strftime('%B %Y')}.")
        return

    response = f"**Reports for {user.display_name} in {target_month.strftime('%B %Y')}:**\n"
    for report in reports:
        date_str = report.get('date', 'Unbekanntes Datum')
        dreams = report.get('dreams', '0')
        lucid = report.get('lucid', '0')
        response += f"- {date_str}: {dreams} dreams, {lucid} lucid\n"
    
    await send_long_message(ctx, response)

@bot.command(name='month')
async def month_trend(ctx, month_str: str = None):
    """Shows monthly trends like !trend"""
    user_id = str(ctx.author.id)
    if user_id not in user_reports or not user_reports[user_id]:
        await ctx.send("No reports found.")
        return

    # Monatsbestimmung
    if month_str:
        try:
            month, year = map(int, month_str.split('.'))
            if year < 100:
                current_year = datetime.now().year
                year += (current_year // 100) * 100
                if year > current_year + 1:
                    year -= 100
            target_month = datetime(year=year, month=month, day=1)
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
            return
    else:
        now = datetime.now()
        target_month = datetime(year=now.year, month=now.month, day=1)

    # Berichte filtern
    reports = []
    for report in user_reports[user_id]:
        report_date_str = report.get('date', '').strip()
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
            reports.append(report)

    if not reports:
        await ctx.send(f"No reports found for {target_month.strftime('%B %Y')}.")
        return

    # Statistiken berechnen
    def calculate_summary(data):
        summary = {
            "dreams": 0,
            "quality": 0,
            "wbtb": 0,
            "lucid": 0,
            "sleep_time": 0,
            "focus": 0,
            "techniques": defaultdict(int)
        }
        count = len(data)
        sleep_time_count = 0
        focus_count = 0
        if count == 0:
            return summary

        for report in data:
            summary["dreams"] += int(report.get("dreams", 0))
            
            quality = report.get("quality", "0")
            if '/' in quality:
                parts = quality.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1]) if len(parts) > 1 else 10
                summary["quality"] += (numerator / denominator) * 10
            else:
                summary["quality"] += int(quality)
            
            summary["wbtb"] += int(report.get("wbtb", 0))
            summary["lucid"] += int(report.get("lucid", 0))
            
            # Track sleep_time and focus if present
            if report.get("sleep_time") and str(report.get("sleep_time")).isdigit():
                summary["sleep_time"] += int(report.get("sleep_time", 0))
                sleep_time_count += 1
            if report.get("focus") and str(report.get("focus")).isdigit():
                summary["focus"] += int(report.get("focus", 0))
                focus_count += 1
            
            technique = report.get("technique", "").lower()
            if technique:
                summary["techniques"][technique] += 1

        summary["dreams"] = round(summary["dreams"] / count, 2)
        summary["quality"] = round(summary["quality"] / count, 2)
        summary["wbtb"] = round(summary["wbtb"] / count, 2)
        summary["lucid"] = round(summary["lucid"] / count, 2)
        
        if sleep_time_count > 0:
            summary["sleep_time"] = round(summary["sleep_time"] / sleep_time_count, 1)
        if focus_count > 0:
            summary["focus"] = round(summary["focus"] / focus_count, 1)
        
        if summary["techniques"]:
            most_used = max(summary["techniques"], key=summary["techniques"].get)
            summary["most_used"] = (most_used, summary["techniques"][most_used])
        else:
            summary["most_used"] = ("None", 0)
        
        return summary

    month_summary = calculate_summary(reports)
    response = (
        f"📊 **Monthly statistic for {target_month.strftime('%B %Y')}**\n\n"
        f"Averages:\n"
        f"• dreams: {month_summary['dreams']}\n"
        f"• quality: {month_summary['quality']}/10\n"
        f"• wbtb: {month_summary['wbtb']}\n"
        f"• lucids: {month_summary['lucid']}\n"
        f"• sleep time: {month_summary['sleep_time']} hrs\n"
        f"• focus: {month_summary['focus']}/10\n"
        f"• technique: {month_summary['most_used'][0]} ({month_summary['most_used'][1]}x)"
    )
    await send_long_message(ctx, response)

    # Diagramm erstellen
    dates = []
    dreams = []
    quality = []
    wbtb = []
    lucid = []

    for report in reports:
        report_date_str = report.get('date', '')
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date:
            dates.append(parsed_date)
            dreams.append(int(report.get('dreams', 0)))
            
            qual = report.get('quality', '0')
            if '/' in qual:
                num, den = qual.split('/')[:2]
                qual_val = (int(num) / int(den)) * 10
            else:
                qual_val = int(qual)
            quality.append(qual_val)
            
            wbtb.append(int(report.get('wbtb', 0)))
            lucid.append(int(report.get('lucid', 0)))

    # Sortieren nach Datum
    sorted_data = sorted(zip(dates, dreams, quality, wbtb, lucid), key=lambda x: x[0])
    if not sorted_data:
        return

    dates, dreams, quality, wbtb, lucid = zip(*sorted_data)

    plt.figure(figsize=(12, 8))
    metrics = ['dreams', 'quality', 'wbtb', 'lucid']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = []
        if metric == 'dreams':
            values = dreams
        elif metric == 'quality':
            values = quality
        elif metric == 'wbtb':
            values = wbtb
        else:
            values = lucid
        
        plt.plot(dates, values, marker='o', color='#1f77b4')
        plt.title(metric.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    
    filename = unique_output_path(f'month_{user_id}_{target_month.strftime("%m_%Y")}')
    await asyncio.to_thread(plt.savefig, filename)
    plt.close()
    
    await send_generated_file(ctx, filename)


@bot.command(name='month_group')
async def month_group(ctx, month_str: str = None):
    """Shows monthly trends for the entire group."""
    if not user_reports:
        await ctx.send("No reports found for the group.")
        return

    # Determine the target month
    if month_str:
        try:
            month, year = map(int, month_str.split('.'))
            if year < 100:
                current_year = datetime.now().year
                year += (current_year // 100) * 100
                if year > current_year + 1:
                    year -= 100
            target_month = datetime(year=year, month=month, day=1)
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
            return
    else:
        now = datetime.now()
        target_month = datetime(year=now.year, month=now.month, day=1)

    # Filter reports for the target month
    reports = []
    for user_id, user_data in user_reports.items():
        for report in user_data:
            report_date_str = report.get('date', '').strip()
            parsed_date = None
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    parsed_date = datetime.strptime(report_date_str, fmt)
                    break
                except (ValueError, TypeError, ZeroDivisionError, IndexError):
                    continue
            if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
                reports.append(report)

    if not reports:
        await ctx.send(f"No reports found for {target_month.strftime('%B %Y')}.")
        return

    # Calculate statistics
    def calculate_summary(data):
        summary = {
            "dreams": 0,
            "quality": 0,
            "wbtb": 0,
            "lucid": 0,
            "techniques": defaultdict(int)
        }
        count = len(data)
        if count == 0:
            return summary

        for report in data:
            summary["dreams"] += int(report.get("dreams", 0))
            
            quality = report.get("quality", "0")
            if '/' in quality:
                parts = quality.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1]) if len(parts) > 1 else 10
                summary["quality"] += (numerator / denominator) * 10
            else:
                summary["quality"] += int(quality)
            
            summary["wbtb"] += int(report.get("wbtb", 0))
            summary["lucid"] += int(report.get("lucid", 0))  # Sum lucids
            
            technique = report.get("technique", "").lower()
            if technique:
                summary["techniques"][technique] += 1

        # Calculate averages for all metrics except lucids
        summary["dreams"] = round(summary["dreams"] / count, 2)
        summary["quality"] = round(summary["quality"] / count, 2)
        summary["wbtb"] = round(summary["wbtb"] / count, 2)
        
        if summary["techniques"]:
            most_used = max(summary["techniques"], key=summary["techniques"].get)
            summary["most_used"] = (most_used, summary["techniques"][most_used])
        else:
            summary["most_used"] = ("None", 0)
        
        return summary

    month_summary = calculate_summary(reports)
    response = (
        f"📊 **Group Monthly Statistics for {target_month.strftime('%B %Y')}**\n\n"
        f"Averages:\n"
        f"• Dreams: {month_summary['dreams']}\n"
        f"• Quality: {month_summary['quality']}/10\n"
        f"• WBTB: {month_summary['wbtb']}\n"
        f"• Total Lucids: {month_summary['lucid']}\n"  # Total, not average
        f"• Technique: {month_summary['most_used'][0]} ({month_summary['most_used'][1]}x)"
    )
    await send_long_message(ctx, response)

    # Prepare data for plotting
    dates = []
    dreams = []
    quality = []
    wbtb = []
    lucid = []

    for report in reports:
        report_date_str = report.get('date', '')
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date:
            dates.append(parsed_date)
            dreams.append(int(report.get('dreams', 0)))
            
            qual = report.get('quality', '0')
            if '/' in qual:
                num, den = qual.split('/')[:2]
                qual_val = (int(num) / int(den)) * 10
            else:
                qual_val = int(qual)
            quality.append(qual_val)
            
            wbtb.append(int(report.get('wbtb', 0)))
            lucid.append(int(report.get('lucid', 0)))

    # Sort data by date
    sorted_data = sorted(zip(dates, dreams, quality, wbtb, lucid), key=lambda x: x[0])
    if not sorted_data:
        return

    dates, dreams, quality, wbtb, lucid = zip(*sorted_data)

    # Calculate averages for plotting
    unique_dates = sorted(set(dates))
    avg_dreams = []
    avg_quality = []
    avg_wbtb = []
    total_lucid = []

    for report_date in unique_dates:
        indices = [i for i, d in enumerate(dates) if d == report_date]
        avg_dreams.append(round(sum(dreams[i] for i in indices) / len(indices), 2))
        avg_quality.append(round(sum(quality[i] for i in indices) / len(indices), 2))
        avg_wbtb.append(round(sum(wbtb[i] for i in indices) / len(indices), 2))
        total_lucid.append(sum(lucid[i] for i in indices))  # Sum lucids

    # Create the plot
    plt.figure(figsize=(12, 8))
    metrics = ['dreams', 'quality', 'wbtb', 'lucid']
    values = [avg_dreams, avg_quality, avg_wbtb, total_lucid]
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        plt.plot(unique_dates, values[i - 1], marker='o', color='#1f77b4')
        plt.title(metric.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.tight_layout()
    
    filename = unique_output_path(f'month_group_{target_month.strftime("%m_%Y")}')
    await asyncio.to_thread(plt.savefig, filename)
    plt.close()
    
    await send_generated_file(ctx, filename)


@bot.command(name='dreams')
async def dreams(ctx):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        reports = user_reports[user_id]


        def calculate_total_dreams(data):
            total = 0
            for report in data:
                total += int(report.get("dreams", 0))
            return total

        total = calculate_total_dreams(reports)

        response = (
            f"Total dreams: {total}\n"
        )

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='correlate')
async def correlate(ctx, *, keyword: str):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        reports = user_reports[user_id]
        
        # Helper to normalize quality
        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        # Separate reports
        with_keyword = []
        without_keyword = []
        for report in reports:
            if keyword.lower() in report.get('notes', '').lower():
                with_keyword.append(report)
            else:
                without_keyword.append(report)

        if not with_keyword:
            await ctx.send(f"No reports found containing the keyword '{keyword}'.")
            return

        # Calculation logic
        def calculate_stats(report_list):
            if not report_list:
                return {"avg_dreams": 0, "avg_quality": 0, "lucid_rate": 0, "count": 0}
            
            total_dreams = sum(int(r.get('dreams', 0)) for r in report_list)
            total_quality = sum(normalize_quality(r.get('quality', '0')) for r in report_list)
            total_lucid = sum(int(r.get('lucid', 0)) > 0 for r in report_list)
            count = len(report_list)
            
            return {
                "avg_dreams": round(total_dreams / count, 2) if count > 0 else 0,
                "avg_quality": round(total_quality / count, 2) if count > 0 else 0,
                "lucid_rate": round((total_lucid / count) * 100, 1) if count > 0 else 0,
                "count": count
            }

        stats_with = calculate_stats(with_keyword)
        stats_without = calculate_stats(without_keyword)

        # Formatting the response
        response = (
            f"**Correlation Analysis for '{keyword}'**\n\n"
            f"**When notes INCLUDE '{keyword}'** ({stats_with['count']} reports):\n"
            f"• Avg. Dream Recall: **{stats_with['avg_dreams']}**\n"
            f"• Avg. Quality: **{stats_with['avg_quality']:.1f}/10**\n"
            f"• Lucid Dream Rate: **{stats_with['lucid_rate']}%**\n\n"
            f"**When notes DO NOT INCLUDE '{keyword}'** ({stats_without['count']} reports):\n"
            f"• Avg. Dream Recall: **{stats_without['avg_dreams']}**\n"
            f"• Avg. Quality: **{stats_without['avg_quality']:.1f}/10**\n"
            f"• Lucid Dream Rate: **{stats_without['lucid_rate']}%**"
        )

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)



@bot.command(name='effectiveness')
async def effectiveness(ctx, *, technique: str = None):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        reports = user_reports[user_id]
        failure_keywords = ["fell asleep", "couldn't focus", "gave up", "skipped", "forgot", "did nothing", "no tech"]

        def calculate_lucid_rate(report_list):
            if not report_list:
                return 0, 0
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            total_count = len(report_list)
            rate = round((lucid_count / total_count) * 100, 1) if total_count > 0 else 0
            return rate, total_count

        # If no technique is specified, rank all techniques
        if not technique:
            all_techniques = defaultdict(list)
            for report in reports:
                # Normalize technique so order doesn't matter (e.g., 'mild, ssild' == 'ssild, mild')
                tech_entry = normalize_technique(report.get('technique', ''))
                if tech_entry and tech_entry != 'none':
                    all_techniques[tech_entry].append(report)
            
            if not all_techniques:
                await ctx.send("No techniques found in your reports.")
                return

            ranked_techniques = []
            for tech, tech_reports in all_techniques.items():
                successful_execution = [r for r in tech_reports if not any(keyword in r.get('notes', '').lower() for keyword in failure_keywords)]
                success_rate, success_count = calculate_lucid_rate(successful_execution)
                
                total_count = len(tech_reports)
                consistency = round((success_count / total_count) * 100, 1) if total_count > 0 else 0
                
                # Scoring: success_rate is weighted higher, but consistency is a factor
                # This is a simple scoring model, can be adjusted
                score = (success_rate * 0.7) + (consistency * 0.3)

                ranked_techniques.append({
                    'name': tech,
                    'rate': success_rate,
                    'consistency': consistency,
                    'uses': total_count,
                    'score': score
                })
            
            ranked_techniques.sort(key=lambda x: x['score'], reverse=True)

            response = "**Your Personal Technique Effectiveness Rankings**\n"
            response += "_(Ranked by a combined score of success rate and execution consistency)_\n\n"
            for i, tech_data in enumerate(ranked_techniques, 1):
                response += f"{i}. **{tech_data['name'].upper()}** - **{tech_data['rate']}%** Success Rate ({tech_data['consistency']}% consistency over {tech_data['uses']} uses)\n"
            
            await send_long_message(ctx, response)
            return

        # If a technique is specified, do the detailed analysis
        # Normalize the input technique for comparison
        technique_normalized = normalize_technique(technique)
        technique_reports = [r for r in reports if normalize_technique(r.get('technique', '')) == technique_normalized]

        if not technique_reports:
            await ctx.send(f"No reports found where you used the '{technique.upper()}' technique.")
            return

        successful_execution = []
        failed_execution = []

        for report in technique_reports:
            notes = report.get('notes', '').lower()
            if any(keyword in notes for keyword in failure_keywords):
                failed_execution.append(report)
            else:
                successful_execution.append(report)

        success_rate, success_count = calculate_lucid_rate(successful_execution)
        failure_rate, failure_count = calculate_lucid_rate(failed_execution)

        response = f"**Effectiveness Analysis for {technique.upper()}**\n\n"

        if success_count > 0:
            response += f"**Proper Execution** ({success_count} reports):\n"
            response += f"When you completed the technique properly, your success rate was **{success_rate}%**.\n\n"
        else:
            response += "No reports found where you properly completed the technique.\n\n"

        if failure_count > 0:
            response += f"**Incomplete/Failed Execution** ({failure_count} reports):\n"
            response += f"When you were distracted or didn't complete the technique, your success rate was **{failure_rate}%**.\n"
        else:
            response += "No reports found with incomplete or failed attempts.\n"

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='wbtb_impact')
async def wbtb_impact(ctx, user: discord.User = None):
    """Visualizes the impact of WBTB on dream recall and lucidity."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send(f"No reports found for {target_user.display_name}.")
            return

        reports = user_reports[user_id]

        # Group reports by WBTB count
        wbtb_groups = defaultdict(list)
        for r in reports:
            wbtb_count = int(r.get('wbtb', 0))
            wbtb_groups[wbtb_count].append(r)

        if not wbtb_groups:
            await ctx.send("No WBTB data found in the reports.")
            return

        # Calculate stats for each group
        wbtb_stats = {}
        for count, group_reports in wbtb_groups.items():
            total_dreams = sum(int(r.get('dreams', 0)) for r in group_reports)
            total_lucid = sum(int(r.get('lucid', 0)) for r in group_reports)
            num_reports = len(group_reports)
            
            wbtb_stats[count] = {
                'avg_dreams': round(total_dreams / num_reports, 2) if num_reports > 0 else 0,
                'avg_lucid': round(total_lucid / num_reports, 2) if num_reports > 0 else 0,
                'report_count': num_reports
            }

        sorted_wbtb_counts = sorted(wbtb_stats.keys())

        # Create x-tick labels with report counts
        xticklabels = []
        for c in sorted_wbtb_counts:
            report_count = wbtb_stats[c]['report_count']
            xticklabels.append(f"{c}\n(n={report_count})")

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Bar chart for average dreams
        ax1.set_xlabel('Number of WBTB Attempts (n=number of reports)')
        ax1.set_ylabel('Average Dream Recall', color='tab:blue')
        ax1.bar(
            [x - 0.2 for x in sorted_wbtb_counts], 
            [wbtb_stats[c]['avg_dreams'] for c in sorted_wbtb_counts], 
            width=0.4, 
            color='tab:blue', 
            label='Avg Dreams'
        )
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        # Bar chart for average lucidity on a second y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Lucid Dreams', color='tab:orange')
        ax2.bar(
            [x + 0.2 for x in sorted_wbtb_counts], 
            [wbtb_stats[c]['avg_lucid'] for c in sorted_wbtb_counts], 
            width=0.4, 
            color='tab:orange', 
            label='Avg Lucid'
        )
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        fig.tight_layout()
        plt.title(f'WBTB Impact for {target_user.display_name}')
        ax1.set_xticks(sorted_wbtb_counts)
        ax1.set_xticklabels(xticklabels)
        
        # Save and send plot
        filename = unique_output_path(f'wbtb_impact_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='lucid_factors')
async def lucid_factors(ctx, user: discord.User = None):
    """Analyzes how dream recall and quality correlate with lucidity."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send(f"No reports found for {target_user.display_name}.")
            return

        reports = user_reports[user_id]
        
        if len(reports) < 5: # Need a minimum number of reports for meaningful analysis
            await ctx.send("You need at least 5 reports for this analysis.")
            return

        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        # Calculate averages
        total_dreams = sum(int(r.get('dreams', 0)) for r in reports)
        total_quality = sum(normalize_quality(r.get('quality', '0')) for r in reports)
        avg_dreams = total_dreams / len(reports)
        avg_quality = total_quality / len(reports)

        # Categorize reports
        categories = {
            "High Recall / High Quality": [],
            "High Recall / Low Quality": [],
            "Low Recall / High Quality": [],
            "Low Recall / Low Quality": []
        }

        for r in reports:
            recall = int(r.get('dreams', 0))
            quality = normalize_quality(r.get('quality', '0'))
            
            if recall >= avg_dreams and quality >= avg_quality:
                categories["High Recall / High Quality"].append(r)
            elif recall >= avg_dreams and quality < avg_quality:
                categories["High Recall / Low Quality"].append(r)
            elif recall < avg_dreams and quality >= avg_quality:
                categories["Low Recall / High Quality"].append(r)
            else:
                categories["Low Recall / Low Quality"].append(r)

        # Calculate lucid rates
        lucid_rates = {}
        for category, report_list in categories.items():
            if not report_list:
                lucid_rates[category] = 0
                continue
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            lucid_rates[category] = (lucid_count / len(report_list)) * 100

        # Plotting
        labels = list(lucid_rates.keys())
        rates = list(lucid_rates.values())
        report_counts = [len(categories[cat]) for cat in labels]
        
        x_labels_with_counts = [f"{label}\n(n={count})" for label, count in zip(labels, report_counts)]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(labels, rates, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        
        plt.ylabel('Lucid Dream Rate (%)')
        plt.title(f'Lucid Dream Factors for {target_user.display_name}')
        plt.xticks(range(len(labels)), x_labels_with_counts, rotation=0) # Keep labels horizontal
        plt.ylim(0, max(rates) * 1.15 if max(rates) > 0 else 10) # Adjust y-axis limit

        # Add percentage labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom' if yval > 0 else 'top')

        plt.tight_layout()
        
        filename = unique_output_path(f'lucid_factors_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='overview')
async def overview(ctx, user: discord.User = None):
    """Provides a comprehensive overview of the user's dream journal."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send(f"No reports found for {target_user.display_name}.")
            return

        reports = user_reports[user_id]

        # --- Helper Functions ---
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None

        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        # --- 1. Text Summary ---
        summary_message = f"### 🌟 **Your Dream Journal Overview for {target_user.display_name}** 🌟\n\n"

        # 1.1 Recent Trends
        this_week, last_week = rolling_report_windows(reports, user_local_today(user_id))
        if this_week and last_week:
            
            def calculate_simple_summary(data):
                summary = {"dreams": 0, "quality": 0, "lucid": 0}
                for r in data:
                    summary["dreams"] += int(r.get("dreams", 0))
                    summary["quality"] += normalize_quality(r.get("quality", "0"))
                    summary["lucid"] += int(r.get("lucid", 0))
                count = len(data)
                if count > 0:
                    summary["dreams"] /= count
                    summary["quality"] /= count
                return summary

            this_summary = calculate_simple_summary(this_week)
            last_summary = calculate_simple_summary(last_week)
            
            summary_message += "**📈 Recent Trends (Last 7 vs. Previous 7 Days):**\n"
            summary_message += f"*   **Dream Recall:** `{this_summary['dreams']:.1f}` vs `{last_summary['dreams']:.1f}` (`{this_summary['dreams'] - last_summary['dreams']:.1f}`)\n"
            summary_message += f"*   **Dream Quality:** `{this_summary['quality']:.1f}` vs `{last_summary['quality']:.1f}` (`{this_summary['quality'] - last_summary['quality']:.1f}`)\n"
            summary_message += f"*   **Lucid Dreams:** `{this_summary['lucid']}` vs `{last_summary['lucid']}` (`{this_summary['lucid'] - last_summary['lucid']:+}`)\n\n"

        # 1.2 Top 5 Techniques
        all_techniques = defaultdict(list)
        for report in reports:
            tech_entry = report.get('technique', '').lower().strip()
            if tech_entry:
                all_techniques[tech_entry].append(report)
        
        ranked_techniques = []
        for tech, tech_reports in all_techniques.items():
            lucid_count = sum(1 for r in tech_reports if int(r.get('lucid', 0)) > 0)
            total_count = len(tech_reports)
            rate = (lucid_count / total_count) * 100 if total_count > 0 else 0
            ranked_techniques.append({'name': tech, 'rate': rate, 'uses': total_count})
        
        ranked_techniques.sort(key=lambda x: x['rate'], reverse=True)
        
        summary_message += "**🚀 Top 5 Most Effective Techniques:**\n"
        for i, tech in enumerate(ranked_techniques[:5], 1):
            summary_message += f"*   {i}. **{tech['name'].upper()}**: `{tech['rate']:.1f}%` Lucid Rate ({tech['uses']} uses)\n"
        summary_message += "\n"

        # 1.3 Key Lucidity Factors
        total_dreams = sum(int(r.get('dreams', 0)) for r in reports)
        total_quality = sum(normalize_quality(r.get('quality', '0')) for r in reports)
        avg_dreams = total_dreams / len(reports)
        avg_quality = total_quality / len(reports)
        
        categories = {
            "High Recall / High Quality": [], "High Recall / Low Quality": [],
            "Low Recall / High Quality": [], "Low Recall / Low Quality": []
        }
        for r in reports:
            recall = int(r.get('dreams', 0))
            quality = normalize_quality(r.get('quality', '0'))
            if recall >= avg_dreams and quality >= avg_quality: categories["High Recall / High Quality"].append(r)
            elif recall >= avg_dreams and quality < avg_quality: categories["High Recall / Low Quality"].append(r)
            elif recall < avg_dreams and quality >= avg_quality: categories["Low Recall / High Quality"].append(r)
            else: categories["Low Recall / Low Quality"].append(r)

        summary_message += "**📊 Key Lucidity Factors:**\n"
        for category, report_list in categories.items():
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            rate = (lucid_count / len(report_list)) * 100 if report_list else 0
            summary_message += f"*   **{category}**: `{rate:.1f}%` Lucid Rate ({len(report_list)} reports)\n"
        summary_message += "\n"

        # 1.4 Journaling Consistency
        today = datetime.now().date()
        current_month_reports = [r for r in reports if parse_date(r.get('date')) and parse_date(r.get('date')).month == today.month and parse_date(r.get('date')).year == today.year]
        reported_days = {parse_date(r.get('date')).day for r in current_month_reports}
        all_month_days = set(range(1, today.day + 1))
        missed_days_count = len(all_month_days - reported_days)

        summary_message += f"**📓 Journaling Consistency ({today.strftime('%B %Y')}):**\n"
        summary_message += f"*   **Total Reports:** `{len(reports)}`\n"
        summary_message += f"*   **Reports This Month:** `{len(current_month_reports)}`\n"
        summary_message += f"*   **Missed Days This Month:** `{missed_days_count}`\n"

        await send_long_message(ctx, summary_message)

        # --- 2. Graph Generation ---

        # Image 1: Main Overview
        fig1, axs1 = plt.subplots(3, 1, figsize=(12, 18))
        fig1.suptitle(f"Main Overview for {target_user.display_name}", fontsize=16)
        
        # Graph 1.1: Recall Trend
        if this_week and last_week:
            anchor = user_local_today(user_id)
            current_start = anchor - timedelta(days=6)
            previous_start = anchor - timedelta(days=13)
            current_dates, current_dreams = daily_metric_series(this_week, current_start, 'dreams', int)
            _, previous_dreams = daily_metric_series(last_week, previous_start, 'dreams', int)
            _, current_quality = daily_metric_series(this_week, current_start, 'quality', normalize_quality)
            _, previous_quality = daily_metric_series(last_week, previous_start, 'quality', normalize_quality)
            this_week_data = {'dreams': current_dreams, 'quality': current_quality}
            last_week_data = {'dreams': previous_dreams, 'quality': previous_quality}
            date_labels = [d.strftime('%a') for d in current_dates]
            axs1[0].plot(date_labels, this_week_data['dreams'], marker='o', label='This Week')
            axs1[0].plot(date_labels, last_week_data['dreams'], marker='o', linestyle='--', label='Last Week')
            axs1[0].set_title("Dream Recall Trend")
            axs1[0].set_ylabel("Average Dream Recall")
            axs1[0].legend()

        # Graph 1.2: Quality Trend
        if this_week and last_week:
            axs1[1].plot(date_labels, this_week_data['quality'], marker='s', color='r', label='This Week')
            axs1[1].plot(date_labels, last_week_data['quality'], marker='s', linestyle='--', color='m', label='Last Week')
            axs1[1].set_title("Dream Quality Trend")
            axs1[1].set_ylabel("Average Dream Quality (/10)")
            axs1[1].legend()

        # Graph 1.3: Day of Week
        dow_stats = defaultdict(lambda: {'lucid': 0, 'count': 0})
        for r in reports:
            dt = parse_date(r.get('date'))
            if dt:
                dow = dt.strftime('%A')
                dow_stats[dow]['lucid'] += int(r.get('lucid', 0)) > 0
                dow_stats[dow]['count'] += 1
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_rates = [(dow_stats[d]['lucid'] / dow_stats[d]['count'] * 100 if dow_stats[d]['count'] > 0 else 0) for d in days]
        axs1[2].bar(days, dow_rates, color='#bcbd22')
        axs1[2].set_title("Lucid Rate by Day of the Week")
        axs1[2].set_ylabel("Lucid Dream Rate (%)")
        axs1[2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overview_filename = unique_output_path(f'overview_{user_id}')
        await asyncio.to_thread(plt.savefig, overview_filename)
        plt.close()
        await send_generated_file(ctx, overview_filename)

        # Image 2: WBTB Impact
        wbtb_groups = defaultdict(list)
        for r in reports: wbtb_groups[int(r.get('wbtb', 0))].append(r)
        if wbtb_groups:
            wbtb_stats = {}
            for count, group_reports in wbtb_groups.items():
                wbtb_stats[count] = {
                    'avg_dreams': round(sum(int(r.get('dreams', 0)) for r in group_reports) / len(group_reports), 2),
                    'avg_lucid': round(sum(int(r.get('lucid', 0)) for r in group_reports) / len(group_reports), 2),
                    'report_count': len(group_reports)
                }
            sorted_wbtb_counts = sorted(wbtb_stats.keys())
            xticklabels = [f"{c}\n(n={wbtb_stats[c]['report_count']})" for c in sorted_wbtb_counts]
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.bar([x - 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_dreams'] for c in sorted_wbtb_counts], width=0.4, color='tab:blue', label='Avg Dreams')
            ax2_twin = ax2.twinx()
            ax2_twin.bar([x + 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_lucid'] for c in sorted_wbtb_counts], width=0.4, color='tab:orange', label='Avg Lucid')
            ax2.set_xlabel("Number of WBTB Attempts")
            ax2.set_ylabel("Average Dream Recall", color='tab:blue')
            ax2_twin.set_ylabel("Average Lucid Dreams", color='tab:orange')
            ax2.set_xticks(sorted_wbtb_counts)
            ax2.set_xticklabels(xticklabels)
            plt.title(f"WBTB Impact for {target_user.display_name}")
            wbtb_filename = unique_output_path(f'overview_wbtb_{user_id}')
            await asyncio.to_thread(plt.savefig, wbtb_filename)
            plt.close()
            await send_generated_file(ctx, wbtb_filename)

        # Image 3: Journaling Time
        valid_reports = [r for r in reports if parse_date(r.get('date')) is not None]
        sorted_reports = sorted(valid_reports, key=lambda r: parse_date(r['date']))
        reports_with_journal_time = [r for r in sorted_reports if r.get('journal_time') and int(r.get('journal_time', 0)) > 0]
        if len(reports_with_journal_time) >= 2:
            journal_times_same_day = [int(r['journal_time']) for r in reports_with_journal_time]
            qualities_same_day = [normalize_quality(r.get('quality', 0)) for r in reports_with_journal_time]
            dreams_same_day = [int(r.get('dreams', 0)) for r in reports_with_journal_time]
            journal_times_next_day, qualities_next_day, dreams_next_day = [], [], []
            for i in range(len(sorted_reports) - 1):
                current_report, next_report = sorted_reports[i], sorted_reports[i+1]
                if current_report.get('journal_time') and int(current_report.get('journal_time', 0)) > 0:
                    journal_times_next_day.append(int(current_report['journal_time']))
                    qualities_next_day.append(normalize_quality(next_report.get('quality', 0)))
                    dreams_next_day.append(int(next_report.get('dreams', 0)))
            
            fig3, axs3 = plt.subplots(2, 2, figsize=(15, 12))
            fig3.suptitle(f"Journaling Time Analysis for {target_user.display_name}", fontsize=16)
            
            def plot_scatter_with_trend(ax, x_data, y_data, title, xlabel, ylabel):
                if not x_data or len(x_data) < 2: return
                point_counts = Counter(zip(x_data, y_data))
                unique_x, unique_y, colors = [p[0] for p in point_counts.keys()], [p[1] for p in point_counts.keys()], list(point_counts.values())
                sc = ax.scatter(unique_x, unique_y, c=colors, cmap='viridis', alpha=0.9)
                cbar = fig3.colorbar(sc, ax=ax)
                cbar.set_label('Quantity of data points')
                ax.set_title(title)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(np.unique(x_data), p(np.unique(x_data)), "r--", label=f"Trend (Correlation: {np.corrcoef(x_data, y_data)[0,1]:.2f})")
                ax.legend()

            plot_scatter_with_trend(axs3[0, 0], journal_times_same_day, qualities_same_day, 'Journal-time vs. Quality (Same Day)', 'Journaling Minutes', 'Quality (/10)')
            plot_scatter_with_trend(axs3[0, 1], journal_times_same_day, dreams_same_day, 'Journal-time vs. Recall (Same Day)', 'Journaling Minutes', 'Number of Dreams')
            plot_scatter_with_trend(axs3[1, 0], journal_times_next_day, qualities_next_day, 'Journal-time vs. Quality (Next Day)', 'Journaling Minutes (Day Before)', 'Quality (/10)')
            plot_scatter_with_trend(axs3[1, 1], journal_times_next_day, dreams_next_day, 'Journal-time vs. Recall (Next Day)', 'Journaling Minutes (Day Before)', 'Number of Dreams')
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            journaltime_filename = unique_output_path(f'overview_journaltime_{user_id}')
            await asyncio.to_thread(plt.savefig, journaltime_filename)
            plt.close()
            await send_generated_file(ctx, journaltime_filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='heatmap')
async def heatmap(ctx, year: int = None):
    try:
        user_id = str(ctx.author.id)
        
        if year is None:
            year = datetime.now().year
            
        # Select only a data source that actually represents the requested year.
        if year == 2025:
            source = user_reports_25
        elif year == ACTIVE_REPORT_YEAR:
            source = user_reports
        else:
            await ctx.send(f"No data file is configured for {year}.")
            return
        
        if user_id not in source or not source[user_id]:
            await ctx.send(f"No reports found for {year} to generate a heatmap.")
            return

        reports = source[user_id]
        
        # Create a dictionary to store lucid dream counts for each day of the year
        lucid_days = defaultdict(int)
        for report in reports:
            report_date_str = report.get("date")
            if report_date_str:
                for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                    try:
                        report_date = datetime.strptime(report_date_str, fmt).date()
                        if report_date.year == year:
                            lucid_days[report_date] += int(report.get("lucid", 0))
                        break
                    except ValueError:
                        continue

        # Create a matrix to represent the calendar heatmap
        # 7 days a week, 53 weeks a year
        calendar_matrix = np.full((7, 53), np.nan)
        
        # Populate the matrix with lucid dream counts
        for day, count in lucid_days.items():
            # Adjust week number to handle edge cases better
            week_num = day.isocalendar()[1] - 1
            if week_num >= 53: week_num = 52
            day_of_week = day.weekday()
            calendar_matrix[day_of_week, week_num] = count

        # Create the plot
        fig, ax = plt.subplots(figsize=(20, 5))
        cmap = plt.get_cmap('Greens')
        cmap.set_bad(color='lightgray') # Set color for NaN values
        im = ax.imshow(calendar_matrix, cmap=cmap, interpolation='nearest', aspect='auto')

        # Set labels
        ax.set_yticks(np.arange(7))
        ax.set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        ax.set_xticks(np.arange(0, 53, 5))
        ax.set_xticklabels([f'Week {i}' for i in np.arange(1, 54, 5)])

        # Add month labels
        for month in range(1, 13):
            first_day = date(year, month, 1)
            week_num = first_day.isocalendar()[1] - 1
            if week_num >= 53: week_num = 52
            ax.text(week_num, -1.5, first_day.strftime('%b'), ha='center', va='center')

        # Add color bar
        vmax = np.nanmax(calendar_matrix) if not np.all(np.isnan(calendar_matrix)) else 1
        cbar = plt.colorbar(im, ax=ax, ticks=np.arange(vmax + 1))
        cbar.set_label('Lucid Dreams')

        ax.set_title(f'Lucid Dream Heatmap for {year}')
        plt.tight_layout()

        # Save and send the plot
        filename = unique_output_path(f'heatmap_{user_id}_{year}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='day_of_week')
async def day_of_week(ctx):
    try:
        user_id = str(ctx.author.id)
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found to generate a day of the week analysis.")
            return

        reports = user_reports[user_id]
        
        # Data structure to hold stats for each day
        day_stats = {i: {'lucid_total': 0, 'dream_total': 0, 'report_count': 0} for i in range(7)}

        for report in reports:
            report_date_str = report.get("date")
            if report_date_str:
                for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                    try:
                        report_date = datetime.strptime(report_date_str, fmt).date()
                        day_of_week = report_date.weekday() # Monday is 0 and Sunday is 6
                        
                        day_stats[day_of_week]['lucid_total'] += int(report.get("lucid", 0))
                        day_stats[day_of_week]['dream_total'] += int(report.get("dreams", 0))
                        day_stats[day_of_week]['report_count'] += 1
                        break
                    except ValueError:
                        continue

        # Prepare data for plotting
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        lucid_totals = [day_stats[i]['lucid_total'] for i in range(7)]
        avg_dreams = [(day_stats[i]['dream_total'] / day_stats[i]['report_count']) if day_stats[i]['report_count'] > 0 else 0 for i in range(7)]

        # Create the plot
        fig, ax1 = plt.subplots(figsize=(12, 7))

        # Bar chart for lucid dreams
        ax1.bar(days, lucid_totals, color='#2ca02c', label='Total Lucid Dreams')
        ax1.set_ylabel('Total Lucid Dreams', color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')

        # Line chart for average dream recall
        ax2 = ax1.twinx()
        ax2.plot(days, avg_dreams, color='#1f77b4', marker='o', label='Average Dream Recall')
        ax2.set_ylabel('Average Dream Recall', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')

        ax1.set_xlabel('Day of the Week')
        ax1.set_title('Dream Statistics by Day of the Week')
        fig.tight_layout()

        # Save and send the plot
        filename = unique_output_path(f'day_of_week_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='journaltime')
async def journaltime(ctx, user: discord.User = None): ## GEÄNDERT: Optionales user-Argument
    try:
        # --- Zielnutzer bestimmen --- ## GEÄNDERT
        target_user = user or ctx.author # Kurzschreibweise für: target_user = user if user is not None else ctx.author

        user_id = str(target_user.id) ## GEÄNDERT: ID vom Zielnutzer

        # Die Helferfunktionen bleiben unverändert
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try: return datetime.strptime(date_str.strip(), fmt)
                except ValueError: continue
            return None

        def normalize_quality(quality_str):
            try:
                parts = str(quality_str).split('/')
                if len(parts) == 1: return int(parts[0])
                num, den = map(int, parts)
                return (num / den) * 10
            except (ValueError, TypeError, ZeroDivisionError, IndexError): return 0

        if user_id not in user_reports or not user_reports[user_id]:
            # Angepasste Nachricht, falls ein anderer Nutzer keine Daten hat
            if user:
                await ctx.send(f"For {target_user.name} no reports were found.")
            else:
                await ctx.send("No reports were found for you.")
            return

        reports = user_reports[user_id]

        # Der Rest der Datenvorbereitung bleibt identisch...
        valid_reports = [r for r in reports if parse_date(r.get('date')) is not None]
        sorted_reports = sorted(valid_reports, key=lambda r: parse_date(r['date']))
        reports_with_journal_time = [r for r in sorted_reports if r.get('journal_time') and int(r.get('journal_time', 0)) > 0]
        
        if len(reports_with_journal_time) < 2:
            await ctx.send(f"At least 2 reports with journaltime are required, {target_user.name} , to generate an analysis.")
            return

        journal_times_same_day = [int(r['journal_time']) for r in reports_with_journal_time]
        qualities_same_day = [normalize_quality(r.get('quality', 0)) for r in reports_with_journal_time]
        dreams_same_day = [int(r.get('dreams', 0)) for r in reports_with_journal_time]
        journal_times_next_day, qualities_next_day, dreams_next_day = [], [], []
        for i in range(len(sorted_reports) - 1):
            current_report, next_report = sorted_reports[i], sorted_reports[i+1]
            if current_report.get('journal_time') and int(current_report.get('journal_time', 0)) > 0:
                journal_times_next_day.append(int(current_report['journal_time']))
                qualities_next_day.append(normalize_quality(next_report.get('quality', 0)))
                dreams_next_day.append(int(next_report.get('dreams', 0)))

        # --- Plotten ---
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"Analysis of journaling time for {target_user.name}", fontsize=16) ## GEÄNDERT: Name vom Zielnutzer
        
        # Die Hilfsfunktion zum Plotten bleibt exakt gleich. Das ist der Vorteil von gut strukturiertem Code!
        def plot_scatter_with_trend(ax, x_data, y_data, title, xlabel, ylabel):
            if not x_data or len(x_data) < 2:
                ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=10)
                return
            point_counts = Counter(zip(x_data, y_data))
            unique_x, unique_y, colors = [p[0] for p in point_counts.keys()], [p[1] for p in point_counts.keys()], list(point_counts.values())
            sc = ax.scatter(unique_x, unique_y, c=colors, cmap='viridis', alpha=0.9)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Quantity of data points', fontsize=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(np.unique(x_data), p(np.unique(x_data)), "r--", linewidth=2, label=f"Trend (Correlation: {np.corrcoef(x_data, y_data)[0,1]:.2f})")
                ax.legend()
            except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
                pass

        # Die Aufrufe bleiben identisch
        plot_scatter_with_trend(axs[0, 0], journal_times_same_day, qualities_same_day, 'Journal-time vs. quality (same day)', 'Journaling-minutes', 'quality (/10)')
        plot_scatter_with_trend(axs[0, 1], journal_times_same_day, dreams_same_day, 'Journal-time vs. number of dreams (same day)', 'Journaling-minutes', 'number of dreams')
        plot_scatter_with_trend(axs[1, 0], journal_times_next_day, qualities_next_day, 'Journal-time vs. quality (next day)', 'Journaling-minutes from the day before', 'quality (/10)')
        plot_scatter_with_trend(axs[1, 1], journal_times_next_day, dreams_next_day, 'Journal-time vs. number of dreams (next day)', 'Journaling-minutes from the day before', 'number of dreams')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        filename = unique_output_path(f'journaltime_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()

        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)

@bot.command(name='journaltime_all')
async def journaltime_all(ctx):
    try:
        # Helferfunktionen, identisch zum vorherigen Befehl
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None

        def normalize_quality(quality_str):
            try:
                parts = str(quality_str).split('/')
                if len(parts) == 1:
                    return int(parts[0])
                numerator, denominator = map(int, parts)
                return (numerator / denominator) * 10
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                return 0

        # Schritt 1: Sammle Berichte von ALLEN Nutzern
        all_reports = []
        for user_report_list in user_reports.values():
            all_reports.extend(user_report_list)

        if not all_reports:
            await ctx.send("No reports found.")
            return

        # Der Rest der Logik ist fast identisch, verwendet aber "all_reports"
        valid_reports = [r for r in all_reports if parse_date(r.get('date')) is not None]
        sorted_reports = sorted(valid_reports, key=lambda r: parse_date(r['date']))
        
        reports_with_journal_time = [r for r in sorted_reports if r.get('journal_time') and int(r.get('journal_time', 0)) > 0]
        if len(reports_with_journal_time) < 2:
            await ctx.send("users need to have at least 2 reports with journaling time.")
            return

        # --- Datenvorbereitung (unverändert, nutzt jetzt aggregierte Daten)---
        journal_times_same_day = [int(r['journal_time']) for r in reports_with_journal_time]
        qualities_same_day = [normalize_quality(r.get('quality', 0)) for r in reports_with_journal_time]
        dreams_same_day = [int(r.get('dreams', 0)) for r in reports_with_journal_time]

        journal_times_next_day = []
        qualities_next_day = []
        dreams_next_day = []

        for i in range(len(sorted_reports) - 1):
            current_report = sorted_reports[i]
            next_report = sorted_reports[i+1]
            
            if current_report.get('journal_time') and int(current_report.get('journal_time', 0)) > 0:
                journal_times_next_day.append(int(current_report['journal_time']))
                qualities_next_day.append(normalize_quality(next_report.get('quality', 0)))
                dreams_next_day.append(int(next_report.get('dreams', 0)))

        # --- Plotten ---
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        # Der Titel wird für die Gesamtanalyse angepasst
        fig.suptitle("Analysis of journaling time - dream recall and quality", fontsize=16)
        
        # Die Plot-Hilfsfunktion ist exakt die gleiche und kann wiederverwendet werden!
        def plot_scatter_with_trend(ax, x_data, y_data, title, xlabel, ylabel):
            if not x_data or len(x_data) < 2:
                ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=10)
                return
            point_counts = Counter(zip(x_data, y_data))
            unique_x = [point[0] for point in point_counts.keys()]
            unique_y = [point[1] for point in point_counts.keys()]
            colors = list(point_counts.values())
            sc = ax.scatter(unique_x, unique_y, c=colors, cmap='viridis', alpha=0.9)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Quantity of Data points', fontsize=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(np.unique(x_data), p(np.unique(x_data)), "r--", linewidth=2, label=f"Trend (Correlation: {np.corrcoef(x_data, y_data)[0,1]:.2f})")
                ax.legend()
            except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
                pass
        
        # Aufrufe bleiben identisch
        plot_scatter_with_trend(axs[0, 0], journal_times_same_day, qualities_same_day, 'Journal-time vs. quality (same day)', 'Journaling-minutes', 'quality (/10)')
        plot_scatter_with_trend(axs[0, 1], journal_times_same_day, dreams_same_day, 'Journal-time vs. number of dreams (same day)', 'Journaling-minutes', 'number of dreams')
        plot_scatter_with_trend(axs[1, 0], journal_times_next_day, qualities_next_day, 'Journal-time vs. quality (next day)', 'Journaling-minutes from the day before', 'quality (/10)')
        plot_scatter_with_trend(axs[1, 1], journal_times_next_day, dreams_next_day, 'Journal-time vs. number of dreams (next day)', 'Journaling-minutes from the day before', 'number of dreams')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Allgemeiner Dateiname
        filename = unique_output_path('journaltime_all_users')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()

        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)

@bot.command(name='lucid_history')
async def lucid_history(ctx):
    """Shows your 10 most recent lucid dream reports and offers a .txt file with all such reports."""
    try:
        user_id = str(ctx.author.id)
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return

        # Filter reports with at least 1 lucid dream
        lucid_reports = [r for r in user_reports[user_id] if int(r.get('lucid', 0)) > 0]
        if not lucid_reports:
            await ctx.send("No lucid dream reports found.")
            return

        # Sort by date (most recent first)
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except (ValueError, TypeError):
                    continue
            return datetime.min
        lucid_reports_sorted = sorted(lucid_reports, key=lambda r: parse_date(r.get('date', '01.01.1970')), reverse=True)

        # Show the 10 most recent
        to_show = lucid_reports_sorted[:10]
        msg = "**Your 10 Most Recent Lucid Dream Reports:**\n"
        for i, report in enumerate(to_show, 1):
            date_str = report.get('date', 'Unknown date')
            dreams = report.get('dreams', '0')
            quality = report.get('quality', '-')
            technique = report.get('technique', '-')
            lucid = report.get('lucid', '1')
            notes = report.get('notes', '-')
            msg += f"{i}. Date: {date_str} | Dreams: {dreams} | Lucid: {lucid} | Quality: {quality} | Technique: {technique}\n   Notes: {notes[:100]}{'...' if len(notes) > 100 else ''}\n"
        msg += "\nWould you like a .txt file with ALL your lucid dream reports? Reply with 'yes' or 'no' within 3 minutes. (Any other command or no reply = no file.)"
        await send_long_message(ctx, msg)

        def check(m):
            return m.author == ctx.author and m.channel == ctx.channel and not m.content.startswith('!')

        try:
            reply = await bot.wait_for('message', timeout=180, check=check)
            if reply.content.strip().lower() in ['yes', 'y', 'ja', 'sure', 'ok', 'yea', 'yeah', 'yep']:
                # Prepare .txt file
                lines = []
                for report in lucid_reports_sorted:
                    date_str = report.get('date', 'Unknown date')
                    dreams = report.get('dreams', '0')
                    quality = report.get('quality', '-')
                    technique = report.get('technique', '-')
                    lucid = report.get('lucid', '1')
                    notes = report.get('notes', '-')
                    lines.append(f"Date: {date_str}\nDreams: {dreams}\nLucid: {lucid}\nQuality: {quality}\nTechnique: {technique}\nNotes: {notes}\n{'-'*40}\n")
                txt_content = ''.join(lines)
                filename = unique_output_path(f"lucid_reports_{user_id}", suffix=".txt")
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(txt_content)
                await send_generated_file(ctx, filename)
            else:
                await ctx.send("No file will be sent.")
        except asyncio.TimeoutError:
            await ctx.send("No response received in 3 minutes. No file will be sent.")
    except Exception as e:
        await send_internal_error(ctx, e)

@bot.command(name='lucid_gaps')
async def lucid_gaps(ctx, user: discord.User = None):
    """Analyzes the time gaps between lucid dreams with multiple views."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send(f"No reports found for {target_user.display_name}.")
            return

        # Parse date helper
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt).date()
                except ValueError:
                    continue
            return None

        # Get all reports with lucid dreams
        lucid_reports = []
        for report in user_reports[user_id]:
            if int(report.get('lucid', 0)) > 0:
                report_date = parse_date(report.get('date'))  # ← FIXED: was 'date'
                if report_date:
                    lucid_reports.append(report_date)

        if len(lucid_reports) < 2:
            await ctx.send(f"{target_user.display_name} needs at least 2 lucid dreams to analyze gaps.")
            return

        # Sort dates
        lucid_reports.sort()

        # Calculate gaps between lucids
        gaps = []
        gap_end_dates = []
        for i in range(1, len(lucid_reports)):
            gap = (lucid_reports[i] - lucid_reports[i-1]).days
            gaps.append(gap)
            gap_end_dates.append(lucid_reports[i])

        # Calculate current gap (days since last lucid)
        today = datetime.now().date()
        current_gap = (today - lucid_reports[-1]).days
        
        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Lucid Dream Gap Analysis for {target_user.display_name}', fontsize=16, fontweight='bold')

        # Color coding function
        def get_color(gap):
            if gap <= 7:
                return '#2ca02c'  # Green - excellent
            elif gap <= 14:
                return '#ff7f0e'  # Orange - good
            elif gap <= 30:
                return '#d62728'  # Red - needs work
            else:
                return '#8b0000'  # Dark red - dry spell

        # 1. Raw gaps with dotted line
        colors = [get_color(g) for g in gaps]
        axs[0, 0].scatter(gap_end_dates, gaps, c=colors, s=60, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)
        axs[0, 0].plot(gap_end_dates, gaps, 'k:', alpha=0.4, linewidth=1.5, zorder=2)
        
        # Add trend line
        if len(gaps) >= 3:
            z = np.polyfit(range(len(gaps)), gaps, 1)
            p = np.poly1d(z)
            trend_line = p(range(len(gaps)))
            axs[0, 0].plot(gap_end_dates, trend_line, 'b--', alpha=0.5, linewidth=2, 
                          label=f'Trend: {"↓ Improving" if z[0] < 0 else "↑ Declining"}', zorder=1)
        
        # Add current gap if ongoing
        if current_gap > 0:
            axs[0, 0].scatter([today], [current_gap], c='purple', s=150, marker='*', 
                            edgecolors='black', linewidths=1.5, 
                            label=f'Current dry spell: {current_gap} days', zorder=5)
            # Add dotted line from last lucid to today
            axs[0, 0].plot([lucid_reports[-1], today], [gaps[-1] if gaps else 0, current_gap], 
                          'purple', linestyle=':', linewidth=2, alpha=0.5, zorder=1)
        
        axs[0, 0].set_title('Days Between Lucid Dreams (Raw Data)', fontweight='bold')
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel('Days Since Previous Lucid')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(loc='best')
        axs[0, 0].tick_params(axis='x', rotation=45)

        # 2. 3-event moving average
        if len(gaps) >= 3:
            moving_avg_3 = []
            moving_avg_dates = []
            for i in range(2, len(gaps)):
                avg = sum(gaps[i-2:i+1]) / 3
                moving_avg_3.append(avg)
                moving_avg_dates.append(gap_end_dates[i])
            
            axs[0, 1].plot(moving_avg_dates, moving_avg_3, 'b-', linewidth=2.5, marker='o', 
                          markersize=5, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=1.5)
            axs[0, 1].fill_between(moving_avg_dates, moving_avg_3, alpha=0.3)
            axs[0, 1].axhline(y=sum(gaps)/len(gaps), color='red', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Overall avg: {sum(gaps)/len(gaps):.1f} days')
            axs[0, 1].set_title('3-Lucid Moving Average', fontweight='bold')
            axs[0, 1].set_xlabel('')
            axs[0, 1].set_ylabel('Average Gap (days)')
            axs[0, 1].grid(True, alpha=0.3)
            axs[0, 1].legend()
            axs[0, 1].tick_params(axis='x', rotation=45)
        else:
            axs[0, 1].text(0.5, 0.5, 'Need at least 3 gaps\nfor moving average', 
                          ha='center', va='center', fontsize=12)
            axs[0, 1].set_title('3-Lucid Moving Average', fontweight='bold')

        # 3. Monthly average (group gaps by month)
        monthly_data = defaultdict(list)
        for gap_date, gap in zip(gap_end_dates, gaps):  # ← FIXED: renamed to gap_date
            month_key = (gap_date.year, gap_date.month)
            monthly_data[month_key].append(gap)
        
        if len(monthly_data) >= 2:
            month_keys = sorted(monthly_data.keys())
            monthly_avgs = [sum(monthly_data[m]) / len(monthly_data[m]) for m in month_keys]
            month_dates = [date(m[0], m[1], 15) for m in month_keys]
            
            axs[1, 0].plot(month_dates, monthly_avgs, 'g:', linewidth=2.5, marker='s', 
                          markersize=8, markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=1.5)
            axs[1, 0].set_title('Monthly Average Gap', fontweight='bold')
            axs[1, 0].set_xlabel('Month')
            axs[1, 0].set_ylabel('Average Gap (days)')
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].tick_params(axis='x', rotation=45)
            
            # Add annotations for best and worst months
            best_month_idx = monthly_avgs.index(min(monthly_avgs))
            worst_month_idx = monthly_avgs.index(max(monthly_avgs))
            axs[1, 0].annotate(f'Best: {monthly_avgs[best_month_idx]:.1f}d', 
                              xy=(month_dates[best_month_idx], monthly_avgs[best_month_idx]),
                              xytext=(10, 10), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='lightgreen', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
            axs[1, 0].annotate(f'Worst: {monthly_avgs[worst_month_idx]:.1f}d', 
                              xy=(month_dates[worst_month_idx], monthly_avgs[worst_month_idx]),
                              xytext=(10, -20), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='lightcoral', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        else:
            axs[1, 0].text(0.5, 0.5, 'Need lucids in\nat least 2 different months', 
                          ha='center', va='center', fontsize=12)
            axs[1, 0].set_title('Monthly Average Gap', fontweight='bold')

       # 4. Gap Distribution - Individual Days
        gap_counts = Counter(gaps)
        unique_gaps = sorted(gap_counts.keys())
        frequencies = [gap_counts[g] for g in unique_gaps]

        # Create scatter plot with color coding and size variation
        scatter_colors = [get_color(g) for g in unique_gaps]
        sizes = [freq * 100 for freq in frequencies]  # Scale size by frequency

        # Connect the dots with a line
        axs[1, 1].plot(unique_gaps, frequencies, 'k-', alpha=0.4, linewidth=1.5, zorder=1)

        # Plot scatter points on top
        axs[1, 1].scatter(unique_gaps, frequencies, c=scatter_colors, s=sizes, 
                 alpha=0.6, edgecolors='black', linewidths=2, zorder=3)

        # Add frequency labels above points
        for gap, freq in zip(unique_gaps, frequencies):
            axs[1, 1].text(gap, freq + 0.2, str(freq), ha='center', va='bottom', 
                  fontsize=9, fontweight='bold')

        mean_gap = sum(gaps) / len(gaps)
        axs[1, 1].axvline(mean_gap, color='red', linestyle='--', linewidth=2.5, 
                 label=f'Mean: {mean_gap:.1f} days', zorder=10)
        axs[1, 1].axvline(np.median(gaps), color='blue', linestyle='--', linewidth=2.5, 
                 label=f'Median: {np.median(gaps):.1f} days', zorder=10)
        axs[1, 1].set_title('Gap Distribution (Individual Days)', fontweight='bold')
        axs[1, 1].set_xlabel('Days Between Lucids')
        axs[1, 1].set_ylabel('Frequency (Number of Occurrences)')

        # Improve x-axis spacing - only show labels for actual data points with better spacing
        axs[1, 1].set_xticks(unique_gaps)
        axs[1, 1].set_xticklabels(unique_gaps, rotation=45 if len(unique_gaps) > 10 else 0)

        # Add padding to x-axis limits to prevent crowding
        x_padding = max(1, (max(unique_gaps) - min(unique_gaps)) * 0.05)
        axs[1, 1].set_xlim(min(unique_gaps) - x_padding, max(unique_gaps) + x_padding)

        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        axs[1, 1].set_ylim(bottom=0)

        # Detailed statistics message
        gap_trend = "improving 📈" if len(gaps) >= 3 and np.polyfit(range(len(gaps)), gaps, 1)[0] < 0 else "declining 📉"
        consistency_score = 100 - (np.std(gaps) / mean_gap * 100) if mean_gap > 0 else 0
        
        stats_msg = (
            f"**📊 Lucid Dream Gap Statistics for {target_user.display_name}**\n\n"
            f"**Overview:**\n"
            f"• Total days with lucid dreams: **{len(lucid_reports)}**\n"
            f"• Average gap: **{mean_gap:.1f} days**\n"
            f"• Median gap: **{np.median(gaps):.1f} days**\n"
            f"• Shortest gap: **{min(gaps)} days** ✨\n"
            f"• Longest gap: **{max(gaps)} days**\n"
            f"• Current gap: **{current_gap} days** {'🌟' if current_gap < mean_gap else '⚠️'}\n\n"
            f"**Trend Analysis:**\n"
            f"• Overall trend: **{gap_trend}**\n"
            f"• Consistency score: **{consistency_score:.1f}%** {'🎯' if consistency_score > 50 else '📊'}\n\n"
            f"**Timeline:**\n"
            f"• First lucid: **{lucid_reports[0].strftime('%d.%m.%Y')}**\n"
            f"• Most recent lucid: **{lucid_reports[-1].strftime('%d.%m.%Y')}**\n"
            f"• Total tracking period: **{(lucid_reports[-1] - lucid_reports[0]).days} days**\n\n"
            f"_Color coding: 🟢 ≤7 days | 🟠 8-14 days | 🔴 15-30 days | 🟤 >30 days_"
        )
        await send_long_message(ctx, stats_msg)

        # Save and send
        filename = unique_output_path(f'lucid_gaps_{user_id}')
        await asyncio.to_thread(plt.savefig, filename, dpi=150, bbox_inches='tight')
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)

# ============================================================
# HISTORICAL COMMANDS (2025 DATA ACCESS)
# ============================================================

@bot.command(name='list_25', aliases=['list_previous'])
async def list_reports_25(ctx):
    """Lists all reports for the user from 2025 data."""
    user_id = str(ctx.author.id)

    if user_id not in user_reports_25:
        await ctx.send("You have no reports from 2025.")
        return

    stats_by_month = {}

    for report in user_reports_25[user_id]:
        report_date_str = report.get("date", "").strip().replace(" ", "")
        parsed_report_date = parse_report_date(report_date_str)
        if not parsed_report_date:
            await ctx.send(f"Error parsing date: {report_date_str}. Please check your input.")
            return
        report_date = parsed_report_date.date()

        key = report_date.strftime("%B %Y")
        if key not in stats_by_month:
            stats_by_month[key] = {
                "dates": {},
                "lucid_total": 0,
                "dreams_total": 0,
                "month": report_date.month,
                "year": report_date.year
            }
        stats_by_month[key]["dates"][report_date] = stats_by_month[key]["dates"].get(report_date, 0) + 1
        try:
            lucid = int(report.get("lucid", 0))
        except ValueError:
            lucid = 0
        stats_by_month[key]["lucid_total"] += lucid

        try:
            dreams = int(report.get("dreams", 0))
        except ValueError:
            dreams = 0
        stats_by_month[key]["dreams_total"] += dreams

    overall_total_reports = 0
    overall_unique_days = 0
    overall_duplicates = 0
    overall_lucid = 0
    overall_dreams = 0
    for data in stats_by_month.values():
        overall_total_reports += sum(data["dates"].values())
        overall_unique_days += len(data["dates"])
        overall_duplicates += sum(1 for count in data["dates"].values() if count > 1)
        overall_lucid += data["lucid_total"]
        overall_dreams += data["dreams_total"]

    message = "**2025 Yearly Summary**\n"
    message += f"Total Reports: {overall_total_reports}\n"
    message += f"Unique Reported Days: {overall_unique_days}\n"
    message += f"Days with Duplicates: {overall_duplicates}\n"
    message += f"Total Lucid Dreams: {overall_lucid}\n"
    message += f"Total dreams: {overall_dreams}\n\n"

    month_list = []
    for key, data in stats_by_month.items():
        month_list.append((data["year"], data["month"], key, data))
    month_list.sort()

    for (yr, mon, key, data) in month_list:
        month_start = date(yr, mon, 1)
        last_day = calendar.monthrange(yr, mon)[1]
        month_end = date(yr, mon, last_day)
        days_in_period = [month_start + timedelta(days=i) for i in range((month_end - month_start).days + 1)]

        reported_dates = data["dates"]
        missing_days = [d for d in days_in_period if d not in reported_dates]

        if not missing_days:
            summary_line = f"**{key}: reported every day | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        else:
            summary_line = f"**{key}: missed {len(missing_days)} day(s) | Lucid dreams: {data['lucid_total']} | Total dreams: {data['dreams_total']}**"
        message += summary_line + "\n"

        show_details = missing_days or any(count > 1 for count in reported_dates.values())

        if show_details:
            for d in days_in_period:
                formatted = d.strftime("%d.%m.%y")
                if d in reported_dates:
                    status = "reported"
                    if reported_dates[d] > 1:
                        status += " Duplicate!"
                    lucid_for_day = 0
                    for report in user_reports_25[user_id]:
                        rep_str = report.get("date", "").strip().replace(" ", "")
                        parsed_rep_date = parse_report_date(rep_str)
                        if not parsed_rep_date:
                            continue
                        rep_date = parsed_rep_date.date()
                        if rep_date == d:
                            try:
                                lucid_for_day += int(report.get("lucid", 0))
                            except ValueError:
                                continue
                    if lucid_for_day > 0:
                        status += " Lucid!"
                    message += f"  {formatted}: {status}\n"
                else:
                    message += f"  {formatted}: MISSING\n"
        message += "\n"

    await send_long_message(ctx, message)


@bot.command(name='overview_25', aliases=['overview_previous'])
async def overview_25(ctx, user: discord.User = None):
    """Provides a comprehensive overview of the user's 2025 dream journal with charts."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send(f"No 2025 reports found for {target_user.display_name}.")
            return

        reports = user_reports_25[user_id]

        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None

        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        # --- 1. Text Summary ---
        summary_message = f"### 🌟 **2025 Dream Journal Overview for {target_user.display_name}** 🌟\n\n"

        # 1.1 Recent Trends
        valid_report_dates = [parse_date(r.get('date', '')) for r in reports if parse_date(r.get('date', ''))]
        historical_anchor = max(valid_report_dates).date() if valid_report_dates else date(2025, 12, 31)
        this_week, last_week = rolling_report_windows(reports, historical_anchor)
        if this_week and last_week:
            
            def calculate_simple_summary(data):
                summary = {"dreams": 0, "quality": 0, "lucid": 0}
                for r in data:
                    summary["dreams"] += int(r.get("dreams", 0))
                    summary["quality"] += normalize_quality(r.get("quality", "0"))
                    summary["lucid"] += int(r.get("lucid", 0))
                count = len(data)
                if count > 0:
                    summary["dreams"] /= count
                    summary["quality"] /= count
                return summary

            this_summary = calculate_simple_summary(this_week)
            last_summary = calculate_simple_summary(last_week)
            
            summary_message += "**📈 Recent Trends (Last 7 vs. Previous 7 Days):**\n"
            summary_message += f"*   **Dream Recall:** `{this_summary['dreams']:.1f}` vs `{last_summary['dreams']:.1f}` (`{this_summary['dreams'] - last_summary['dreams']:.1f}`)\n"
            summary_message += f"*   **Dream Quality:** `{this_summary['quality']:.1f}` vs `{last_summary['quality']:.1f}` (`{this_summary['quality'] - last_summary['quality']:.1f}`)\n"
            summary_message += f"*   **Lucid Dreams:** `{this_summary['lucid']}` vs `{last_summary['lucid']}` (`{this_summary['lucid'] - last_summary['lucid']:+}`)\n\n"

        # 1.2 Top 5 Techniques
        all_techniques = defaultdict(list)
        for report in reports:
            tech_entry = report.get('technique', '').lower().strip()
            if tech_entry:
                all_techniques[tech_entry].append(report)
        
        ranked_techniques = []
        for tech, tech_reports in all_techniques.items():
            lucid_count = sum(1 for r in tech_reports if int(r.get('lucid', 0)) > 0)
            total_count = len(tech_reports)
            rate = (lucid_count / total_count) * 100 if total_count > 0 else 0
            ranked_techniques.append({'name': tech, 'rate': rate, 'uses': total_count})
        
        ranked_techniques.sort(key=lambda x: x['rate'], reverse=True)
        
        summary_message += "**🚀 Top 5 Most Effective Techniques (2025):**\n"
        for i, tech in enumerate(ranked_techniques[:5], 1):
            summary_message += f"*   {i}. **{tech['name'].upper()}**: `{tech['rate']:.1f}%` Lucid Rate ({tech['uses']} uses)\n"
        summary_message += "\n"

        # 1.3 Key Lucidity Factors
        total_dreams = sum(int(r.get('dreams', 0)) for r in reports)
        total_quality = sum(normalize_quality(r.get('quality', '0')) for r in reports)
        avg_dreams = total_dreams / len(reports) if reports else 0
        avg_quality = total_quality / len(reports) if reports else 0
        
        categories = {
            "High Recall / High Quality": [], "High Recall / Low Quality": [],
            "Low Recall / High Quality": [], "Low Recall / Low Quality": []
        }
        for r in reports:
            recall = int(r.get('dreams', 0))
            quality = normalize_quality(r.get('quality', '0'))
            if recall >= avg_dreams and quality >= avg_quality: categories["High Recall / High Quality"].append(r)
            elif recall >= avg_dreams and quality < avg_quality: categories["High Recall / Low Quality"].append(r)
            elif recall < avg_dreams and quality >= avg_quality: categories["Low Recall / High Quality"].append(r)
            else: categories["Low Recall / Low Quality"].append(r)

        summary_message += "**📊 Key Lucidity Factors:**\n"
        for category, report_list in categories.items():
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            rate = (lucid_count / len(report_list)) * 100 if report_list else 0
            summary_message += f"*   **{category}**: `{rate:.1f}%` Lucid Rate ({len(report_list)} reports)\n"
        summary_message += "\n"

        # 1.4 Year Summary
        total_lucid = sum(int(r.get('lucid', 0)) for r in reports)
        summary_message += "**📊 2025 Year Summary:**\n"
        summary_message += f"*   **Total Reports:** `{len(reports)}`\n"
        summary_message += f"*   **Total Dreams:** `{total_dreams}`\n"
        summary_message += f"*   **Total Lucid Dreams:** `{total_lucid}`\n"
        summary_message += f"*   **Average Dreams/Report:** `{avg_dreams:.1f}`\n"
        summary_message += f"*   **Average Quality:** `{avg_quality:.1f}/10`\n"

        await send_long_message(ctx, summary_message)

        # --- 2. Graph Generation ---

        # Image 1: Main Overview
        fig1, axs1 = plt.subplots(3, 1, figsize=(12, 18))
        fig1.suptitle(f"2025 Main Overview for {target_user.display_name}", fontsize=16)
        
        # Graph 1.1: Recall Trend
        if this_week and last_week:
            current_start = historical_anchor - timedelta(days=6)
            previous_start = historical_anchor - timedelta(days=13)
            current_dates, current_dreams = daily_metric_series(this_week, current_start, 'dreams', int)
            _, previous_dreams = daily_metric_series(last_week, previous_start, 'dreams', int)
            _, current_quality = daily_metric_series(this_week, current_start, 'quality', normalize_quality)
            _, previous_quality = daily_metric_series(last_week, previous_start, 'quality', normalize_quality)
            this_week_data = {'dreams': current_dreams, 'quality': current_quality}
            last_week_data = {'dreams': previous_dreams, 'quality': previous_quality}
            date_labels = [d.strftime('%a') for d in current_dates]
            axs1[0].plot(date_labels, this_week_data['dreams'], marker='o', label='This Week')
            axs1[0].plot(date_labels, last_week_data['dreams'], marker='o', linestyle='--', label='Last Week')
            axs1[0].set_title("Dream Recall Trend")
            axs1[0].set_ylabel("Average Dream Recall")
            axs1[0].legend()

            axs1[1].plot(date_labels, this_week_data['quality'], marker='s', color='r', label='This Week')
            axs1[1].plot(date_labels, last_week_data['quality'], marker='s', linestyle='--', color='m', label='Last Week')
            axs1[1].set_title("Dream Quality Trend")
            axs1[1].set_ylabel("Average Dream Quality (/10)")
            axs1[1].legend()

        # Graph 1.3: Day of Week
        dow_stats = defaultdict(lambda: {'lucid': 0, 'count': 0})
        for r in reports:
            dt = parse_date(r.get('date'))
            if dt:
                dow = dt.strftime('%A')
                dow_stats[dow]['lucid'] += int(r.get('lucid', 0)) > 0
                dow_stats[dow]['count'] += 1
        days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        dow_rates = [(dow_stats[d]['lucid'] / dow_stats[d]['count'] * 100 if dow_stats[d]['count'] > 0 else 0) for d in days]
        axs1[2].bar(days, dow_rates, color='#bcbd22')
        axs1[2].set_title("Lucid Rate by Day of the Week")
        axs1[2].set_ylabel("Lucid Dream Rate (%)")
        axs1[2].tick_params(axis='x', rotation=45)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        overview_filename = unique_output_path(f'overview_25_{user_id}')
        await asyncio.to_thread(plt.savefig, overview_filename)
        plt.close()
        await send_generated_file(ctx, overview_filename)

        # Image 2: WBTB Impact
        wbtb_groups = defaultdict(list)
        for r in reports: wbtb_groups[int(r.get('wbtb', 0))].append(r)
        if wbtb_groups:
            wbtb_stats = {}
            for count, group_reports in wbtb_groups.items():
                wbtb_stats[count] = {
                    'avg_dreams': round(sum(int(r.get('dreams', 0)) for r in group_reports) / len(group_reports), 2),
                    'avg_lucid': round(sum(int(r.get('lucid', 0)) for r in group_reports) / len(group_reports), 2),
                    'report_count': len(group_reports)
                }
            sorted_wbtb_counts = sorted(wbtb_stats.keys())
            xticklabels = [f"{c}\n(n={wbtb_stats[c]['report_count']})" for c in sorted_wbtb_counts]
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            ax2.bar([x - 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_dreams'] for c in sorted_wbtb_counts], width=0.4, color='tab:blue', label='Avg Dreams')
            ax2_twin = ax2.twinx()
            ax2_twin.bar([x + 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_lucid'] for c in sorted_wbtb_counts], width=0.4, color='tab:orange', label='Avg Lucid')
            ax2.set_xlabel("Number of WBTB Attempts")
            ax2.set_ylabel("Average Dream Recall", color='tab:blue')
            ax2_twin.set_ylabel("Average Lucid Dreams", color='tab:orange')
            ax2.set_xticks(sorted_wbtb_counts)
            ax2.set_xticklabels(xticklabels)
            plt.title(f"2025 WBTB Impact for {target_user.display_name}")
            wbtb_filename = unique_output_path(f'overview_wbtb_25_{user_id}')
            await asyncio.to_thread(plt.savefig, wbtb_filename)
            plt.close()
            await send_generated_file(ctx, wbtb_filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='journaltime_25', aliases=['journaltime_previous'])
async def journaltime_25(ctx, user: discord.User = None):
    try:
        target_user = user or ctx.author
        user_id = str(target_user.id)

        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try: return datetime.strptime(date_str.strip(), fmt)
                except ValueError: continue
            return None

        def normalize_quality(quality_str):
            try:
                parts = str(quality_str).split('/')
                if len(parts) == 1: return int(parts[0])
                num, den = map(int, parts)
                return (num / den) * 10
            except (ValueError, TypeError, ZeroDivisionError, IndexError): return 0

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send(f"No 2025 reports found for {target_user.name}.")
            return

        reports = user_reports_25[user_id]
        valid_reports = [r for r in reports if parse_date(r.get('date')) is not None]
        sorted_reports = sorted(valid_reports, key=lambda r: parse_date(r['date']))
        reports_with_journal_time = [r for r in sorted_reports if r.get('journal_time') and int(r.get('journal_time', 0)) > 0]
        
        if len(reports_with_journal_time) < 2:
            await ctx.send(f"At least 2 reports with journaltime from 2025 are required for {target_user.name}.")
            return

        journal_times_same_day = [int(r['journal_time']) for r in reports_with_journal_time]
        qualities_same_day = [normalize_quality(r.get('quality', 0)) for r in reports_with_journal_time]
        dreams_same_day = [int(r.get('dreams', 0)) for r in reports_with_journal_time]
        journal_times_next_day, qualities_next_day, dreams_next_day = [], [], []
        for i in range(len(sorted_reports) - 1):
            current_report, next_report = sorted_reports[i], sorted_reports[i+1]
            if current_report.get('journal_time') and int(current_report.get('journal_time', 0)) > 0:
                journal_times_next_day.append(int(current_report['journal_time']))
                qualities_next_day.append(normalize_quality(next_report.get('quality', 0)))
                dreams_next_day.append(int(next_report.get('dreams', 0)))

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f"2025 Analysis of journaling time for {target_user.name}", fontsize=16)
        
        def plot_scatter_with_trend(ax, x_data, y_data, title, xlabel, ylabel):
            if not x_data or len(x_data) < 2:
                ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center', fontsize=12)
                ax.set_title(title, fontsize=10)
                return
            point_counts = Counter(zip(x_data, y_data))
            unique_x, unique_y, colors = [p[0] for p in point_counts.keys()], [p[1] for p in point_counts.keys()], list(point_counts.values())
            sc = ax.scatter(unique_x, unique_y, c=colors, cmap='viridis', alpha=0.9)
            cbar = fig.colorbar(sc, ax=ax)
            cbar.set_label('Quantity of data points', fontsize=10)
            ax.set_title(title, fontsize=12, pad=10)
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            try:
                z = np.polyfit(x_data, y_data, 1)
                p = np.poly1d(z)
                ax.plot(np.unique(x_data), p(np.unique(x_data)), "r--", linewidth=2, label=f"Trend (Correlation: {np.corrcoef(x_data, y_data)[0,1]:.2f})")
                ax.legend()
            except (ValueError, TypeError, np.linalg.LinAlgError, FloatingPointError):
                pass

        plot_scatter_with_trend(axs[0, 0], journal_times_same_day, qualities_same_day, 'Journal-time vs. quality (same day)', 'Journaling-minutes', 'quality (/10)')
        plot_scatter_with_trend(axs[0, 1], journal_times_same_day, dreams_same_day, 'Journal-time vs. recall (same day)', 'Journaling-minutes', 'number of dreams')
        plot_scatter_with_trend(axs[1, 0], journal_times_next_day, qualities_next_day, 'Journal-time vs. quality (next day)', 'Minutes from day before', 'quality (/10)')
        plot_scatter_with_trend(axs[1, 1], journal_times_next_day, dreams_next_day, 'Journal-time vs. recall (next day)', 'Minutes from day before', 'number of dreams')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        filename = unique_output_path(f'journaltime_25_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='day_of_week_25', aliases=['day_of_week_previous'])
async def day_of_week_25(ctx):
    try:
        user_id = str(ctx.author.id)
        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send("No 2025 reports found to generate a day of the week analysis.")
            return

        reports = user_reports_25[user_id]
        day_stats = {i: {'lucid_total': 0, 'dream_total': 0, 'report_count': 0} for i in range(7)}

        for report in reports:
            report_date_str = report.get("date")
            if report_date_str:
                for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                    try:
                        report_date = datetime.strptime(report_date_str, fmt).date()
                        day_of_week = report_date.weekday()
                        day_stats[day_of_week]['lucid_total'] += int(report.get("lucid", 0))
                        day_stats[day_of_week]['dream_total'] += int(report.get("dreams", 0))
                        day_stats[day_of_week]['report_count'] += 1
                        break
                    except ValueError:
                        continue

        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        lucid_totals = [day_stats[i]['lucid_total'] for i in range(7)]
        avg_dreams = [(day_stats[i]['dream_total'] / day_stats[i]['report_count']) if day_stats[i]['report_count'] > 0 else 0 for i in range(7)]

        fig, ax1 = plt.subplots(figsize=(12, 7))
        ax1.bar(days, lucid_totals, color='#2ca02c', label='Total Lucid Dreams')
        ax1.set_ylabel('Total Lucid Dreams', color='#2ca02c')
        ax1.tick_params(axis='y', labelcolor='#2ca02c')

        ax2 = ax1.twinx()
        ax2.plot(days, avg_dreams, color='#1f77b4', marker='o', label='Average Dream Recall')
        ax2.set_ylabel('Average Dream Recall', color='#1f77b4')
        ax2.tick_params(axis='y', labelcolor='#1f77b4')

        plt.title('2025 Dream Statistics by Day of the Week')
        fig.tight_layout()

        filename = unique_output_path(f'day_of_week_25_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='wbtb_impact_25', aliases=['wbtb_impact_previous'])
async def wbtb_impact_25(ctx, user: discord.User = None):
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send(f"No 2025 reports found for {target_user.display_name}.")
            return

        reports = user_reports_25[user_id]
        wbtb_groups = defaultdict(list)
        for r in reports:
            wbtb_count = int(r.get('wbtb', 0))
            wbtb_groups[wbtb_count].append(r)

        if not wbtb_groups:
            await ctx.send("No 2025 WBTB data found in the reports.")
            return

        wbtb_stats = {}
        for count, group_reports in wbtb_groups.items():
            total_dreams = sum(int(r.get('dreams', 0)) for r in group_reports)
            total_lucid = sum(int(r.get('lucid', 0)) for r in group_reports)
            num_reports = len(group_reports)
            wbtb_stats[count] = {
                'avg_dreams': round(total_dreams / num_reports, 2) if num_reports > 0 else 0,
                'avg_lucid': round(total_lucid / num_reports, 2) if num_reports > 0 else 0,
                'report_count': num_reports
            }

        sorted_wbtb_counts = sorted(wbtb_stats.keys())
        xticklabels = [f"{c}\n(n={wbtb_stats[c]['report_count']})" for c in sorted_wbtb_counts]

        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel('Number of WBTB Attempts (n=number of reports)')
        ax1.set_ylabel('Average Dream Recall', color='tab:blue')
        ax1.bar([x - 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_dreams'] for c in sorted_wbtb_counts], width=0.4, color='tab:blue', label='Avg Dreams')
        ax1.tick_params(axis='y', labelcolor='tab:blue')

        ax2 = ax1.twinx()
        ax2.set_ylabel('Average Lucid Dreams', color='tab:orange')
        ax2.bar([x + 0.2 for x in sorted_wbtb_counts], [wbtb_stats[c]['avg_lucid'] for c in sorted_wbtb_counts], width=0.4, color='tab:orange', label='Avg Lucid')
        ax2.tick_params(axis='y', labelcolor='tab:orange')

        plt.title(f'2025 WBTB Impact for {target_user.display_name}')
        ax1.set_xticks(sorted_wbtb_counts)
        ax1.set_xticklabels(xticklabels)
        
        filename = unique_output_path(f'wbtb_impact_25_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='dreams_25', aliases=['dreams_previous'])
async def dreams_25(ctx):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send("No 2025 reports found for you.")
            return

        reports = user_reports_25[user_id]

        def calculate_total_dreams(data):
            total = 0
            for report in data:
                total += int(report.get("dreams", 0))
            return total

        total = calculate_total_dreams(reports)

        response = (
            f"Total dreams (2025): {total}\n"
        )

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='correlate_25', aliases=['correlate_previous'])
async def correlate_25(ctx, *, keyword: str):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send("No 2025 reports found for you.")
            return

        reports = user_reports_25[user_id]
        
        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        with_keyword = []
        without_keyword = []
        for report in reports:
            if keyword.lower() in report.get('notes', '').lower():
                with_keyword.append(report)
            else:
                without_keyword.append(report)

        if not with_keyword:
            await ctx.send(f"No 2025 reports found containing the keyword '{keyword}'.")
            return

        def calculate_stats(report_list):
            if not report_list:
                return {"avg_dreams": 0, "avg_quality": 0, "lucid_rate": 0, "count": 0}
            
            total_dreams = sum(int(r.get('dreams', 0)) for r in report_list)
            total_quality = sum(normalize_quality(r.get('quality', '0')) for r in report_list)
            total_lucid = sum(int(r.get('lucid', 0)) > 0 for r in report_list)
            count = len(report_list)
            
            return {
                "avg_dreams": round(total_dreams / count, 2) if count > 0 else 0,
                "avg_quality": round(total_quality / count, 2) if count > 0 else 0,
                "lucid_rate": round((total_lucid / count) * 100, 1) if count > 0 else 0,
                "count": count
            }

        stats_with = calculate_stats(with_keyword)
        stats_without = calculate_stats(without_keyword)

        response = (
            f"**2025 Correlation Analysis for '{keyword}'**\n\n"
            f"**When notes INCLUDE '{keyword}'** ({stats_with['count']} reports):\n"
            f"• Avg. Dream Recall: **{stats_with['avg_dreams']}**\n"
            f"• Avg. Quality: **{stats_with['avg_quality']:.1f}/10**\n"
            f"• Lucid Dream Rate: **{stats_with['lucid_rate']}%**\n\n"
            f"**When notes DO NOT INCLUDE '{keyword}'** ({stats_without['count']} reports):\n"
            f"• Avg. Dream Recall: **{stats_without['avg_dreams']}**\n"
            f"• Avg. Quality: **{stats_without['avg_quality']:.1f}/10**\n"
            f"• Lucid Dream Rate: **{stats_without['lucid_rate']}%**"
        )

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='effectiveness_25', aliases=['effectiveness_previous'])
async def effectiveness_25(ctx, *, technique: str = None):
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send("No 2025 reports found for you.")
            return

        reports = user_reports_25[user_id]
        failure_keywords = ["fell asleep", "couldn't focus", "gave up", "skipped", "forgot", "did nothing", "no tech"]

        def calculate_lucid_rate(report_list):
            if not report_list:
                return 0, 0
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            total_count = len(report_list)
            rate = round((lucid_count / total_count) * 100, 1) if total_count > 0 else 0
            return rate, total_count

        if not technique:
            all_techniques = defaultdict(list)
            for report in reports:
                tech_entry = report.get('technique', '').lower().strip()
                if tech_entry:
                    all_techniques[tech_entry].append(report)
            
            if not all_techniques:
                await ctx.send("No techniques found in your 2025 reports.")
                return

            ranked_techniques = []
            for tech, tech_reports in all_techniques.items():
                successful_execution = [r for r in tech_reports if not any(keyword in r.get('notes', '').lower() for keyword in failure_keywords)]
                success_rate, success_count = calculate_lucid_rate(successful_execution)
                total_count = len(tech_reports)
                consistency = round((success_count / total_count) * 100, 1) if total_count > 0 else 0
                score = (success_rate * 0.7) + (consistency * 0.3)

                ranked_techniques.append({
                    'name': tech, 'rate': success_rate, 'consistency': consistency, 'uses': total_count, 'score': score
                })
            
            ranked_techniques.sort(key=lambda x: x['score'], reverse=True)

            response = "**Your 2025 Personal Technique Effectiveness Rankings**\n\n"
            for i, tech_data in enumerate(ranked_techniques, 1):
                response += f"{i}. **{tech_data['name'].upper()}** - **{tech_data['rate']}%** Success Rate ({tech_data['consistency']}% consistency over {tech_data['uses']} uses)\n"
            
            await send_long_message(ctx, response)
            return

        technique = technique.lower().strip()
        technique_reports = [r for r in reports if technique == r.get('technique', '').lower().strip()]

        if not technique_reports:
            await ctx.send(f"No 2025 reports found where you used the '{technique.upper()}' technique.")
            return

        successful_execution = []
        failed_execution = []

        for report in technique_reports:
            notes = report.get('notes', '').lower()
            if any(keyword in notes for keyword in failure_keywords):
                failed_execution.append(report)
            else:
                successful_execution.append(report)

        success_rate, success_count = calculate_lucid_rate(successful_execution)
        failure_rate, failure_count = calculate_lucid_rate(failed_execution)

        response = f"**2025 Effectiveness Analysis for {technique.upper()}**\n\n"

        if success_count > 0:
            response += f"**Proper Execution** ({success_count} reports):\n"
            response += f"When you completed the technique properly, your success rate was **{success_rate}%**.\n\n"
        
        if failure_count > 0:
            response += f"**Incomplete/Failed Execution** ({failure_count} reports):\n"
            response += f"When you were distracted or didn't complete the technique, your success rate was **{failure_rate}%**.\n"

        await send_long_message(ctx, response)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='personal_recap_25', aliases=['personal_recap_previous'])
async def personal_recap_25(ctx):
    """Shows your personal recap for 2025."""
    try:
        user_id = str(ctx.author.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send("No 2025 reports found for you.")
            return

        reports = user_reports_25[user_id]
        total_reports = len(reports)

        def calculate_summary(data):
            summary = {"dreams": 0, "quality": 0, "wbtb": 0, "lucid": 0, "techniques": {}}
            count = len(data)

            for report in data:
                summary["dreams"] += int(report.get("dreams", 0))
                summary["quality"] += normalized_quality(report.get("quality", 0)) or 0
                summary["wbtb"] += int(report.get("wbtb", 0))
                summary["lucid"] += int(report.get("lucid", 0))

                technique = report.get("technique", "").lower()
                if technique:
                    summary["techniques"][technique] = summary["techniques"].get(technique, 0) + 1

            summary["quality"] = round(summary["quality"] / count, 2) if count else 0
            summary["dreams"] = round(summary["dreams"] / count, 2) if count else 0
            summary["wbtb"] = round(summary["wbtb"] / count, 2) if count else 0
            summary["lucid"] = round(summary["lucid"] / count, 2) if count else 0
            summary["total_lucid"] = sum(int(report.get("lucid", 0)) for report in data)
            summary["total_dreams"] = sum(int(report.get("dreams", 0)) for report in data)
            summary["total_wbtb"] = sum(int(report.get("wbtb", 0)) for report in data)

            most_used_technique = max(summary["techniques"], key=summary["techniques"].get, default="None")
            most_used_technique_count = summary["techniques"].get(most_used_technique, 0)

            summary["most_used_technique"] = (most_used_technique, most_used_technique_count)
            return summary

        yearly_summary = calculate_summary(reports)

        recap_message = (
            f"🌟 Your Recap 2025 🌟\n\n"
            f"Total Reports Submitted: {total_reports}\n"
            f"- Average Dreams per Report: {yearly_summary['dreams']}\n"
            f"- Average Quality per Report: {yearly_summary['quality']}/10\n"
            f"- Average WBTB Attempts: {yearly_summary['wbtb']}\n"
            f"- Average Lucid Dreams per Report: {yearly_summary['lucid']}\n"
            f"- Total Lucid Dreams: {yearly_summary['total_lucid']}\n"
            f"- Total Dreams: {yearly_summary['total_dreams']}\n"
            f"- Total WBTB Attempts: {yearly_summary['total_wbtb']}\n"
            f"- Most Used Technique: {yearly_summary['most_used_technique'][0]} "
            f"({yearly_summary['most_used_technique'][1]} times)"
        )

        await send_long_message(ctx, recap_message)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='final_group_25', aliases=['final_group_previous'])
async def final_group_25(ctx):
    """Shows the group recap for 2025."""
    try:
        if not user_reports_25:
            await ctx.send("No 2025 reports found for the group.")
            return

        all_reports = []
        for user_id, reports in user_reports_25.items():
            all_reports.extend(reports)

        total_reports = len(all_reports)

        def calculate_summary(data):
            summary = {"dreams": 0, "quality": 0, "wbtb": 0, "lucid": 0, "techniques": {}}
            count = len(data)

            for report in data:
                summary["dreams"] += int(report.get("dreams", 0))
                summary["quality"] += normalized_quality(report.get("quality", 0)) or 0
                summary["wbtb"] += int(report.get("wbtb", 0))
                summary["lucid"] += int(report.get("lucid", 0))

                technique = report.get("technique", "").lower()
                if technique:
                    summary["techniques"][technique] = summary["techniques"].get(technique, 0) + 1

            summary["quality"] = round(summary["quality"] / count, 2) if count else 0
            summary["dreams"] = round(summary["dreams"] / count, 2) if count else 0
            summary["wbtb"] = round(summary["wbtb"] / count, 2) if count else 0
            summary["lucid"] = round(summary["lucid"] / count, 2) if count else 0
            summary["total_lucid"] = sum(int(report.get("lucid", 0)) for report in data)
            summary["total_dreams"] = sum(int(report.get("dreams", 0)) for report in data)
            summary["total_wbtb"] = sum(int(report.get("wbtb", 0)) for report in data)

            most_used_technique = max(summary["techniques"], key=summary["techniques"].get, default="None")
            most_used_technique_count = summary["techniques"].get(most_used_technique, 0)

            summary["most_used_technique"] = (most_used_technique, most_used_technique_count)
            return summary

        group_summary = calculate_summary(all_reports)

        group_recap_message = (
            f"🌍 Group Recap 2025 🌍\n\n"
            f"Total Reports Submitted: {total_reports}\n"
            f"- Average Dreams per Report: {group_summary['dreams']}\n"
            f"- Average Quality per Report: {group_summary['quality']}/10\n"
            f"- Average WBTB Attempts: {group_summary['wbtb']}\n"
            f"- Average Lucid Dreams per Report: {group_summary['lucid']}\n"
            f"- Total Lucid Dreams: {group_summary['total_lucid']}\n"
            f"- Total Dreams: {group_summary['total_dreams']}\n"
            f"- Total WBTB Attempts: {group_summary['total_wbtb']}\n"
            f"- Most Used Technique: {group_summary['most_used_technique'][0]} "
            f"({group_summary['most_used_technique'][1]} times)"
        )

        await send_long_message(ctx, group_recap_message)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='lucid_gaps_25', aliases=['lucid_gaps_previous'])
async def lucid_gaps_25(ctx, user: discord.User = None):
    """Analyzes the time gaps between lucid dreams for 2025 data with multiple views."""
    from datetime import date as date_class
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send(f"No 2025 reports found for {target_user.display_name}.")
            return

        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt).date()
                except ValueError:
                    continue
            return None

        lucid_reports = []
        for report in user_reports_25[user_id]:
            if int(report.get('lucid', 0)) > 0:
                report_date = parse_date(report.get('date'))
                if report_date:
                    lucid_reports.append(report_date)

        if len(lucid_reports) < 2:
            await ctx.send(f"{target_user.display_name} needs at least 2 lucid dreams in 2025 to analyze gaps.")
            return

        lucid_reports.sort()

        gaps = []
        gap_end_dates = []
        for i in range(1, len(lucid_reports)):
            gap = (lucid_reports[i] - lucid_reports[i-1]).days
            gaps.append(gap)
            gap_end_dates.append(lucid_reports[i])

        mean_gap = sum(gaps) / len(gaps)

        # Create figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'2025 Lucid Dream Gap Analysis for {target_user.display_name}', fontsize=16, fontweight='bold')

        # Color coding function
        def get_color(gap):
            if gap <= 7:
                return '#2ca02c'  # Green - excellent
            elif gap <= 14:
                return '#ff7f0e'  # Orange - good
            elif gap <= 30:
                return '#d62728'  # Red - needs work
            else:
                return '#8b0000'  # Dark red - dry spell

        # 1. Raw gaps with dotted line
        colors = [get_color(g) for g in gaps]
        axs[0, 0].scatter(gap_end_dates, gaps, c=colors, s=60, alpha=0.7, edgecolors='black', linewidths=0.5, zorder=3)
        axs[0, 0].plot(gap_end_dates, gaps, 'k:', alpha=0.4, linewidth=1.5, zorder=2)
        
        # Add trend line
        if len(gaps) >= 3:
            z = np.polyfit(range(len(gaps)), gaps, 1)
            p = np.poly1d(z)
            trend_line = p(range(len(gaps)))
            axs[0, 0].plot(gap_end_dates, trend_line, 'b--', alpha=0.5, linewidth=2, 
                          label=f'Trend: {"↓ Improving" if z[0] < 0 else "↑ Declining"}', zorder=1)
        
        axs[0, 0].set_title('Days Between Lucid Dreams (Raw Data)', fontweight='bold')
        axs[0, 0].set_xlabel('')
        axs[0, 0].set_ylabel('Days Since Previous Lucid')
        axs[0, 0].grid(True, alpha=0.3)
        axs[0, 0].legend(loc='best')
        axs[0, 0].tick_params(axis='x', rotation=45)

        # 2. 3-event moving average
        if len(gaps) >= 3:
            moving_avg_3 = []
            moving_avg_dates = []
            for i in range(2, len(gaps)):
                avg = sum(gaps[i-2:i+1]) / 3
                moving_avg_3.append(avg)
                moving_avg_dates.append(gap_end_dates[i])
            
            axs[0, 1].plot(moving_avg_dates, moving_avg_3, 'b-', linewidth=2.5, marker='o', 
                          markersize=5, markerfacecolor='lightblue', markeredgecolor='blue', markeredgewidth=1.5)
            axs[0, 1].fill_between(moving_avg_dates, moving_avg_3, alpha=0.3)
            axs[0, 1].axhline(y=sum(gaps)/len(gaps), color='red', linestyle='--', 
                             linewidth=1.5, alpha=0.7, label=f'Overall avg: {sum(gaps)/len(gaps):.1f} days')
            axs[0, 1].set_title('3-Lucid Moving Average', fontweight='bold')
            axs[0, 1].set_xlabel('')
            axs[0, 1].set_ylabel('Average Gap (days)')
            axs[0, 1].grid(True, alpha=0.3)
            axs[0, 1].legend()
            axs[0, 1].tick_params(axis='x', rotation=45)
        else:
            axs[0, 1].text(0.5, 0.5, 'Need at least 3 gaps\nfor moving average', 
                          ha='center', va='center', fontsize=12)
            axs[0, 1].set_title('3-Lucid Moving Average', fontweight='bold')

        # 3. Monthly average
        monthly_data = defaultdict(list)
        for gap_date, gap in zip(gap_end_dates, gaps):
            month_key = (gap_date.year, gap_date.month)
            monthly_data[month_key].append(gap)
        
        if len(monthly_data) >= 2:
            month_keys = sorted(monthly_data.keys())
            monthly_avgs = [sum(monthly_data[m]) / len(monthly_data[m]) for m in month_keys]
            month_dates = [date_class(m[0], m[1], 15) for m in month_keys]
            
            axs[1, 0].plot(month_dates, monthly_avgs, 'g:', linewidth=2.5, marker='s', 
                          markersize=8, markerfacecolor='lightgreen', markeredgecolor='darkgreen', markeredgewidth=1.5)
            axs[1, 0].set_title('Monthly Average Gap', fontweight='bold')
            axs[1, 0].set_xlabel('Month')
            axs[1, 0].set_ylabel('Average Gap (days)')
            axs[1, 0].grid(True, alpha=0.3)
            axs[1, 0].tick_params(axis='x', rotation=45)
        else:
            axs[1, 0].text(0.5, 0.5, 'Need lucids in\nat least 2 different months', 
                          ha='center', va='center', fontsize=12)
            axs[1, 0].set_title('Monthly Average Gap', fontweight='bold')

        # 4. Gap Distribution
        gap_counts = Counter(gaps)
        unique_gaps = sorted(gap_counts.keys())
        frequencies = [gap_counts[g] for g in unique_gaps]
        scatter_colors = [get_color(g) for g in unique_gaps]
        sizes = [freq * 100 for freq in frequencies]

        axs[1, 1].plot(unique_gaps, frequencies, 'k-', alpha=0.4, linewidth=1.5, zorder=1)
        axs[1, 1].scatter(unique_gaps, frequencies, c=scatter_colors, s=sizes, 
                 alpha=0.6, edgecolors='black', linewidths=2, zorder=3)

        for gap, freq in zip(unique_gaps, frequencies):
            axs[1, 1].text(gap, freq + 0.2, str(freq), ha='center', va='bottom', 
                  fontsize=9, fontweight='bold')

        axs[1, 1].axvline(mean_gap, color='red', linestyle='--', linewidth=2.5, 
                 label=f'Mean: {mean_gap:.1f} days', zorder=10)
        axs[1, 1].axvline(np.median(gaps), color='blue', linestyle='--', linewidth=2.5, 
                 label=f'Median: {np.median(gaps):.1f} days', zorder=10)
        axs[1, 1].set_title('Gap Distribution', fontweight='bold')
        axs[1, 1].set_xlabel('Days Between Lucids')
        axs[1, 1].set_ylabel('Frequency')
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3, axis='y')
        axs[1, 1].set_ylim(bottom=0)

        # Statistics message
        gap_trend = "improving 📈" if len(gaps) >= 3 and np.polyfit(range(len(gaps)), gaps, 1)[0] < 0 else "declining 📉"
        consistency_score = 100 - (np.std(gaps) / mean_gap * 100) if mean_gap > 0 else 0
        
        stats_msg = (
            f"**📊 2025 Lucid Dream Gap Statistics for {target_user.display_name}**\n\n"
            f"**Overview:**\n"
            f"• Total days with lucid dreams: **{len(lucid_reports)}**\n"
            f"• Average gap: **{mean_gap:.1f} days**\n"
            f"• Median gap: **{np.median(gaps):.1f} days**\n"
            f"• Shortest gap: **{min(gaps)} days** ✨\n"
            f"• Longest gap: **{max(gaps)} days**\n\n"
            f"**Trend Analysis:**\n"
            f"• Overall trend: **{gap_trend}**\n"
            f"• Consistency score: **{consistency_score:.1f}%** {'🎯' if consistency_score > 50 else '📊'}\n\n"
            f"**Timeline:**\n"
            f"• First lucid: **{lucid_reports[0].strftime('%d.%m.%Y')}**\n"
            f"• Last lucid: **{lucid_reports[-1].strftime('%d.%m.%Y')}**\n"
            f"• Total tracking period: **{(lucid_reports[-1] - lucid_reports[0]).days} days**\n\n"
            f"_Color coding: 🟢 ≤7 days | 🟠 8-14 days | 🔴 15-30 days | 🟤 >30 days_"
        )
        await send_long_message(ctx, stats_msg)

        # Save and send
        filename = unique_output_path(f'lucid_gaps_25_{user_id}')
        await asyncio.to_thread(plt.savefig, filename, dpi=150, bbox_inches='tight')
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='month_25', aliases=['month_previous'])
async def month_trend_25(ctx, month_str: str = None):
    """Shows monthly trends for 2025 data."""
    user_id = str(ctx.author.id)
    if user_id not in user_reports_25 or not user_reports_25[user_id]:
        await ctx.send("No 2025 reports found.")
        return

    if month_str:
        try:
            month, year = map(int, month_str.split('.'))
            if year < 100:
                year += 2000
            target_month = datetime(year=year, month=month, day=1)
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
            return
    else:
        target_month = datetime(year=2025, month=12, day=1)

    reports = []
    for report in user_reports_25[user_id]:
        report_date_str = report.get('date', '').strip()
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
            reports.append(report)

    if not reports:
        await ctx.send(f"No 2025 reports found for {target_month.strftime('%B %Y')}.")
        return

    def calculate_summary(data):
        summary = {
            "dreams": 0,
            "quality": 0,
            "wbtb": 0,
            "lucid": 0,
            "techniques": defaultdict(int)
        }
        count = len(data)
        if count == 0:
            return summary

        for report in data:
            summary["dreams"] += int(report.get("dreams", 0))
            
            quality = report.get("quality", "0")
            if '/' in quality:
                parts = quality.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1]) if len(parts) > 1 else 10
                summary["quality"] += (numerator / denominator) * 10
            else:
                summary["quality"] += int(quality)
            
            summary["wbtb"] += int(report.get("wbtb", 0))
            summary["lucid"] += int(report.get("lucid", 0))
            
            technique = report.get("technique", "").lower()
            if technique:
                summary["techniques"][technique] += 1

        summary["dreams"] = round(summary["dreams"] / count, 2)
        summary["quality"] = round(summary["quality"] / count, 2)
        summary["wbtb"] = round(summary["wbtb"] / count, 2)
        summary["lucid"] = round(summary["lucid"] / count, 2)
        
        if summary["techniques"]:
            most_used = max(summary["techniques"], key=summary["techniques"].get)
            summary["most_used"] = (most_used, summary["techniques"][most_used])
        else:
            summary["most_used"] = ("None", 0)
        
        return summary

    month_summary = calculate_summary(reports)
    response = (
        f"📊 **2025 Monthly statistic for {target_month.strftime('%B %Y')}**\n\n"
        f"Averages:\n"
        f"• dreams: {month_summary['dreams']}\n"
        f"• quality: {month_summary['quality']}/10\n"
        f"• wbtb: {month_summary['wbtb']}\n"
        f"• lucids: {month_summary['lucid']}\n"
        f"• technique: {month_summary['most_used'][0]} ({month_summary['most_used'][1]}x)"
    )
    await send_long_message(ctx, response)

    # Chart generation
    dates = []
    dreams = []
    quality = []
    wbtb = []
    lucid = []

    for report in reports:
        report_date_str = report.get('date', '')
        parsed_date = None
        for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
            try:
                parsed_date = datetime.strptime(report_date_str, fmt)
                break
            except (ValueError, TypeError, ZeroDivisionError, IndexError):
                continue
        if parsed_date:
            dates.append(parsed_date)
            dreams.append(int(report.get('dreams', 0)))
            
            qual = report.get('quality', '0')
            if '/' in qual:
                num, den = qual.split('/')[:2]
                qual_val = (int(num) / int(den)) * 10
            else:
                qual_val = int(qual)
            quality.append(qual_val)
            
            wbtb.append(int(report.get('wbtb', 0)))
            lucid.append(int(report.get('lucid', 0)))

    # Sort by date
    sorted_data = sorted(zip(dates, dreams, quality, wbtb, lucid), key=lambda x: x[0])
    if not sorted_data:
        return

    dates, dreams, quality, wbtb, lucid = zip(*sorted_data)

    plt.figure(figsize=(12, 8))
    metrics = ['dreams', 'quality', 'wbtb', 'lucid']
    
    for i, metric in enumerate(metrics, 1):
        plt.subplot(2, 2, i)
        values = []
        if metric == 'dreams':
            values = dreams
        elif metric == 'quality':
            values = quality
        elif metric == 'wbtb':
            values = wbtb
        else:
            values = lucid
        
        plt.plot(dates, values, marker='o', color='#1f77b4')
        plt.title(metric.capitalize())
        plt.xticks(rotation=45)
        plt.grid(True)

    plt.suptitle(f'2025 Monthly Trends - {target_month.strftime("%B %Y")}', fontweight='bold')
    plt.tight_layout()
    
    filename = unique_output_path(f'month_25_{user_id}_{target_month.strftime("%m_%Y")}')
    await asyncio.to_thread(plt.savefig, filename)
    plt.close()
    
    await send_generated_file(ctx, filename)


@bot.command(name='month_group_25', aliases=['month_group_previous'])
async def month_group_25(ctx, month_str: str = None):
    """Shows monthly group trends for 2025 data."""
    if not user_reports_25:
        await ctx.send("No 2025 reports found for the group.")
        return

    if month_str:
        try:
            month, year = map(int, month_str.split('.'))
            if year < 100:
                year += 2000
            target_month = datetime(year=year, month=month, day=1)
        except (ValueError, TypeError, ZeroDivisionError, IndexError):
            await ctx.send("Wrong format, please use MM.YYYY or MM.YY")
            return
    else:
        target_month = datetime(year=2025, month=12, day=1)

    all_reports = []
    for user_id, reports in user_reports_25.items():
        for report in reports:
            report_date_str = report.get('date', '').strip()
            parsed_date = None
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    parsed_date = datetime.strptime(report_date_str, fmt)
                    break
                except (ValueError, TypeError, ZeroDivisionError, IndexError):
                    continue
            if parsed_date and parsed_date.month == target_month.month and parsed_date.year == target_month.year:
                all_reports.append(report)

    if not all_reports:
        await ctx.send(f"No 2025 reports found for {target_month.strftime('%B %Y')}.")
        return

    def calculate_summary(data):
        summary = {
            "dreams": 0,
            "quality": 0,
            "wbtb": 0,
            "lucid": 0,
            "techniques": defaultdict(int)
        }
        count = len(data)
        if count == 0:
            return summary

        for report in data:
            summary["dreams"] += int(report.get("dreams", 0))
            
            quality = report.get("quality", "0")
            if '/' in quality:
                parts = quality.split('/')
                numerator = int(parts[0])
                denominator = int(parts[1]) if len(parts) > 1 else 10
                summary["quality"] += (numerator / denominator) * 10
            else:
                summary["quality"] += int(quality)
            
            summary["wbtb"] += int(report.get("wbtb", 0))
            summary["lucid"] += int(report.get("lucid", 0))
            
            technique = report.get("technique", "").lower()
            if technique:
                summary["techniques"][technique] += 1

        summary["dreams"] = round(summary["dreams"] / count, 2)
        summary["quality"] = round(summary["quality"] / count, 2)
        summary["wbtb"] = round(summary["wbtb"] / count, 2)
        summary["lucid"] = round(summary["lucid"] / count, 2)
        
        if summary["techniques"]:
            most_used = max(summary["techniques"], key=summary["techniques"].get)
            summary["most_used"] = (most_used, summary["techniques"][most_used])
        else:
            summary["most_used"] = ("None", 0)
        
        return summary

    group_summary = calculate_summary(all_reports)
    response = (
        f"📊 **2025 Group Monthly statistic for {target_month.strftime('%B %Y')}**\n\n"
        f"Total Reports: {len(all_reports)}\n"
        f"Averages:\n"
        f"• dreams: {group_summary['dreams']}\n"
        f"• quality: {group_summary['quality']}/10\n"
        f"• wbtb: {group_summary['wbtb']}\n"
        f"• lucids: {group_summary['lucid']}\n"
        f"• technique: {group_summary['most_used'][0]} ({group_summary['most_used'][1]}x)"
    )
    await send_long_message(ctx, response)


@bot.command(name='lucid_factors_25', aliases=['lucid_factors_previous'])
async def lucid_factors_25(ctx, user: discord.User = None):
    """Shows lucid factors analysis for 2025 data."""
    try:
        target_user = user if user else ctx.author
        user_id = str(target_user.id)

        if user_id not in user_reports_25 or not user_reports_25[user_id]:
            await ctx.send(f"No 2025 reports found for {target_user.display_name}.")
            return

        reports = user_reports_25[user_id]

        def normalize_quality(quality_str):
            try:
                if '/' in str(quality_str):
                    parts = str(quality_str).split('/')
                    if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                        return (int(parts[0]) / int(parts[1])) * 10
                elif str(quality_str).isdigit():
                    return int(quality_str)
                return 0
            except (ValueError, ZeroDivisionError):
                return 0

        total_dreams = sum(int(r.get('dreams', 0)) for r in reports)
        total_quality = sum(normalize_quality(r.get('quality', '0')) for r in reports)
        avg_dreams = total_dreams / len(reports) if reports else 0
        avg_quality = total_quality / len(reports) if reports else 0
        
        categories = {
            "High Recall / High Quality": [], "High Recall / Low Quality": [],
            "Low Recall / High Quality": [], "Low Recall / Low Quality": []
        }
        for r in reports:
            recall = int(r.get('dreams', 0))
            quality = normalize_quality(r.get('quality', '0'))
            if recall >= avg_dreams and quality >= avg_quality: categories["High Recall / High Quality"].append(r)
            elif recall >= avg_dreams and quality < avg_quality: categories["High Recall / Low Quality"].append(r)
            elif recall < avg_dreams and quality >= avg_quality: categories["Low Recall / High Quality"].append(r)
            else: categories["Low Recall / Low Quality"].append(r)

        summary_message = f"**📊 2025 Key Lucidity Factors for {target_user.display_name}:**\n"
        
        # Calculate lucid rates for chart
        lucid_rates = {}
        for category, report_list in categories.items():
            if not report_list:
                lucid_rates[category] = 0
                continue
            lucid_count = sum(1 for r in report_list if int(r.get('lucid', 0)) > 0)
            lucid_rates[category] = (lucid_count / len(report_list)) * 100
            rate = lucid_rates[category]
            summary_message += f"*   **{category}**: `{rate:.1f}%` Lucid Rate ({len(report_list)} reports)\n"

        await send_long_message(ctx, summary_message)

        # Chart generation
        labels = list(lucid_rates.keys())
        rates = list(lucid_rates.values())
        report_counts = [len(categories[cat]) for cat in labels]
        
        x_labels_with_counts = [f"{label}\n(n={count})" for label, count in zip(labels, report_counts)]

        plt.figure(figsize=(12, 7))
        bars = plt.bar(labels, rates, color=['#2ca02c', '#1f77b4', '#ff7f0e', '#d62728'])
        
        plt.ylabel('Lucid Dream Rate (%)')
        plt.title(f'2025 Lucid Dream Factors for {target_user.display_name}')
        plt.xticks(range(len(labels)), x_labels_with_counts, rotation=0)
        plt.ylim(0, max(rates) * 1.15 if max(rates) > 0 else 10)

        # Add percentage labels on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.1f}%', va='bottom' if yval > 0 else 'top')

        plt.tight_layout()
        
        filename = unique_output_path(f'lucid_factors_25_{user_id}')
        await asyncio.to_thread(plt.savefig, filename)
        plt.close()
        await send_generated_file(ctx, filename)

    except Exception as e:
        await send_internal_error(ctx, e)


# ============================================================
# SLASH COMMAND: /reminder
# ============================================================

@bot.tree.command(name="reminder", description="Configure your daily dream report reminder")
@app_commands.describe(
    action="Choose what to do with your reminder",
    time="Set reminder time in HH:MM format (e.g., 19:00, 08:30)",
    timezone="Your timezone (e.g., Europe/Berlin, America/New_York)"
)
@app_commands.choices(action=[
    app_commands.Choice(name="Show my current settings", value="show"),
    app_commands.Choice(name="Opt out of reminders", value="optout"),
    app_commands.Choice(name="Opt back in to reminders", value="optin"),
    app_commands.Choice(name="Set time and timezone", value="set"),
])
async def reminder_command(interaction: discord.Interaction, action: str, time: str = None, timezone: str = None):
    user_id = str(interaction.user.id)
    
    # Initialize user preferences if not exists
    if user_id not in user_preferences:
        user_preferences[user_id] = {"opted_out": False, "time": "19:00", "timezone": "UTC"}
    
    prefs = user_preferences[user_id]
    
    if action == "show":
        status = "❌ Opted out" if prefs.get("opted_out", False) else "✅ Active"
        current_time = prefs.get("time", "19:00")
        current_tz = prefs.get("timezone", "UTC")
        await interaction.response.send_message(
            f"**🔔 Your Reminder Settings:**\n"
            f"• Status: {status}\n"
            f"• Time: `{current_time}`\n"
            f"• Timezone: `{current_tz}`\n\n"
            f"Use `/reminder` with different options to change your settings.",
            ephemeral=True
        )
    
    elif action == "optout":
        prefs["opted_out"] = True
        save_preferences()
        await interaction.response.send_message(
            "✅ You have opted out of daily reminders. Use `/reminder optin` to re-enable them.",
            ephemeral=True
        )
    
    elif action == "optin":
        prefs["opted_out"] = False
        save_preferences()
        await interaction.response.send_message(
            f"✅ You are now receiving daily reminders at `{prefs.get('time', '19:00')}` ({prefs.get('timezone', 'UTC')}).",
            ephemeral=True
        )
    
    elif action == "set":
        if not time or not timezone:
            await interaction.response.send_message(
                "❌ Please provide both `time` (e.g., 19:00) and `timezone` (e.g., Europe/Berlin) when using 'Set time and timezone'.\n\n"
                f"**Available timezones:** {', '.join(COMMON_TIMEZONES)}",
                ephemeral=True
            )
            return
        
        # Validate time format
        try:
            if not re.fullmatch(r"\d{2}:\d{2}", time):
                raise ValueError("Invalid time format")
            hour, minute = map(int, time.split(':'))
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time")
        except (ValueError, TypeError):
            await interaction.response.send_message(
                "❌ Invalid time format. Please use HH:MM (e.g., 19:00, 08:30).",
                ephemeral=True
            )
            return
        
        # Validate timezone
        try:
            pytz.timezone(timezone)
        except (pytz.UnknownTimeZoneError, AttributeError):
            await interaction.response.send_message(
                f"❌ Invalid timezone. Please use a valid timezone.\n\n"
                f"**Example timezones:** {', '.join(COMMON_TIMEZONES)}",
                ephemeral=True
            )
            return
        
        prefs["time"] = time
        prefs["timezone"] = timezone
        prefs["opted_out"] = False  # Re-enable if they're setting a new time
        save_preferences()
        
        await interaction.response.send_message(
            f"✅ Your reminder has been set for `{time}` in timezone `{timezone}`.",
            ephemeral=True
        )


@reminder_command.autocomplete('timezone')
async def timezone_autocomplete(interaction: discord.Interaction, current: str):
    current_lower = current.lower()
    choices = []
    for tz in COMMON_TIMEZONES:
        if current_lower in tz.lower():
            choices.append(app_commands.Choice(name=tz, value=tz))
            if len(choices) == 25:
                break
    return choices


# ============================================================
# NEW COMMANDS
# ============================================================

@bot.command(name='inactive')
async def inactive(ctx):
    """Shows users who haven't submitted a report in the last 2 weeks.
    
    Restricted to specific user IDs only.
    """
    # Check if user is allowed to use this command
    if ctx.author.id not in ALLOWED_INACTIVE_USERS:
        await ctx.send("❌ You don't have permission to use this command.")
        return
    
    try:
        today = datetime.now(timezone.utc).date()
        two_weeks_ago = today - timedelta(days=14)
        
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt).date()
                except ValueError:
                    continue
            return None
        
        inactive_users = []
        
        for user_id, reports in user_reports.items():
            if not reports:
                # User has no reports at all
                inactive_users.append((user_id, None, "No reports"))
                continue
            
            # Find the most recent report date
            most_recent_date = None
            for report in reports:
                report_date = parse_date(report.get("date", ""))
                if report_date:
                    if most_recent_date is None or report_date > most_recent_date:
                        most_recent_date = report_date
            
            if most_recent_date is None:
                inactive_users.append((user_id, None, "No valid dates"))
            elif most_recent_date < two_weeks_ago:
                days_since = (today - most_recent_date).days
                inactive_users.append((user_id, most_recent_date, f"{days_since} days ago"))
        
        if not inactive_users:
            await ctx.send("✅ All users have reported within the last 2 weeks!")
            return
        
        # Sort by days since (most inactive first)
        inactive_users.sort(key=lambda x: x[1] if x[1] else date.min)
        
        response = "**🔕 Users who haven't reported in 2+ weeks:**\n\n"
        for user_id, last_date, description in inactive_users:
            date_str = last_date.strftime("%d.%m.%Y") if last_date else "N/A"
            response += f"• <@{user_id}> - Last report: {date_str} ({description})\n"
        
        await send_long_message(ctx, response, allowed_mentions=discord.AllowedMentions(users=True))
        
    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='edit')
async def edit(ctx, source_date: str = None, target_date: str = None):
    """Edit an existing report. By default edits the most recent report.
    
    Format:
    !edit
    Date: 24.01.26  (optional - defaults to most recent)
    Quality: 8
    Notes: updated notes
    
    To move a report to a different date:
    !edit 25.02.26 24.02.26
    """
    try:
        user_id = str(ctx.author.id)
        
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("You have no reports to edit.")
            return
        
        def parse_date(date_str):
            for fmt in ("%d.%m.%Y", "%d.%m.%y", "%d/%m/%Y", "%d/%m/%y"):
                try:
                    return datetime.strptime(date_str.strip(), fmt)
                except ValueError:
                    continue
            return None
        
        # --- Mode 1: Move a report to a different date (!edit 25.02.26 24.02.26) ---
        if source_date and target_date:
            parsed_source = parse_date(source_date)
            parsed_target = parse_date(target_date)
            
            if not parsed_source:
                await ctx.send(f"Invalid source date format: {source_date}")
                return
            if not parsed_target:
                await ctx.send(f"Invalid target date format: {target_date}")
                return
            if parsed_target.year != ACTIVE_REPORT_YEAR:
                await ctx.send(f"Report dates must remain in {ACTIVE_REPORT_YEAR}.")
                return
            user_tz_name = user_preferences.get(user_id, {}).get("timezone", "UTC")
            try:
                local_today = datetime.now(pytz.timezone(user_tz_name)).date()
            except (pytz.UnknownTimeZoneError, AttributeError):
                local_today = datetime.now(timezone.utc).date()
            if parsed_target.date() > local_today:
                await ctx.send("Reports cannot be moved to a future date.")
                return
            
            # Find all reports matching the source date
            matching_reports = []
            matching_indices = []
            for i, report in enumerate(user_reports[user_id]):
                report_date = parse_date(report.get("date", ""))
                if report_date and report_date.date() == parsed_source.date():
                    matching_reports.append(report)
                    matching_indices.append(i)
            
            if not matching_reports:
                await ctx.send(f"No report found for date: {source_date}")
                return
            
            # Select the report to move
            if len(matching_reports) == 1:
                chosen_report = matching_reports[0].copy()
                chosen_index = matching_indices[0]
            else:
                # Multiple reports — prompt the user to choose
                report_list = "\n".join(
                    [f"{idx + 1}: {json.dumps(report, indent=4)}" for idx, report in enumerate(matching_reports)]
                )
                await ctx.send(f"Multiple reports found for {source_date}. Please choose one to move by typing its number:\n{report_list}")
                
                def check(message):
                    return message.author == ctx.author and message.channel == ctx.channel and message.content.isdigit()
                
                try:
                    response = await bot.wait_for('message', timeout=180, check=check)
                except asyncio.TimeoutError:
                    await ctx.send("No response received in 3 minutes. Edit cancelled.")
                    return
                selection = int(response.content) - 1
                
                if 0 <= selection < len(matching_reports):
                    chosen_report = matching_reports[selection].copy()
                    chosen_index = matching_indices[selection]
                else:
                    await ctx.send("Invalid selection. Edit cancelled.")
                    return
            
            # Change the date
            old_date = chosen_report.get("date", "N/A")
            new_date_formatted = parsed_target.strftime("%d.%m.%Y")
            chosen_report["date"] = new_date_formatted
            user_reports[user_id][chosen_index] = chosen_report
            user_reports[user_id].sort(key=lambda item: parse_report_date(item.get("date", "")) or datetime.min)
            atomic_write_json(DATA_FILE_26, user_reports)
            
            await ctx.send(f"✅ **Report date changed:** `{old_date}` → `{new_date_formatted}`")
            return
        
        # --- Mode 2: Edit fields on a report (!edit with body fields) ---
        content = ctx.message.content.split('\n', 1)
        if len(content) < 2 and not source_date:
            await ctx.send("Please provide fields to edit after `!edit`.\n\nExample:\n```\n!edit\nQuality: 8\nNotes: updated notes\n```\n\nTo move a report to another date:\n```\n!edit 25.02.26 24.02.26\n```")
            return
        
        # Parse body fields (if any)
        edit_data = {}
        if len(content) >= 2:
            lines = content[1].split('\n')
            for line in lines:
                if ':' in line:
                    key, _, value = line.partition(':')
                    normalized_key = key.strip().lower().replace(' ', '_')
                    edit_data[normalized_key] = value.strip()

        unknown_fields = sorted(set(edit_data) - REPORT_ALLOWED_FIELDS)
        if unknown_fields:
            await ctx.send(f"Unknown field(s): {', '.join(unknown_fields)}. Edit cancelled.")
            return
        
        # Find the target report
        target_report = None
        target_index = None
        
        # Use source_date from command args, or "date" from body fields
        lookup_date_str = source_date if source_date else edit_data.get("date")
        
        if lookup_date_str:
            lookup_date = parse_date(lookup_date_str)
            if not lookup_date:
                await ctx.send(f"Invalid date format: {lookup_date_str}")
                return
            
            matching_reports = []
            matching_indices = []
            for i, report in enumerate(user_reports[user_id]):
                report_date = parse_date(report.get("date", ""))
                if report_date and report_date.date() == lookup_date.date():
                    matching_reports.append(report)
                    matching_indices.append(i)
            
            if not matching_reports:
                await ctx.send(f"No report found for date: {lookup_date_str}")
                return
            
            if len(matching_reports) == 1:
                target_report = matching_reports[0].copy()
                target_index = matching_indices[0]
            else:
                # Multiple reports — prompt the user to choose
                report_list = "\n".join(
                    [f"{idx + 1}: {json.dumps(report, indent=4)}" for idx, report in enumerate(matching_reports)]
                )
                await ctx.send(f"Multiple reports found for {lookup_date_str}. Please choose one to edit by typing its number:\n{report_list}")
                
                def check(message):
                    return message.author == ctx.author and message.channel == ctx.channel and message.content.isdigit()
                
                try:
                    response = await bot.wait_for('message', timeout=180, check=check)
                except asyncio.TimeoutError:
                    await ctx.send("No response received in 3 minutes. Edit cancelled.")
                    return
                chosen_index = int(response.content) - 1
                
                if 0 <= chosen_index < len(matching_reports):
                    target_report = matching_reports[chosen_index].copy()
                    target_index = matching_indices[chosen_index]
                else:
                    await ctx.send("Invalid selection. Edit cancelled.")
                    return
            
            # Remove date from edit_data if it was there (used for lookup only)
            if "date" in edit_data:
                del edit_data["date"]
        else:
            # Default to the chronologically most recent report.
            target_index = max(
                range(len(user_reports[user_id])),
                key=lambda i: parse_report_date(user_reports[user_id][i].get("date", "")) or datetime.min,
            )
            target_report = user_reports[user_id][target_index].copy()
        
        if not edit_data:
            await ctx.send("No fields to update (only date was provided).")
            return
        
        # Validate numeric fields if present
        for field in ["dreams", "lucid"]:
            if field in edit_data:
                value = normalized_nonnegative_int(edit_data[field])
                if value is None:
                    await ctx.send(f"The '{field}' field must be a non-negative integer.")
                    return
                edit_data[field] = str(value)

        if "wbtb" in edit_data:
            value = normalized_nonnegative_int(edit_data["wbtb"])
            if value is None:
                await ctx.send("The 'wbtb' field must be a non-negative integer.")
                return
            edit_data["wbtb"] = str(value)
        
        if "sleep_time" in edit_data:
            value = normalized_finite_float(edit_data["sleep_time"], maximum=24)
            if value is None:
                await ctx.send("The 'sleep_time' field must be between 0 and 24.")
                return
            edit_data["sleep_time"] = str(value)
        
        if "quality" in edit_data:
            if normalized_quality(edit_data["quality"]) is None:
                await ctx.send("The 'quality' field must be from 0-10 or a valid X/Y ratio.")
                return

        if "focus" in edit_data:
            value = normalized_nonnegative_int(edit_data["focus"], maximum=10)
            if value is None:
                await ctx.send("The 'focus' field must be an integer from 0-10.")
                return
            edit_data["focus"] = str(value)

        if "journal_time" in edit_data:
            value = normalized_nonnegative_int(edit_data["journal_time"])
            if value is None:
                await ctx.send("The 'journal_time' field must be a non-negative integer.")
                return
            edit_data["journal_time"] = str(value)
        
        # Apply edits
        updated_fields = []
        for key, value in edit_data.items():
            old_value = target_report.get(key, "N/A")
            target_report[key] = value
            updated_fields.append(f"• **{key}**: `{old_value}` → `{value}`")
        
        user_reports[user_id][target_index] = target_report
        
        # Save to file
        user_reports[user_id].sort(key=lambda item: parse_report_date(item.get("date", "")) or datetime.min)
        atomic_write_json(DATA_FILE_26, user_reports)
        
        response = f"✅ **Report for {target_report.get('date', 'Unknown')} updated:**\n" + "\n".join(updated_fields)
        await send_long_message(ctx, response)
        
    except Exception as e:
        await send_internal_error(ctx, e)


@bot.command(name='journal')
async def journal(ctx):
    """Shows journal time statistics: total time, averages, and weekly breakdown."""
    try:
        user_id = str(ctx.author.id)
        
        if user_id not in user_reports or not user_reports[user_id]:
            await ctx.send("No reports found for you.")
            return
        
        reports = user_reports[user_id]
        
        # Filter reports with journal_time
        reports_with_journal = []
        for r in reports:
            jt = r.get("journal_time") or r.get("journal time")
            if jt:
                try:
                    journal_mins = float(jt)
                    if journal_mins > 0:
                        reports_with_journal.append((r, journal_mins))
                except (ValueError, TypeError):
                    continue
        
        if not reports_with_journal:
            await ctx.send("No reports with journal_time found.")
            return
        
        # Calculate total and average
        total_time = sum(jt for _, jt in reports_with_journal)
        avg_time = total_time / len(reports_with_journal)
        
        # Compare adjacent seven-calendar-day periods.
        anchor = user_local_today(user_id)
        current_start = anchor - timedelta(days=6)
        previous_start = anchor - timedelta(days=13)
        this_week_times = [
            jt for report, jt in reports_with_journal
            if (parse_report_date(report.get("date", ""))
                and current_start <= parse_report_date(report.get("date", "")).date() <= anchor)
        ]
        last_week_times = [
            jt for report, jt in reports_with_journal
            if (parse_report_date(report.get("date", ""))
                and previous_start <= parse_report_date(report.get("date", "")).date() < current_start)
        ]
        
        this_week_avg = sum(this_week_times) / len(this_week_times) if this_week_times else 0
        last_week_avg = sum(last_week_times) / len(last_week_times) if last_week_times else 0
        
        # Format response
        response = (
            "📓 **Journal Time Statistics**\n\n"
            f"**Total Time Spent Journaling:** `{total_time:.1f}` minutes ({total_time/60:.1f} hours)\n"
            f"**Reports with Journal Time:** `{len(reports_with_journal)}`\n"
            f"**Overall Average:** `{avg_time:.1f}` min/report\n\n"
            f"**This Week Average:** `{this_week_avg:.1f}` min ({len(this_week_times)} reports)\n"
        )
        
        if last_week_times:
            change = this_week_avg - last_week_avg
            change_str = f"+{change:.1f}" if change >= 0 else f"{change:.1f}"
            response += f"**Last Week Average:** `{last_week_avg:.1f}` min ({len(last_week_times)} reports)\n"
            response += f"**Change:** `{change_str}` min/report"
        else:
            response += "_Not enough data for last week comparison._"
        
        await send_long_message(ctx, response)
        
    except Exception as e:
        await send_internal_error(ctx, e)


def analysis_source_for_year(year):
    if year == ACTIVE_REPORT_YEAR:
        return user_reports
    if year == 2025:
        return user_reports_25
    return None


# Replace the legacy analysis registrations with the normalized command suite.
for command_name in (
    'trend', 'score', 'group', 'personal_recap', 'final_group', 'month', 'month_group',
    'dreams', 'correlate', 'effectiveness', 'wbtb_impact', 'lucid_factors', 'overview',
    'heatmap', 'day_of_week', 'journaltime', 'journaltime_all', 'journal', 'lucid_history',
    'lucid_gaps', 'overview_25', 'month_25', 'wbtb_impact_25', 'effectiveness_25',
    'lucid_gaps_25', 'dreams_25', 'personal_recap_25', 'final_group_25',
    'journaltime_25', 'correlate_25', 'day_of_week_25', 'lucid_factors_25',
    'month_group_25',
):
    bot.remove_command(command_name)


async def setup_analysis_commands():
    # Let the manager's SIGTERM close the Discord session before forced shutdown.
    if os.name == 'posix':
        asyncio.get_running_loop().add_signal_handler(
            signal.SIGTERM, lambda: asyncio.create_task(bot.close())
        )
    await bot.add_cog(AnalysisCommands(
        bot=bot,
        active_year=ACTIVE_REPORT_YEAR,
        source_provider=analysis_source_for_year,
        local_today=lambda user_id: user_local_today(user_id),
        output_path=unique_output_path,
        send_long=send_long_message,
        send_file=send_generated_file,
    ))


bot.setup_hook = setup_analysis_commands


if __name__ == '__main__':
    if not DISCORD_TOKEN:
        raise RuntimeError("DISCORD_TOKEN is not set. Add it to the environment or .env file.")
    bot.run(DISCORD_TOKEN)
