"""Google Calendar sync + ICS export for the MBA Study Tracker."""

from datetime import datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from study_data import DAILY_PLAN, SUBJECTS

BASE_DIR = Path(__file__).parent
CREDENTIALS_FILE = BASE_DIR / "credentials.json"
TOKEN_FILE = BASE_DIR / "token.json"
ICS_FILE = BASE_DIR / "study_plan.ics"

SCOPES = ["https://www.googleapis.com/auth/calendar"]

# Time slots per subject (local time, 24h)
SUBJECT_TIMES: dict[str, tuple[int, int]] = {
    "Mathematics":      (6, 0),
    "Statistics":       (7, 30),
    "Economics":        (19, 0),
    "Machine Learning": (13, 0),
    "Quant Finance":    (6, 0),
    "Financial Markets": (9, 0),
}

# Google Calendar color IDs
SUBJECT_COLORS: dict[str, str] = {
    "Mathematics":      "9",   # Blueberry
    "Statistics":       "6",   # Tangerine
    "Economics":        "2",   # Sage
    "Machine Learning": "3",   # Grape
    "Quant Finance":    "10",  # Basil
    "Financial Markets": "4",  # Flamingo
}

TZ = ZoneInfo("America/New_York")


def _get_google_creds():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_FILE.exists():
                raise FileNotFoundError(f"credentials.json not found at {CREDENTIALS_FILE}")
            flow = InstalledAppFlow.from_client_secrets_file(str(CREDENTIALS_FILE), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def _get_service():
    from googleapiclient.discovery import build
    return build("calendar", "v3", credentials=_get_google_creds())


def sync_to_google() -> int:
    service = _get_service()
    count = 0
    for study_date, tasks in DAILY_PLAN.items():
        for task in tasks:
            subj = task["subject"]
            hour, minute = SUBJECT_TIMES[subj]
            start_dt = datetime.combine(
                study_date, time(hour, minute), tzinfo=TZ
            )
            end_dt = start_dt + timedelta(hours=task["duration_hrs"])

            event = {
                "summary": f"[{subj}] {task['title']}",
                "description": f"{task['details']}\n\n📖 Resources: {task['resources']}",
                "start": {"dateTime": start_dt.isoformat(), "timeZone": "America/New_York"},
                "end":   {"dateTime": end_dt.isoformat(),   "timeZone": "America/New_York"},
                "colorId": SUBJECT_COLORS[subj],
                "extendedProperties": {"private": {"mbaStudyTracker": "true"}},
            }
            service.events().insert(calendarId="primary", body=event).execute()
            count += 1
    return count


def delete_synced_events() -> int:
    service = _get_service()
    count = 0
    page_token = None
    while True:
        resp = service.events().list(
            calendarId="primary",
            privateExtendedProperty="mbaStudyTracker=true",
            pageToken=page_token,
            maxResults=250,
        ).execute()
        for event in resp.get("items", []):
            service.events().delete(calendarId="primary", eventId=event["id"]).execute()
            count += 1
        page_token = resp.get("nextPageToken")
        if not page_token:
            break
    return count


def generate_ics() -> Path:
    from icalendar import Calendar, Event

    cal = Calendar()
    cal.add("prodid", "-//MBA Study Tracker//EN")
    cal.add("version", "2.0")
    cal.add("x-wr-calname", "MBA Study Plan 2026")

    for study_date, tasks in sorted(DAILY_PLAN.items()):
        for idx, task in enumerate(tasks):
            subj = task["subject"]
            hour, minute = SUBJECT_TIMES[subj]
            start_dt = datetime.combine(study_date, time(hour, minute), tzinfo=TZ)
            end_dt = start_dt + timedelta(hours=task["duration_hrs"])

            ev = Event()
            ev.add("summary", f"[{subj}] {task['title']}")
            ev.add("description", f"{task['details']}\n\nResources: {task['resources']}")
            ev.add("dtstart", start_dt)
            ev.add("dtend", end_dt)
            ev.add("uid", f"mba-study-{study_date.isoformat()}-{idx}@harsha")
            cal.add_component(ev)

    ICS_FILE.write_bytes(cal.to_ical())
    return ICS_FILE
