"""
Timezone-aware scheduling service for proactive companion check-ins.
Supports configurable daily check-in slots, quiet hours and frequency limits,
idempotency, and safe channel selection.
"""
import datetime
from typing import List, Optional
from django.utils import timezone
from django.conf import settings
from proactive_messaging.models import ScheduledMessage, CompanionPreference
from proactive_messaging.services.delivery.push import PushNotificationDeliveryHandler


def create_daily_checkin_schedule(
    user,
    target_date: Optional[datetime.date] = None,
    force: bool = False
) -> List[ScheduledMessage]:
    """
    Schedules sensible, natural daily companion check-ins for the specified user
    based on their timezone, quiet hours, and frequency settings.

    Standard slots:
    1. Good Morning check-in (e.g. 08:30 local)
    2. Midday / Casual check-in (e.g. 13:30 local)
    3. Evening check-in (e.g. 19:30 local)

    Ensures idempotency via deterministic idempotency_key so running this multiple
    times per day will not create duplicate messages.
    """
    pref, _ = CompanionPreference.objects.get_or_create(user=user)

    if (not pref.proactive_messaging_enabled or pref.frequency == CompanionPreference.FrequencyChoice.DISABLED) and not force:
        return []

    tz = pref.get_tz()
    now_local = datetime.datetime.now(tz)
    if target_date is None:
        target_date = now_local.date()

    # Enforce few_times_week: Monday (0), Wednesday (2), Friday (4) in user's timezone
    if pref.frequency == CompanionPreference.FrequencyChoice.FEW_TIMES_WEEK and not force:
        if target_date.weekday() not in (0, 2, 4):
            return []

    # Determine default candidate slots (hour, minute, message_type)
    candidate_slots = [
        (datetime.time(8, 30), ScheduledMessage.MessageType.GOOD_MORNING),
        (datetime.time(13, 30), ScheduledMessage.MessageType.CASUAL_CHAT),
        (datetime.time(19, 30), ScheduledMessage.MessageType.EVENING_CHECKIN),
    ]

    # Select delivery channel based on whether push provider is configured and user has active devices
    push_configured = PushNotificationDeliveryHandler().is_configured()
    has_active_push = push_configured and user.push_subscriptions.filter(is_active=True).exists()
    default_channel = (
        ScheduledMessage.DeliveryChannel.BOTH
        if has_active_push
        else ScheduledMessage.DeliveryChannel.IN_APP
    )

    scheduled_items = []
    scheduled_times_today = []

    # Frequency cap: 1 check-in per day for daily & few_times_week; otherwise max_daily_messages
    max_slots = 1 if pref.frequency in (CompanionPreference.FrequencyChoice.DAILY, CompanionPreference.FrequencyChoice.FEW_TIMES_WEEK) else pref.max_daily_messages

    # Check existing scheduled / delivered messages for target_date to avoid exceeding daily max
    start_of_day_utc = datetime.datetime.combine(target_date, datetime.time.min, tzinfo=tz).astimezone(datetime.timezone.utc)
    end_of_day_utc = datetime.datetime.combine(target_date, datetime.time.max, tzinfo=tz).astimezone(datetime.timezone.utc)

    candidate_keys = [f"daily_{user.id}_{msg_type}_{target_date.isoformat()}" for _, msg_type in candidate_slots]
    existing_other_count = ScheduledMessage.objects.filter(
        user=user,
        scheduled_time__gte=start_of_day_utc,
        scheduled_time__lte=end_of_day_utc
    ).exclude(status=ScheduledMessage.DeliveryStatus.CANCELLED).exclude(idempotency_key__in=candidate_keys).count()

    for slot_time, msg_type in candidate_slots:
        if len(scheduled_items) >= max_slots:
            break

        idempotency_key = f"daily_{user.id}_{msg_type}_{target_date.isoformat()}"
        existing_msg = ScheduledMessage.objects.filter(idempotency_key=idempotency_key).first()
        if existing_msg:
            scheduled_items.append(existing_msg)
            scheduled_times_today.append(existing_msg.scheduled_time)
            continue

        if len(scheduled_items) + existing_other_count >= max_slots:
            break

        slot_dt_local = datetime.datetime.combine(target_date, slot_time, tzinfo=tz)
        slot_dt_utc = slot_dt_local.astimezone(datetime.timezone.utc)

        # Skip if within quiet hours
        if pref.is_in_quiet_hours(slot_dt_utc):
            continue

        # Check cooldown with other slots scheduled in this batch
        cooldown_violated = False
        for prev_time in scheduled_times_today:
            elapsed = abs((slot_dt_utc - prev_time).total_seconds())
            if elapsed < pref.min_minutes_between_messages * 60:
                cooldown_violated = True
                break

        if cooldown_violated:
            continue

        msg = ScheduledMessage.objects.create(
            user=user,
            message_type=msg_type,
            scheduled_time=slot_dt_utc,
            channel=default_channel,
            status=ScheduledMessage.DeliveryStatus.PENDING,
            is_ai_generated=True,
            idempotency_key=idempotency_key,
            max_attempts=3,
        )

        scheduled_items.append(msg)
        scheduled_times_today.append(slot_dt_utc)

    return scheduled_items


def schedule_single_checkin(
    user,
    message_type: str = ScheduledMessage.MessageType.CASUAL_CHAT,
    scheduled_time: Optional[datetime.datetime] = None,
    content: str = "",
    channel: Optional[str] = None,
    idempotency_key: Optional[str] = None,
    ai_prompt_context: Optional[dict] = None
) -> ScheduledMessage:
    """
    Schedules an individual check-in message for a user.
    """
    pref, _ = CompanionPreference.objects.get_or_create(user=user)

    if scheduled_time is None:
        scheduled_time = timezone.now() + datetime.timedelta(minutes=1)
    elif timezone.is_naive(scheduled_time):
        tz = pref.get_tz()
        scheduled_time = timezone.make_aware(scheduled_time, tz).astimezone(datetime.timezone.utc)

    if channel is None:
        push_configured = PushNotificationDeliveryHandler().is_configured()
        channel = (
            ScheduledMessage.DeliveryChannel.BOTH
            if push_configured
            else ScheduledMessage.DeliveryChannel.IN_APP
        )

    is_ai_gen = not bool(content and content.strip())

    msg = ScheduledMessage.objects.create(
        user=user,
        message_type=message_type,
        scheduled_time=scheduled_time,
        content=content.strip(),
        is_ai_generated=is_ai_gen,
        channel=channel,
        idempotency_key=idempotency_key,
        ai_prompt_context=ai_prompt_context or {},
        status=ScheduledMessage.DeliveryStatus.PENDING,
    )
    return msg
