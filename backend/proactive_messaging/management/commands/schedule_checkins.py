"""
Django management command to generate proactive companion check-ins for active users.

Usage:
    # Schedule check-ins for all active users:
    python manage.py schedule_checkins

    # Schedule for a specific user:
    python manage.py schedule_checkins --username ashique_user

    # Schedule for a specific date (YYYY-MM-DD):
    python manage.py schedule_checkins --date 2026-10-15
"""
import datetime
from django.core.management.base import BaseCommand
from django.contrib.auth import get_user_model
from proactive_messaging.services.scheduler import create_daily_checkin_schedule

User = get_user_model()


class Command(BaseCommand):
    help = "Generates timezone-aware daily companion check-in schedules for users with proactive messaging enabled."

    def add_arguments(self, parser):
        parser.add_argument(
            '--username',
            type=str,
            default=None,
            help='Schedule check-ins only for the specified username.'
        )
        parser.add_argument(
            '--date',
            type=str,
            default=None,
            help='Target schedule date formatted as YYYY-MM-DD (defaults to today in user timezone).'
        )
        parser.add_argument(
            '--force',
            action='store_true',
            default=False,
            help='Force schedule even if proactive_messaging_enabled is False.'
        )

    def handle(self, *args, **options):
        username = options['username']
        date_str = options['date']
        force = options['force']

        target_date = None
        if date_str:
            try:
                target_date = datetime.date.fromisoformat(date_str)
            except ValueError:
                self.stderr.write(self.style.ERROR(f"Invalid date format '{date_str}'. Use YYYY-MM-DD."))
                return

        users_qs = User.objects.filter(is_active=True)
        if username:
            users_qs = users_qs.filter(username=username)

        total_scheduled = 0
        user_count = 0

        for user in users_qs:
            items = create_daily_checkin_schedule(user, target_date=target_date, force=force)
            if items:
                user_count += 1
                total_scheduled += len(items)
                self.stdout.write(
                    self.style.SUCCESS(
                        f"Scheduled {len(items)} check-ins for user '{user.username}'."
                    )
                )

        self.stdout.write(
            self.style.NOTICE(
                f"Completed: {total_scheduled} check-ins scheduled across {user_count} users."
            )
        )
