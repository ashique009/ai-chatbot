"""
Django management command to periodically process due proactive scheduled messages.

Usage:
    # Run once (suitable for cron or one-off verification):
    python manage.py process_due_messages --once

    # Run continuously as a background worker:
    python manage.py process_due_messages --loop --interval 30
"""
import time
import sys
from django.core.management.base import BaseCommand
from django.utils import timezone
from proactive_messaging.services.worker import ScheduledMessageWorker


class Command(BaseCommand):
    help = "Finds due scheduled messages, claims them atomically, and delivers personalized partner messages."

    def add_arguments(self, parser):
        parser.add_argument(
            '--once',
            action='store_true',
            default=False,
            help='Run a single pass of due messages and exit immediately.'
        )
        parser.add_argument(
            '--loop',
            action='store_true',
            default=False,
            help='Run continuously as a background daemon worker.'
        )
        parser.add_argument(
            '--interval',
            type=int,
            default=30,
            help='Seconds to sleep between sweeps when running in loop mode (default: 30).'
        )
        parser.add_argument(
            '--limit',
            type=int,
            default=50,
            help='Maximum number of messages to process per batch (default: 50).'
        )

    def handle(self, *args, **options):
        once = options['once']
        loop = options['loop']
        interval = options['interval']
        limit = options['limit']

        # Default to once if loop is not explicitly passed
        if not loop:
            once = True

        worker = ScheduledMessageWorker()

        self.stdout.write(
            self.style.SUCCESS(
                f"[{timezone.now().isoformat()}] Proactive messaging worker started "
                f"(mode={'continuous loop (' + str(interval) + 's)' if loop else 'single run'})."
            )
        )

        try:
            while True:
                start_time = timezone.now()
                results = worker.process_due_messages(limit=limit)

                if results['total_due'] > 0:
                    self.stdout.write(
                        self.style.NOTICE(
                            f"Processed batch: {results['total_due']} due | "
                            f"{results['delivered']} delivered | "
                            f"{results['retried']} retried | "
                            f"{results['failed']} failed | "
                            f"{results['postponed']} postponed | "
                            f"{results['skipped']} skipped (already claimed) | "
                            f"{results['cancelled']} cancelled"
                        )
                    )
                elif not loop:
                    self.stdout.write("No due messages found to process.")

                if once:
                    break

                time.sleep(interval)

        except KeyboardInterrupt:
            self.stdout.write(self.style.WARNING("\nWorker shutdown requested. Exiting cleanly."))
            sys.exit(0)
