from django.contrib import admin
from .models import ScheduledMessage, CompanionPreference, InAppMessage, PushDeviceSubscription


@admin.register(ScheduledMessage)
class ScheduledMessageAdmin(admin.ModelAdmin):
    list_display = (
        'id',
        'user',
        'message_type',
        'scheduled_time',
        'status',
        'channel',
        'is_ai_generated',
        'delivered_in_app',
        'delivered_push',
        'created_at',
    )
    list_filter = ('status', 'message_type', 'channel', 'is_ai_generated')
    search_fields = ('user__username', 'user__email', 'content', 'idempotency_key')
    readonly_fields = ('id', 'created_at', 'updated_at', 'delivered_at')
    ordering = ('scheduled_time',)


@admin.register(CompanionPreference)
class CompanionPreferenceAdmin(admin.ModelAdmin):
    list_display = (
        'user',
        'companion_name',
        'preferred_language',
        'companion_tone',
        'proactive_messaging_enabled',
        'quiet_hours_enabled',
        'max_daily_messages',
    )
    search_fields = ('user__username', 'companion_name')


@admin.register(InAppMessage)
class InAppMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'sender_name', 'message_type', 'is_read', 'created_at')
    list_filter = ('is_read', 'message_type')
    search_fields = ('user__username', 'content')


@admin.register(PushDeviceSubscription)
class PushDeviceSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'device_type', 'is_active', 'created_at')
    list_filter = ('device_type', 'is_active')

