from django.contrib import admin
from .models import Conversation, ChatMessage, CompanionMemory


class ChatMessageInline(admin.TabularInline):
    model = ChatMessage
    extra = 0
    readonly_fields = ('id', 'sender', 'content', 'created_at')


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'title', 'message_count', 'updated_at', 'created_at')
    search_fields = ('title', 'user__username', 'id')
    readonly_fields = ('id', 'created_at', 'updated_at')
    inlines = [ChatMessageInline]


@admin.register(ChatMessage)
class ChatMessageAdmin(admin.ModelAdmin):
    list_display = ('id', 'conversation', 'sender', 'content_snippet', 'created_at')
    list_filter = ('sender', 'created_at')
    search_fields = ('content', 'conversation__id')
    readonly_fields = ('id', 'created_at')

    def content_snippet(self, obj):
        return obj.content[:50]


@admin.register(CompanionMemory)
class CompanionMemoryAdmin(admin.ModelAdmin):
    list_display = ('id', 'user', 'category', 'content_snippet', 'importance', 'is_approved', 'updated_at')
    list_filter = ('category', 'importance', 'is_approved')
    search_fields = ('content', 'user__username')
    readonly_fields = ('id', 'created_at', 'updated_at')

    def content_snippet(self, obj):
        return obj.content[:50]
