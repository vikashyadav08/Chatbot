from django.urls import path
from . import views

urlpatterns = [
    path('', views.chatbot_response, name='chat'),
]
