from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
]
# <link rel="stylesheet" type="text/css" href="{% static 'css/style.css' %}">
