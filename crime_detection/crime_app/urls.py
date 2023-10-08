from django.urls import path
from . import views

urlpatterns = [
    # Use the root URL for video uploads
    path('', views.upload_view, name='upload'),
    # URL for displaying uploaded videos
    path('videos/', views.videos_view, name='videos'),

]
