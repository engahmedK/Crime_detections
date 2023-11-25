import os
from django.db import models
from django.utils import timezone


def unique_video_filename(instance, filename):
    title = os.path.splitext(filename)[0]
    return f'videos/{title}/{filename}'


class Video(models.Model):
    title = models.CharField(max_length=100)
    video_file = models.FileField(upload_to=unique_video_filename)
    uploaded_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return os.path.basename(self.video_file.name)
