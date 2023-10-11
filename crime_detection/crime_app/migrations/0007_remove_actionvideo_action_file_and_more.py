# Generated by Django 4.1.2 on 2023-10-09 19:31

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('crime_app', '0006_platevideo_facevideo_enhancedvideo_actionvideo'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='actionvideo',
            name='action_file',
        ),
        migrations.RemoveField(
            model_name='enhancedvideo',
            name='enhanced_file',
        ),
        migrations.RemoveField(
            model_name='facevideo',
            name='face_file',
        ),
        migrations.RemoveField(
            model_name='platevideo',
            name='plate_file',
        ),
        migrations.AddField(
            model_name='actionvideo',
            name='action_video_file',
            field=models.FileField(blank=True, null=True, upload_to='output/action/'),
        ),
        migrations.AddField(
            model_name='actionvideo',
            name='title',
            field=models.CharField(default='untitle', max_length=255),
        ),
        migrations.AddField(
            model_name='enhancedvideo',
            name='enhanced_video_file',
            field=models.FileField(blank=True, null=True, upload_to='output/enhanced/'),
        ),
        migrations.AddField(
            model_name='enhancedvideo',
            name='title',
            field=models.CharField(default='untitle', max_length=255),
        ),
        migrations.AddField(
            model_name='facevideo',
            name='face_video_file',
            field=models.FileField(blank=True, null=True, upload_to='output/face/'),
        ),
        migrations.AddField(
            model_name='facevideo',
            name='title',
            field=models.CharField(default='untitle', max_length=255),
        ),
        migrations.AddField(
            model_name='platevideo',
            name='plate_video_file',
            field=models.FileField(blank=True, null=True, upload_to='output/plate/'),
        ),
        migrations.AddField(
            model_name='platevideo',
            name='title',
            field=models.CharField(default='untitle', max_length=255),
        ),
        migrations.AlterField(
            model_name='actionvideo',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='action_videos', to='crime_app.video'),
        ),
        migrations.AlterField(
            model_name='enhancedvideo',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='enhanced_videos', to='crime_app.video'),
        ),
        migrations.AlterField(
            model_name='facevideo',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='face_videos', to='crime_app.video'),
        ),
        migrations.AlterField(
            model_name='platevideo',
            name='video',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='plate_videos', to='crime_app.video'),
        ),
        migrations.AlterField(
            model_name='video',
            name='title',
            field=models.CharField(default='untitle', max_length=255),
        ),
    ]