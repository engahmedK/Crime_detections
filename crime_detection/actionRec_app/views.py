import os
from django.utils.text import slugify
from django.shortcuts import render, redirect
from .forms import VideoForm
from .models import Video


def update_video(request, video_id):
    video = Video.objects.get(pk=video_id)

    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES, instance=video)
        if form.is_valid():
            form.save()
            return redirect('videos')
    else:
        form = VideoForm(instance=video)

    return render(request, 'update_video.html', {'form': form})


def generate_safe_filename(filename):
    # Get the file extension
    _, ext = os.path.splitext(filename)

    # Generate a unique filename using a slugified version of the title
    title_slug = slugify(Video.title)
    unique_filename = f"{title_slug}{ext}"

    # Return the path to store the file in the 'videos' directory
    return os.path.join("videos", unique_filename)


def upload_view(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            # Capture the title field from the form
            title = form.cleaned_data['title']

            # Create a Video model instance and assign the title
            video = Video(title=title, video_file=request.FILES['video_file'])
            video.save()
            return redirect('videos')
    else:
        form = VideoForm()
    return render(request, 'home.html', {'form': form})


def videos_view(request):
    videos = Video.objects.all()
    return render(request, 'videos.html', {'videos': videos})
