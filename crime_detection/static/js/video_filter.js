document.getElementById("videoSearch").addEventListener("input", function () {
  var filter = this.value.toLowerCase();
  var videos = document.querySelectorAll(".video-box");

  videos.forEach(function (video) {
    var name = video.querySelector("p strong").textContent.toLowerCase();
    if (name.includes(filter)) {
      video.style.display = "block";
    } else {
      video.style.display = "none";
    }
  });
});
