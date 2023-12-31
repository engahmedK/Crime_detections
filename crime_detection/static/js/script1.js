function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();

    reader.onload = function (e) {
      $(".image-upload-wrap").hide();
      $(".file-upload-video source").attr("src", e.target.result);
      $(".file-upload-video")[0].load();
      $(".file-upload-content").show();
      $(".image-title").html(input.files[0].name);
    };

    reader.readAsDataURL(input.files[0]);
  } else {
    removeUpload();
  }
}

function removeUpload() {
  $(".file-upload-input").val("");
  $(".file-upload-content").hide();
  $(".image-upload-wrap").show();
}

function submitUpload() {
  $(".file-upload-input").click();
}

$(".image-upload-wrap").bind("dragover", function () {
  $(".image-upload-wrap").addClass("image-dropping");
});

$(".image-upload-wrap").bind("dragleave", function () {
  $(".image-upload-wrap").removeClass("image-dropping");
});
