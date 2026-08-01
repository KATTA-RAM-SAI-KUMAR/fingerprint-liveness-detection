function previewImage(event){

    const preview=document.getElementById("preview");

    preview.src=URL.createObjectURL(event.target.files[0]);

    preview.style.display="block";

}

function removePreview(){

    const preview=document.getElementById("preview");

    preview.src="";

    preview.style.display="none";

}