function toggleInput() {

    let type = document.getElementById("type").value;

    document.getElementById("textInput").style.display = "none";
    document.getElementById("imageInput").style.display = "none";
    document.getElementById("audioInput").style.display = "none";

    if (type === "text") {
        document.getElementById("textInput").style.display = "block";
    }

    if (type === "image") {
        document.getElementById("imageInput").style.display = "block";
    }

    if (type === "audio") {
        document.getElementById("audioInput").style.display = "block";
    }
}


async function detect() {

    let type = document.getElementById("type").value;
    let resultBox = document.getElementById("result");
    resultBox.innerText = "Processing... ⏳";

    let formData = new FormData();
    formData.append("type", type);

    if (type === "text") {
        formData.append("text", document.getElementById("textInput").value);
    }

    if (type === "image") {
        formData.append("file", document.getElementById("imageInput").files[0]);
    }

    if (type === "audio") {
        formData.append("file", document.getElementById("audioInput").files[0]);
    }

    try {

        const response = await fetch("/predict", {
            method: "POST",
            body: formData
        });

        const data = await response.json();

        resultBox.innerText =
            "Result: " + data.prediction +
            " | Confidence: " + data.confidence + "%";

        document.getElementById("progress").style.width =
            data.confidence + "%";

    } catch (error) {

        resultBox.innerText = "Error ❌ Check Backend";
        console.log(error);

    }
}