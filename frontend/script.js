const API_URL = "";

const imageInput = document.getElementById("imageInput");
const fileName = document.getElementById("fileName");

const imagePreview = document.getElementById("imagePreview");
const previewContainer = document.getElementById("previewContainer");

const analyzeButton = document.getElementById("analyzeButton");

const loading = document.getElementById("loading");
const results = document.getElementById("results");

const severity = document.getElementById("severity");
const confidence = document.getElementById("confidence");

const probabilityList =
    document.getElementById("probabilityList");

const gradcamImage =
    document.getElementById("gradcamImage");

const recommendationList =
    document.getElementById("recommendationList");

const disclaimer =
    document.getElementById("disclaimer");

const printReportButton =
    document.getElementById("printReportButton");

const newAnalysisButton =
    document.getElementById("newAnalysisButton");
const classNames = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
];


/* =========================================
   IMAGE SELECTION
========================================= */

imageInput.addEventListener("change", function () {
    
    results.classList.add("hidden");
    gradcamImage.removeAttribute("src");

    console.log("IMAGE SELECTED");

    const file = this.files[0];

    if (!file) {

        fileName.textContent = "No image selected";

        previewContainer.classList.add("hidden");

        return;
    }

    fileName.textContent = file.name;

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;

    previewContainer.classList.remove("hidden");

});


/* =========================================
   ANALYZE IMAGE
========================================= */

analyzeButton.addEventListener("click", async function () {

    console.log("ANALYZE BUTTON CLICKED");

    const file = imageInput.files[0];

    if (!file) {

        alert("Please select a retinal image first.");

        return;
    }

    analyzeButton.disabled = true;
    console.log("STARTING ANALYSIS");
   

    /* Show loading */

    loading.classList.remove("hidden");

    results.classList.add("hidden");


    /* Create form data */

    const formData = new FormData();

    formData.append("file", file);


    try {

        console.log("SENDING REQUEST TO FASTAPI");


       const response = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    body: formData
});

const data = await response.json();

if (!response.ok) {
    throw new Error(
        typeof data.detail === "string"
            ? data.detail
            : "Image analysis failed. Please try again."
    );
}


        /* -------------------------------------
           DISPLAY SEVERITY
        ------------------------------------- */

        severity.textContent =
            data.severity;


        /* -------------------------------------
           DISPLAY CONFIDENCE
        ------------------------------------- */

        confidence.textContent =
            `${(data.confidence * 100).toFixed(2)}%`;


        /* -------------------------------------
           DISPLAY PROBABILITIES
        ------------------------------------- */

        probabilityList.innerHTML = "";


        data.probabilities.forEach(
            (probability, index) => {

                const percentage =
                    (probability * 100).toFixed(2);


                const item =
                    document.createElement("div");

                item.className =
                    "probability-item";


                item.innerHTML = `

                    <div class="probability-header">

                        <span>
                            ${classNames[index]}
                        </span>

                        <strong>
                            ${percentage}%
                        </strong>

                    </div>

                    <div class="progress-bar">

                        <div
                            class="progress-fill"
                            style="width: ${percentage}%"
                        ></div>

                    </div>

                `;


                probabilityList.appendChild(item);

            }
        );


        /* -------------------------------------
           DISPLAY GRAD-CAM
        ------------------------------------- */

       gradcamImage.src = data.gradcam_url;


        /* -------------------------------------
           DISPLAY RECOMMENDATIONS
        ------------------------------------- */

        recommendationList.innerHTML = "";


        data.recommendations.forEach(
            recommendation => {

                const item =
                    document.createElement("li");

                item.textContent =
                    recommendation;

                recommendationList.appendChild(item);

            }
        );


        /* -------------------------------------
           DISPLAY DISCLAIMER
        ------------------------------------- */

        disclaimer.textContent =
            data.disclaimer;


        /* -------------------------------------
           SHOW RESULTS
        ------------------------------------- */

        console.log("SHOWING RESULTS");


        results.classList.remove("hidden");

        loading.classList.add("hidden");


        console.log(
            "RESULTS CLASS:",
            results.className
        );


        console.log(
            "RESULTS DISPLAY:",
            window.getComputedStyle(results).display
        );


        console.log("RESULTS SHOULD NOW BE VISIBLE");


    } catch (error) {

        console.error(
            "FRONTEND ANALYSIS ERROR:",
            error
        );

         alert(error.message || "Image analysis failed.");


    } finally {
        

        loading.classList.add("hidden");
        analyzeButton.disabled = false;
        console.log("ANALYSIS FINISHED");

    }

});
/* =========================================================
   REPORT ACTIONS
   ========================================================= */
   printReportButton.addEventListener("click", function () {

    window.print();

});

newAnalysisButton.addEventListener("click", function () {

    imageInput.value = "";

    fileName.textContent = "No image selected";

    previewContainer.classList.add("hidden");

    results.classList.add("hidden");

    loading.classList.add("hidden");

    imagePreview.src = "";

    gradcamImage.src = "";

    severity.textContent = "-";

    confidence.textContent = "-";

    probabilityList.innerHTML = "";

    recommendationList.innerHTML = "";

    disclaimer.textContent = "";

    window.scrollTo({
        top: 0,
        behavior: "smooth"
    });

});
