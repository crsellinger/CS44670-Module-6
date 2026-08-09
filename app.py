# ============================================================
# app.py (for Hugging Face Space)
# ============================================================
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
import joblib
import pandas as pd
from pydantic import BaseModel

app = FastAPI(title="Fabricator Type Classifier")

# CORS for webpage
app.add_middleware(
    CORSMiddleware,
    allow_origins=[""],
    allow_credentials=True,
    allow_methods=[""],
    allow_headers=["*"],
)

# Load model
MODEL_PATH = Path(__file__).parent / "artifacts" / "csellinger_model.joblib"
model = joblib.load(MODEL_PATH)


class FabricatorTypeClassificationRequest(BaseModel):
    monthly_wafer_capacity: float
    process_node_nm: float


# Webpage for Hugging Face Space

# @app.get("/")
# def root():
#     return {"status": "ok", "message": "Fabricator Type Classifier API is running."}


@app.get("/", response_class=HTMLResponse)
def root():

    return """

<!DOCTYPE html>

<html lang="en">

<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

```
<title>Fabricator Type Classifier</title>

<style>

    body {
        font-family: Arial, sans-serif;
        background-color: #f4f6f8;
        display: flex;
        justify-content: center;
        align-items: center;
        min-height: 100vh;
    }

    .container {
        background: white;
        padding: 30px;
        border-radius: 12px;
        width: 400px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    h1 {
        text-align: center;
    }

    label {
        display: block;
        margin-top: 20px;
        margin-bottom: 5px;
        font-weight: bold;
    }

    input {
        width: 100%;
        box-sizing: border-box;
        padding: 12px;
        font-size: 16px;
        border: 1px solid #ccc;
        border-radius: 6px;
    }

    button {
        width: 100%;
        margin-top: 25px;
        padding: 12px;
        background-color: #2563eb;
        color: white;
        border: none;
        border-radius: 6px;
        font-size: 16px;
        cursor: pointer;
    }

    button:hover {
        background-color: #1d4ed8;
    }

    #result {
        display: none;
        margin-top: 25px;
        padding: 20px;
        background-color: #ecfdf5;
        border: 1px solid #10b981;
        border-radius: 8px;
        text-align: center;
    }

    #prediction {
        font-size: 24px;
        font-weight: bold;
        color: #047857;
        margin-top: 10px;
    }

    #error {
        display: none;
        margin-top: 20px;
        padding: 15px;
        background-color: #fef2f2;
        color: #b91c1c;
        border-radius: 8px;
    }

</style>
```

</head>

<body>

<div class="container">

```
<h1>Fabricator Type Classifier</h1>


<label for="capacity">
    Monthly Wafer Capacity
</label>

<input
    type="number"
    id="capacity"
    placeholder="Example: 50000"
    step="any"
>


<label for="processNode">
    Process Node (nm)
</label>

<input
    type="number"
    id="processNode"
    placeholder="Example: 7"
    step="any"
>


<button onclick="predict()">
    Predict
</button>


<!-- OUTPUT BOX -->

<div id="result">

    <div>Predicted Fabricator Type:</div>

    <div id="prediction"></div>

</div>


<!-- ERROR BOX -->

<div id="error"></div>
```

</div>

<script>

async function predict() {

    const capacity =
        document.getElementById("capacity").value;

    const processNode =
        document.getElementById("processNode").value;

    const resultBox =
        document.getElementById("result");

    const predictionBox =
        document.getElementById("prediction");

    const errorBox =
        document.getElementById("error");


    // Clear previous messages

    resultBox.style.display = "none";
    errorBox.style.display = "none";


    // Check that both inputs were entered

    if (capacity === "" || processNode === "") {

        errorBox.textContent =
            "Please enter both values.";

        errorBox.style.display = "block";

        return;
    }


    try {

        // Send input to FastAPI

        const response = await fetch(
            "/predict",
            {
                method: "POST",

                headers: {
                    "Content-Type": "application/json"
                },

                body: JSON.stringify({

                    monthly_wafer_capacity:
                        Number(capacity),

                    process_node_nm:
                        Number(processNode)

                })
            }
        );


        // Check for API errors

        if (!response.ok) {

            throw new Error(
                "API returned status " + response.status
            );

        }


        // Read prediction returned by FastAPI

        const data = await response.json();


        // PUT THE PREDICTION INTO THE OUTPUT BOX

        predictionBox.textContent =
            data.Fab_type;


        // Show the output box

        resultBox.style.display = "block";


    } catch (error) {

        errorBox.textContent =
            "Error: " + error.message;

        errorBox.style.display = "block";

    }

}

</script>

</body>

</html>
 """


@app.post("/predict")
def predict(request: FabricatorTypeClassificationRequest):
    X = pd.DataFrame(
        [
            {
                "monthly_wafer_capacity": request.monthly_wafer_capacity,
                "process_node_nm": request.process_node_nm,
            }
        ]
    )
    # Perform prediction using the loaded model
    prediction = model.predict(X)
    return {"Fab_type": prediction[0]}
