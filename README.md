# SuperKart Sales Prediction

Flask API (port 7860) + Streamlit UI (port 8501), run together with Docker Compose.

## Run (GitHub Codespace or any Docker host)

    docker compose up -d --build
    docker compose ps
    curl http://localhost:7860/health

- UI:  port 8501
- API: port 7860

## API

    curl -X POST http://localhost:7860/v1/predict -H "Content-Type: application/json" -d '{
      "Product_Weight": 12.66, "Product_Sugar_Content": "Low Sugar",
      "Product_Allocated_Area": 0.027, "Product_MRP": 117.08,
      "Store_Size": "Medium", "Store_Location_City_Type": "Tier 2",
      "Store_Type": "Supermarket Type2", "Product_Id_char": "FD",
      "Store_Age_Years": 16, "Product_Type_Category": "Non Perishables"}'

    curl -X POST http://localhost:7860/v1/predictbatch -F "file=@Batch_Data_SuperKart.csv"

## Stop

    docker compose down
