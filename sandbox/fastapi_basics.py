from fastapi import FastAPI

# Initialize the "Waiter"
app = FastAPI()

# Define the endpoint (menu item)
# When the user goes to the home page ("/") 

@app.get("/")
def read_root():
    # The kitchen returns this JSON
    return {"message": "Welcome to my Stock Predictor API"}