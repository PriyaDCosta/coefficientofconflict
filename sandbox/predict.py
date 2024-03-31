
import sys
from predictor import PredictionBuilder

# Main Function
if __name__ == "__main__":
    # Check if enough command line arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py <model_1> <model_2>")
        sys.exit(1)

    # Extracting command line arguments
    model_1_arg = sys.argv[1]
    model_2_arg = sys.argv[2]

    # Creating an instance of PredictionBuilder with command line arguments
    predictor = PredictionBuilder(
        model_1=model_1_arg,
        model_2=model_2_arg
    )

    # Calling the method to get predictions
    predictor.predict()
