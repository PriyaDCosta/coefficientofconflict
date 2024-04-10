
import sys
from predictor import PredictionBuilder

# Main Function
if __name__ == "__main__":
    # Check if enough command line arguments are provided
    if len(sys.argv) < 3:
        print("Usage: python3 predict.py <data_source> <dimensions>  <regularization> <stage1_model> <stage2_model>")
        sys.exit(1)

    # Extracting command line arguments
    data_source = sys.argv[1]
    dimensions = sys.argv[2]
    regularization = sys.argv[3]
    stage_1_model = sys.argv[4]
    stage_2_model = sys.argv[5]

    # Creating an instance of PredictionBuilder with command line arguments
    predictor = PredictionBuilder(
        data_source = data_source,
        dimensions = dimensions,
        regularization = regularization,
        stage_1_model = stage_1_model,
        stage_2_model = stage_2_model
    )

    # Calling the method to get predictions
    predictor.predict()
