from app.models.parkinsons_inference import (
    train_and_save_xgboost_model,
    load_xgboost_model,
    predict_clinical_risk,
)
from app.preprocessing.parkinsons_pipeline import (
    prepare_parkinsons_data,
)


if __name__ == "__main__":
    print("Training and saving clinical model...")

    training_result = train_and_save_xgboost_model()

    print(
        f"Model saved at: "
        f"{training_result['model_path']}"
    )

    bundle = load_xgboost_model(
        training_result["model_path"]
    )

    dataset = prepare_parkinsons_data()

    raw_dataset = __import__("pandas").read_csv(
        "datasets/raw/clinical/Parkinsson disease.csv"
    )

    sample_features = (
        raw_dataset.drop(
            columns=["name", "status"]
        )
        .iloc[0]
        .to_dict()
    )

    result = predict_clinical_risk(
        sample_features,
        bundle,
    )

    print("\nClinical prediction:")
    print(result)