from pathlib import Path

import pandas as pd


class DataLoadingError(Exception):
    """Raised when a dataset cannot be loaded."""


def load_clinical_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load a clinical CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame containing the clinical data.

    Raises:
        DataLoadingError: If the file does not exist or cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise DataLoadingError(f"Dataset not found: {path}")

    if not path.is_file():
        raise DataLoadingError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".csv":
        raise DataLoadingError("Expected a CSV file.")

    try:
        data = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataLoadingError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise DataLoadingError(
            f"Could not parse CSV file: {path}"
        ) from exc
    except OSError as exc:
        raise DataLoadingError(
            f"Could not read file: {path}"
        ) from exc

    if data.empty:
        raise DataLoadingError("The CSV contains no data rows.")

    return data