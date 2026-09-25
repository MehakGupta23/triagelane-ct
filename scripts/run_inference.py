import argparse
import json
import logging
from pathlib import Path
import sys

# Allow imports from src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from triagelane_ct.pipeline import score_study_from_folder
from triagelane_ct.dicom_io import InvalidDicomSliceError


def main():
    parser = argparse.ArgumentParser(
        description="Run TriageLane CT inference on one DICOM study."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to a folder containing DICOM files.",
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Directory where result.json and run.log will be written.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = output_dir / "run.log"

    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("Starting TriageLane CT inference")
    logging.info("Input: %s", input_dir)

    try:
        if not input_dir.exists():
            raise FileNotFoundError(
                f"Input folder does not exist: {input_dir}"
            )

        result = score_study_from_folder(
            str(input_dir)
        )

        result_file = output_dir / "result.json"

        with open(
            result_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                result,
                f,
                indent=2,
            )

        logging.info(
            "Inference completed successfully"
        )
        logging.info(
            "Result written to: %s",
            result_file,
        )

        print(json.dumps(result, indent=2))

    except InvalidDicomSliceError as e:
        logging.exception(
            "Invalid DICOM slice: %s",
            e,
        )
        print(f"ERROR: Invalid DICOM slice: {e}")
        raise SystemExit(1)

    except Exception as e:
        logging.exception(
            "Inference failed: %s",
            e,
        )
        print(f"ERROR: Inference failed: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()