import argparse
from hseb.core.report import Report
from hseb.core.submission import Submission
from structlog import get_logger

logger = get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to a submission file")
    parser.add_argument("--report", type=str, required=True, help="Name of report file to write")
    args = parser.parse_args()
    logger.info(f"Writing report: {args}")

    submission = Submission.from_json(args.input)
    report = Report.from_experiments(submission.experiments)
    report.df.to_json(args.report)
    logger.info(f"Report written to {args.report}")
