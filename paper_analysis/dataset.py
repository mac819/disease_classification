import re
import typer
from tqdm import tqdm
from pathlib import Path

from paper_analysis.logger import logger
from paper_analysis.config import PROCESSED_DATA_DIR, RAW_DATA_DIR
logger.add("application.log", rotation="10 MB", retention="10 days", level="INFO")


from torch.utils.data import DataLoader, Dataset

class PaperDataset(Dataset):
    def __init__(self, data_dir: Path):

        self.pattern = r"<ID:(\d+)>\nTitle: (.*?)\nAbstract: (.*)"
        self.data_dir = data_dir
        # Load your data here
        # For example, you can read a CSV file or load images
        label_dirs = [x for x in Path(self.data_dir).glob("*") if x.is_dir()]
        logger.info(f"Found {len(label_dirs)} label directories.")
        # logger.info(f"Label directories: {label_dirs}")
        self.paper_file_records = []
        for label_dir in label_dirs:
            # logger.info(f"Label directory: {label_dir}")
            label = label_dir.name
            logger.info(f"Label: {label}")
            # Get all files in the label directory
            files = [{'file': x, 'label': label} for x in label_dir.glob("*.txt") if x.is_file()]
            self.paper_file_records.extend(files)

        _ = self.read_files()

    def read_files(self):
        # Read the files and store the content
        for record in self.paper_file_records:
            file = record['file']
            with open(file, 'r') as f:
                content = f.read()
                match = re.search(self.pattern, content, re.DOTALL)
                if match:
                    id = match.group(1)
                    title = match.group(2)
                    abstract = match.group(3)
                    logger.info(f"ID: {id}")
                    logger.info(f"Title: {title}")
                    logger.info(f"Abstract: {abstract}")
                else:
                    id = ""
                    title = ""
                    abstract = ""
                    logger.info("No match found.")
            record['id'] = id
            record['title'] = title
            record['abstract'] = abstract



    def __len__(self):
        # Return the number of samples in your dataset
        return len(self.paper_file_records)

    def __getitem__(self, idx):
        # Return a single sample from the dataset
        # You can use self.data_dir[idx] to access your data
        return self.paper_file_records[idx]



# app = typer.Typer()

# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     input_path: Path = RAW_DATA_DIR / "dataset.csv",
#     output_path: Path = PROCESSED_DATA_DIR / "dataset.csv",
#     # ----------------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Processing dataset...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Processing dataset complete.")
#     # -----------------------------------------


if __name__ == "__main__":
    # app()
    data_dir = Path(RAW_DATA_DIR) / 'Dataset'
    paper_dataset = PaperDataset(data_dir=data_dir)
