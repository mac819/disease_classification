from langchain import LlamaCpp, PromptTemplate
from langchain.output_parsers import OutputFixingParser
from langchain.output_parsers import PydanticOutputParser

from pathlib import Path
from paper_analysis.config import MODELS_DIR, PROCESSED_DATA_DIR
from paper_analysis.prompts import (
    DiseaseProcedureExtraction, # PyDantic Model
    Disease, # PyDanticModel
    template1, 
    template2
)
from paper_analysis.logger import logger


class DiseaseExtraction:

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.llm = LlamaCpp(
            model_path=str(self.model_path),
            n_gpu_layers=-1,
            max_tokens=500,
            n_ctx=2048,
            seed=42,
            verbose=False
        )
        self.parser = self.create_parsers(self.llm)
        self.prompt1 = PromptTemplate(
            template=template1,
            input_variables=['abstract'],
            partial_variables={'format_instructions': self.parser['parser1'].get_format_instructions()}
        )
        self.prompt2 = PromptTemplate(
            template=template2,
            input_variables=['disease'],
            partial_variables={'format_instructions': self.parser['parser2'].get_format_instructions()}
        )
        self.chain = self.build_chian(self.prompt1, self.prompt2)

    def create_parsers(self, model):
        parser1 = PydanticOutputParser(
            pydantic_object=DiseaseProcedureExtraction
        )
        fixed_parser1 = OutputFixingParser.from_llm(
            llm=model,
            parser=parser1,
            max_retries=2
        )

        parser2 = PydanticOutputParser(
            pydantic_object=Disease
        )
        fixed_parser2 = OutputFixingParser.from_llm(
            llm=model,
            parser=parser2,
            max_retries=2
        )

        return {
            'parser1': parser1,
            'fixed_parser1': fixed_parser1,
            'parser2': parser2,
            'fixedparser2': fixed_parser2
        }
    
    def build_chian(self, prompt1, prompt2):
        chain = (
            prompt1 
            | self.llm 
            | self.parser['parser1']
            | {'disease': lambda x: x.diseases} 
            | prompt2 
            | self.llm
            | self.parser['parser2']
        )
        return chain
    
    def predict(self, abstract: str):
        try:
            result = self.chain.invoke({'abstract': abstract})
            disease = result.disease
            is_carcinogenic = result.is_carcinogenic
            return disease, is_carcinogenic
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return None, None

















# app = typer.Typer()


# @app.command()
# def main(
#     # ---- REPLACE DEFAULT PATHS AS APPROPRIATE ----
#     features_path: Path = PROCESSED_DATA_DIR / "test_features.csv",
#     model_path: Path = MODELS_DIR / "model.pkl",
#     predictions_path: Path = PROCESSED_DATA_DIR / "test_predictions.csv",
#     # -----------------------------------------
# ):
#     # ---- REPLACE THIS WITH YOUR OWN CODE ----
#     logger.info("Performing inference for model...")
#     for i in tqdm(range(10), total=10):
#         if i == 5:
#             logger.info("Something happened for iteration 5.")
#     logger.success("Inference complete.")
#     # -----------------------------------------


# if __name__ == "__main__":
#     app()
