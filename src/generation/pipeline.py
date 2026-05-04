from src.generation.two_step_generation import TwoStepGenerator
from src.generation.llm import LocalLLM


class GenerationPipeline:

    def __init__(self, config):
        self.config = config

        if config["generation_mode"] == "two_step":
            self.generator = TwoStepGenerator(
                llm=LocalLLM(config["model_name"]),
                config=config
            )

    def run(self, query, retrieved_chunks):
        return self.generator.generate(query, retrieved_chunks)