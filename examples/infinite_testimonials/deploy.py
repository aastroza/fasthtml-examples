import os
from modal import App, Secret, gpu, method, Image

app = App(name="outlines-app")

image = Image.debian_slim(python_version="3.11").pip_install(
    "outlines==0.0.46",
    "transformers",
    "datasets",
    "accelerate",
    "fastapi",
)

@app.cls(image=image, secrets=[Secret.from_dotenv()], gpu=gpu.H100(), timeout=300)
class Model:
    def __init__(self, model_name: str = "microsoft/Phi-3.5-mini-instruct") -> None:
        import outlines

        self.model = outlines.models.transformers(
            model_name,
            device="cuda",
            model_kwargs={
                "token": os.environ["HF_TOKEN"],
                "trust_remote_code": True,
            },
        )

    @method()
    async def generate(self, json_schema: str, prompt: str, whitespace_pattern: str = None):
        import outlines

        if whitespace_pattern:
            generator = outlines.generate.json(self.model, json_schema.strip(), whitespace_pattern=whitespace_pattern)
        else:
            generator = outlines.generate.json(self.model, json_schema.strip())

        result = generator(prompt)

        return result