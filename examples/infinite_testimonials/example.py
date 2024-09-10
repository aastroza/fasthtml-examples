from modal import Cls
from textwrap import dedent
from pydantic import BaseModel

class Testimonial(BaseModel):
    quote: str
    name: str
    role: str
    company: str

Model = Cls.lookup("outlines-app", "Model")

m = Model(model_name="mistralai/Mistral-7B-Instruct-v0.2")

schema = """{
    "title": "Testimonial",
    "type": "object",
    "properties": {
        "quote": {
            "title": "Quote",
            "type": "string"
        },
        "name": {
            "title": "Name",
            "type": "string"
        },
        "role": {
            "title": "Role",
            "description": "The professional role of the individual giving the testimonial.",
            "type": "string"
        },
        "company": {
            "title": "Company",
            "description": "The name of the company where the individual works or which they represent.",
            "type": "string"
        }
    },
    "required": ["quote", "name", "role", "company"]
}
"""
examples = """[
    {
        "quote": "FastHTML is as intuitive as FastAPI, lends itself to clean architecture, and its HTML+HTMX structure makes it a good competitor to Django for building webapps. Most importantly, it's fun to use.",
        "name": "Daniel Roy Greenfeld",
        "role": "Co-author",
        "company": "Two Scoops of Django"
    },
    {
        "quote": "Python has always been a wonderful tool for creating web applications; with FastHTML, it's even better!",
        "name": "Giles Thomas",
        "role": "Founder",
        "company": "PythonAnywhere"
    },
    {
        "quote": "With FastHTML and Railway, Pythonistas can now have a real web application running in minutes, and can scale it all the way up to sophisticated production deployments.",
        "name": "Jake Cooper",
        "role": "CEO",
        "company": "Railway.app"
    }
]
"""
prompt_template = dedent(
                                    """\
                                <s>[INST]
                                You are a world class marketing copywriter. You must write new testimonials for a website by replying VALID JSON that matches the schema below.
                                
                                ```json
                                {schema}
                                ```
                                
                                ---
                                
                                These are examples of testimonials that you can use as inspiration:
                                
                                ```text
                                {examples}
                                ```
                                
                                Be creative and write new testimonials that are unique and engaging.
                                [/INST]
                                """)

result = m.generate.remote(schema.strip(),  prompt_template.format(schema=schema.strip(), examples=examples))
parsed_testimonial = Testimonial(**result)
print(parsed_testimonial)