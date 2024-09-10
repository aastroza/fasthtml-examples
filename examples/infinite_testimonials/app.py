from fasthtml.common import *
from pydantic import BaseModel
from modal import Cls
from textwrap import dedent
import random

# Example testimonial data
testimonials = [
    ("FastHTML is as intuitive as FastAPI, lends itself to clean architecture, and its HTML+HTMX structure makes it a good competitor to Django for building webapps. Most importantly, it's fun to use.",
        "Daniel Roy Greenfeld", "Co-author", "Two Scoops of Django"),
    ("Python has always been a wonderful tool for creating web applications; with FastHTML, it's even better!",
        "Giles Thomas", "Founder", "PythonAnywhere"),
    ("With FastHTML and Railway, Pythonistas can now have a real web application running in minutes, and can scale it all the way up to sophisticated production deployments.",
        "Jake Cooper", "CEO", "Railway.app"),
]

class Testimonial(BaseModel):
    quote: str
    name: str
    role: str
    company: str

    def __ft__(self):
        return Div(
            Blockquote(P(self.quote)),
            H4(self.name),
            P(f"{self.role}, {self.company}"),
        )
    
# Outlines and Modal stuff

Model = Cls.lookup("outlines-app", "Model")

m = Model()

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

# Generate a single testimonial card
def generate_testimonial_card() -> Testimonial:
    result = m.generate.remote(schema.strip(),  prompt_template.format(schema=schema.strip(), examples=examples))
    return Testimonial(**result)
    

# FastHTML app setup
app, rt = fast_app(hdrs=(picolink,))

@rt("/")
def get():
    return Titled('Infinite Testimonials',
                  Div(
                      Div(generate_testimonial_card(), id="testimonial-card"),
                      Div(
                          # Only Next button which fetches a random testimonial
                          Button("Next", id="next-btn", hx_get="/page", hx_target="#testimonial-card"),
                      )
                  ))

@rt("/page/")
def get():
    # Generate a new random testimonial
    return generate_testimonial_card()

serve()