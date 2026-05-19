from pathlib import Path

TEMPLATE_PATH = (
    Path(__file__).resolve().parent.parent
    / "templates"
    / "triton_translation.txt"
)
class TritonPromptBuilder:
    def __init__(self):
        self.template = TEMPLATE_PATH.read_text()

    def build(self, instruction: str, input_text: str = ""):
        user_content = instruction

        if input_text:
            user_content += f"\n\n{input_text}"

        return [
            {
                "role": "system",
                "content": self.template,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]