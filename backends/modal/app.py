import modal

from backends.modal.image import image

app = modal.App(
    "triton-grammar-constrains",
    image=image,
)

volume = modal.Volume.from_name(
    "triton-grammar-constrains-volume",
    create_if_missing=True,
)