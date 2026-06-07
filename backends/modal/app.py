import modal

from backends.modal.image import benchmark_image, production_image

benchmark_app = modal.App(
    "triton-grammar-constrains-benchmark",
    image=benchmark_image,
)

production_app = modal.App(
    "triton-grammar-constrains-production",
    image=production_image,
)

# For backwards compatibility
app = benchmark_app

volume = modal.Volume.from_name(
    "triton-grammar-constrains-volume",
    create_if_missing=True,
)
