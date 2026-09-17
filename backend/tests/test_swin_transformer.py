import torch

from app.models.mri.swin_transformer import (
    DEVICE,
    create_swin_model,
    MRI_CLASSES,
)


def test_swin_model_creation():
    model = create_swin_model(
        num_classes=len(MRI_CLASSES),
        pretrained=False,
    )

    assert model is not None
    assert next(model.parameters()).device == DEVICE


def test_swin_model_output_shape():
    model = create_swin_model(
        num_classes=len(MRI_CLASSES),
        pretrained=False,
    )

    model.eval()

    sample = torch.randn(
        1,
        3,
        224,
        224,
        device=DEVICE,
    )

    with torch.no_grad():
        output = model(sample)

    assert output.shape == (1, 4)