import argparse

from .mFormerV0 import *


def export_model(model_path):
    raise NotImplementedError("Exporting models is not yet supported.")
    # TODO: Refactor to load the appropriate model type, then export
    # # Load your Metaformer model from the .pth file
    # checkpoint = torch.load(model_path, map_location='cpu')
    # model = MetaFG_Meta(**checkpoint['config'])  # Adjust based on your model class name
    # model.load_state_dict(checkpoint['model'])

    # # Prepare dummy input for tracing
    # dummy_input = torch.randn(1, 3, 224, 224)  # Adjust based on your model's input shape
    # meta = torch.randn(1, 7)  # Adjust based on your model's meta input shape

    # # Use torch.export to export the model to ExportedProgram format
    # exported_model = torch.export(model, (dummy_input, meta), opset_version=11)

    # # Save the exported model to a .pt2 file
    # output_path = os.path.splitext(model_path)[0] + '.pt2'
    # torch.export.save(exported_model, output_path)
    # print(f"Exported model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export Metaformer model to .pt2 format"
    )
    parser.add_argument(
        "model_path", type=str, help="Absolute path to the .pth model file"
    )
    args = parser.parse_args()

    export_model(args.model_path)
