# Serving with LitServe

LitServe provides a lightweight way to expose the `LinnaeusInferenceHandler` (which can be loaded from a local inference bundle or a Hugging Face model identifier) as a REST API.  After installing `litserve`, create a small server script that loads the handler and registers its `predict` method.

```python
# server.py
import litserve as ls
from linnaeus.inference.handler import LinnaeusInferenceHandler

# Option 1: Load from a local inference bundle
# handler = LinnaeusInferenceHandler.load_from_artifacts(
#     config_file_path="path/to/your_model_bundle/inference_config.yaml"
# )

# Option 2: Load from Hugging Face Hub (conceptual - adapt to actual API)
# This assumes the handler can be loaded directly or you have a helper function.
# See 'running_inference_with_pretrained_models.md' for more detailed loading.
# For example, if LinnaeusInferenceHandler has a method like 'load_from_hf_hub':
# handler = LinnaeusInferenceHandler.load_from_hf_hub(
#     hf_model_id="polli-caleb/linnaeus-aves-mformerV1_sm-v1"
# )
# For this example, we'll assume the handler is already loaded, e.g., from a local bundle:
handler = LinnaeusInferenceHandler.load_from_artifacts(
     config_file_path="path/to/your_model_bundle/inference_config.yaml" # Replace this
)
# Ensure 'handler' is correctly initialized using one of the methods above
# before passing to LitServer.

if handler is None:
    raise ValueError("LinnaeusInferenceHandler could not be loaded. Please check configuration.")

app = ls.LitServer()
app.add_route("/predict", handler.predict, methods=["POST"])
app.add_route("/info", lambda: handler.info(), methods=["GET"])

if __name__ == "__main__":
    app.run()
```

By default LitServe will batch concurrent requests and place the model on the best available device (CPU, CUDA or MPS).  The `/info` route returns the metadata produced by `handler.info()` so clients can discover the model’s expected inputs and taxonomy details.
