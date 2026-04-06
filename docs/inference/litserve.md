# LitServe Sketch

> Status: narrow integration sketch, not a production deployment guide.

If you already trust a bundle and want a thin HTTP wrapper around the local
handler, LitServe is one reasonable adapter.

```python
# server.py
import litserve as ls
from linnaeus.inference.handler import LinnaeusInferenceHandler

handler = LinnaeusInferenceHandler.load_from_artifacts(
    config_file_path="/abs/path/to/inference_bundle/inference_config.yaml"
)

app = ls.LitServer()
app.add_route("/predict", handler.predict, methods=["POST"])
app.add_route("/info", lambda: handler.info(), methods=["GET"])

if __name__ == "__main__":
    app.run()
```

The `/info` route exposes the metadata returned by `handler.info()`, which is
usually the easiest way for a client to discover expected inputs and taxonomy
details.

What this page does not claim:

- that LitServe is the blessed long-term serving stack
- that multi-view or observation-level inference is solved here
- that the repo ships a complete production deployment recipe
