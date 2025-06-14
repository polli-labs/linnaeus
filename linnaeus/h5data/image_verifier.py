# linnaeus/h5data/image_verifier.py
import concurrent.futures
import json
import os
import time
from pathlib import Path

from tqdm import tqdm  # Import tqdm

# Get h5data logger specifically
from linnaeus.utils.logging.logger import get_h5data_logger

logger = get_h5data_logger()


class ImageVerifier:
    """Efficiently verifies image existence in parallel for hybrid datasets."""

    def __init__(
        self,
        images_dir: str,
        file_extension: str,
        num_workers: int = -1,
        chunk_size: int = 1000,
        logger_override=None,  # Allow passing a logger
    ):
        """Initialize the image verifier."""
        self.images_dir = Path(images_dir)
        # Ensure consistent dot prefix for extension
        self.file_extension = file_extension.strip()
        if self.file_extension and not self.file_extension.startswith("."):
            self.file_extension = f".{self.file_extension}"

        # Determine worker count
        self.num_workers = os.cpu_count() if num_workers <= 0 else num_workers
        self.chunk_size = max(1, chunk_size)  # Ensure chunk size is positive
        self.logger = logger_override or logger  # Use passed or default logger

        if not self.images_dir.is_dir():
            self.logger.warning(
                f"Images directory specified for verification does not exist: {self.images_dir}"
            )

        self.logger.info(
            f"ImageVerifier initialized: dir='{self.images_dir}', ext='{self.file_extension}', workers={self.num_workers}, chunk_size={self.chunk_size}"
        )

    def verify_images(self, img_identifiers: list[str]) -> tuple[set[int], set[str]]:
        """Verify if all images specified by identifiers exist on disk.

        Args:
            img_identifiers: List of image identifiers (filenames without extension, usually).

        Returns:
            Tuple of (missing_indices, missing_identifiers): Sets containing indices
            and identifiers of images that were not found.
        """
        start_time = time.time()
        total_to_check = len(img_identifiers)
        self.logger.info(
            f"Starting verification for {total_to_check:,} images using {self.num_workers} workers..."
        )

        missing_indices = set()
        missing_identifiers = set()

        # Don't proceed if the directory doesn't exist
        if not self.images_dir.is_dir():
            self.logger.error(
                f"Cannot verify images: Directory '{self.images_dir}' not found."
            )
            # Return empty sets, the error will likely be caught elsewhere or handled by ALLOW_MISSING
            return missing_indices, missing_identifiers

        # Prepare chunks
        indices = list(range(total_to_check))
        chunks_with_indices = [
            (img_identifiers[i : i + self.chunk_size], indices[i : i + self.chunk_size])
            for i in range(0, total_to_check, self.chunk_size)
        ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            futures = {
                executor.submit(self._check_chunk, chunk_ids, chunk_idxs): i
                for i, (chunk_ids, chunk_idxs) in enumerate(chunks_with_indices)
            }

            # Process results with tqdm progress bar
            progress_bar = tqdm(
                concurrent.futures.as_completed(futures),
                total=len(futures),
                desc="Verifying images",
                disable=None,
            )  # Use None to let tqdm decide based on context

            for future in progress_bar:
                chunk_missing_indices, chunk_missing_ids = future.result()
                missing_indices.update(chunk_missing_indices)
                missing_identifiers.update(chunk_missing_ids)

                # Update tqdm description dynamically if needed
                if len(missing_indices) > 0:
                    progress_bar.set_description(
                        f"Verifying images ({len(missing_indices)} missing)"
                    )

        elapsed = time.time() - start_time
        rate = total_to_check / elapsed if elapsed > 0 else 0
        self.logger.info(
            f"Image verification finished in {elapsed:.2f}s ({rate:.1f} images/sec). Found {len(missing_indices)} missing images."
        )

        return missing_indices, missing_identifiers

    def _check_chunk(
        self, chunk_ids: list[str], chunk_indices: list[int]
    ) -> tuple[set[int], set[str]]:
        """Checks a single chunk of image identifiers."""
        chunk_missing_indices = set()
        chunk_missing_ids = set()
        for i, img_id in enumerate(chunk_ids):
            # Ensure img_id is a string before path operations
            if not isinstance(img_id, str):
                self.logger.warning(
                    f"Skipping invalid image identifier (not a string): {img_id}"
                )
                continue  # Or handle appropriately, maybe add to missing?

            img_path = self._get_image_path(img_id)
            # Using os.path.exists as it's generally faster than Path.exists for many calls
            # lexists could be used if symlinks are a concern
            if not os.path.exists(img_path):
                original_index = chunk_indices[i]  # Get the original index
                chunk_missing_indices.add(original_index)
                chunk_missing_ids.add(img_id)
        return chunk_missing_indices, chunk_missing_ids

    def _get_image_path(self, img_id: str) -> Path:
        """Consistently construct the full image path."""
        # Ensure img_id is decoded string (not bytes)
        if isinstance(img_id, bytes):
            img_id_str = img_id.decode("utf-8", errors="replace")
        else:
            img_id_str = str(img_id)

        # Append extension only if necessary
        if self.file_extension and not img_id_str.lower().endswith(
            self.file_extension.lower()
        ):
            filename = f"{img_id_str}{self.file_extension}"
        else:
            filename = img_id_str
        return self.images_dir / filename

    def generate_report(
        self,
        missing_indices: set[int],
        missing_identifiers: set[str],
        total_count: int,
        report_path: str | None = None,
        log_missing: bool = True,
        log_limit: int = 50,
    ) -> dict:
        """Generates and optionally saves a report of missing images."""
        report = {
            "total_images_checked": total_count,
            "missing_count": len(missing_indices),
            "missing_ratio": len(missing_indices) / total_count
            if total_count > 0
            else 0.0,
            "images_dir": str(self.images_dir),
            "verification_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "missing_identifiers": sorted(list(missing_identifiers))
            if log_missing
            else f"Logging disabled ({len(missing_identifiers)} found)",
            "missing_indices": sorted(list(missing_indices))
            if log_missing
            else f"Logging disabled ({len(missing_indices)} found)",
        }

        if log_missing and missing_identifiers:
            self.logger.warning(f"Listing first {log_limit} missing image identifiers:")
            for i, identifier in enumerate(sorted(list(missing_identifiers))):
                if i >= log_limit:
                    self.logger.warning(
                        f"...and {len(missing_identifiers) - log_limit} more."
                    )
                    break
                # Ensure identifier is displayed as a string, not bytes
                if isinstance(identifier, bytes):
                    display_id = identifier.decode("utf-8", errors="replace")
                else:
                    display_id = identifier
                self.logger.warning(f"  - {display_id}")

        if report_path:
            try:
                report_path_obj = Path(report_path)
                report_path_obj.parent.mkdir(parents=True, exist_ok=True)
                with open(report_path_obj, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
                self.logger.info(f"Saved missing images report to: {report_path}")
            except Exception as e:
                self.logger.error(
                    f"Failed to save missing images report to {report_path}: {e}"
                )

        return report
