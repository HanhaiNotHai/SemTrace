from semtrace.data.manifest import ManifestImageDataset


class GenImageDataset(ManifestImageDataset):
    """Manifest-backed GenImage adapter with real=0 and fake=1 labels."""
