from semtrace.data.manifest import ManifestImageDataset


class ForenSynthsDataset(ManifestImageDataset):
    """Manifest-backed ForenSynths adapter with real=0 and fake=1 labels."""
