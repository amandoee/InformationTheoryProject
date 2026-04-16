from construct import Struct, Int32ub, PascalString, PrefixedArray, Bytes





class Encoder:
    """
    Encodes semantically important subimages(Snippets) S_i in jpeg format
    We also encode relevant metadata for lossy image reconstruction of background
    The encoding format will be something like this:

    height     | width | Number of snippets |
    SizeOf S_0 | S_0   | bbox_0 | SizeOf S_1 | S_1 | bbox_1
    |             ...           | SizeOf S_N | S_N | bbox_N
    """
    def __init__(self, entries, imageDimensions):
        """
        entries: List of tuples [(jpeg encoding), (x, y, width, height)]
        """
        self.entries = entries
        self.height, self.width = imageDimensions
