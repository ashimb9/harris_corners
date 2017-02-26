# harris_corners
The script evaluates Harris-Stephens corner points for a give RGB or grayscale image

@:param img is the image whose corners are to be located
@:param w is the length (=width) of the patch that will be convolved with gradient product matrices
@:param thres is the threshold value (i.e. Harris value) for a potential corner point
@:param killzone is the window length(=width) to be used during non-local-maxima suppression
@:param count is the number of corner points to be returned (might be lower if #positive Harris-valued points < count)
@:returns a tuple whose elements are the row and column indices of the Harris-Stephens corner
