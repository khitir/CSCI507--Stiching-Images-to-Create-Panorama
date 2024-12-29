import cv2
import numpy as np

fileNames = [
    'mural01.jpg', 'mural02.jpg', 'mural03.jpg',
    'mural04.jpg', 'mural05.jpg', 'mural06.jpg',
    'mural07.jpg', 'mural08.jpg', 'mural09.jpg',
    'mural10.jpg', 'mural11.jpg', 'mural12.jpg',
]

# Set the size of the output mosaic image, so that the entire mural fits in the image
# (trial and error).
MOSAIC_WIDTH = 7000
MOSAIC_HEIGHT = 1300

def main():
    print("Stitch together images of a planar object")

    # Feature detector.
    detector = cv2.ORB_create(nfeatures=2000,  # default = 500
                              nlevels=8,  # default = 8
                              firstLevel=0,  # default = 0
                              patchSize=31,  # default = 31
                              edgeThreshold=31)  # default = 31

    # Matcher object.
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    # Get the first image..
    bgr_first= cv2.imread(fileNames[0])

    # # Ask the user to click on the four corners of the leftmost mural.
    # print("Click on four corners of the mural, in clockwise order,")
    # print("starting from the top left.")
    # corners = []
    # cv2.namedWindow("First image", cv2.WINDOW_NORMAL)
    # cv2.setMouseCallback("First image", on_mouse=get_xy, param=corners)
    # cv2.imshow("First image", bgr_first)
    # while len(corners) < 4:
    #     cv2.waitKey(30)
    corners = [(156, 123), (377, 73), (299, 495), (32, 475)]   # hardcode for development

    print("Ok, here are the corner locations:", corners)

    # Define the desired corner locations in an orthophoto version of this image.
    # The actual size of the mural was measured by hand.
    leftmost_mural_width_meters = 1.593
    leftmost_mural_height_meters = 2.178

    # Set the desired scale of the output orthophoto.
    pixel_size_cm = 0.5
    pixel_size_m = pixel_size_cm/100
    pixels_per_meter = 1/pixel_size_m

    # Set the size of the leftmost mural, in the output orthophoto.
    ortho_width = int(round(leftmost_mural_width_meters * pixels_per_meter))
    ortho_height = int(round(leftmost_mural_height_meters * pixels_per_meter))

    # This is the offset of the first panel, in the output image.
    offset_x_pixels = 200
    offset_y_pixels = 400

    ortho_corners = np.array([
        [0 + offset_x_pixels, 0 + offset_y_pixels],
        [ortho_width + offset_x_pixels, 0 + offset_y_pixels],
        [ortho_width + offset_x_pixels, ortho_height + offset_y_pixels],
        [0 + offset_x_pixels, ortho_height + offset_y_pixels]
    ])

    # Find a homography that maps the input image onto the orthophoto.
    src_pts = np.array(corners)
    H, _ = cv2.findHomography(srcPoints=src_pts, dstPoints=ortho_corners)

    # Warp the input image to the orthophoto image.  This is the start of the mosaic.
    bgr_mosaic = cv2.warpPerspective(bgr_first, H, (MOSAIC_WIDTH, MOSAIC_HEIGHT))

    cv2.namedWindow("Mosaic", cv2.WINDOW_NORMAL)
    cv2.imshow("Mosaic", bgr_mosaic)
    cv2.waitKey(10)

    # Ok, process all other images.  Each new image is registered to the previous image.
    bgr_previous = bgr_first
    H_prev_mosaic = H   # Remember the initial homography
    for image_count in range(1, len(fileNames)):
        bgr_current = cv2.imread(fileNames[image_count])

        # Extract features from the previous and current images.
        gray_previous = cv2.cvtColor(bgr_previous, cv2.COLOR_BGR2GRAY)
        kp_prev, des_prev = detector.detectAndCompute(gray_previous, None)
        gray_current = cv2.cvtColor(bgr_current, cv2.COLOR_BGR2GRAY)
        kp_curr, des_curr = detector.detectAndCompute(gray_current, None)

        # Match descriptors.
        matches = matcher.knnMatch(des_curr,des_prev, k=2)

        # Apply ratio test.
        good_matches = []
        for m,n in matches:
            if m.distance < 0.8*n.distance:
                good_matches.append(m)

        # Find a homography that maps the current image to the previous image.
        src_pts = np.float32([kp_curr[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        dst_pts = np.float32([kp_prev[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

        H_current_prev, mask = cv2.findHomography(
            srcPoints=src_pts, dstPoints=dst_pts, method=cv2.RANSAC,
            ransacReprojThreshold=3.0,  # default is 3.0
            maxIters=2000  # default is 2000
        )
        num_inliers = sum(mask)  # mask[i] is 1 if point i is an inlier, else 0
        print("Number of inliers: %d" % num_inliers)

        # Optionally display all matches.
        matchesMask = mask.ravel().tolist()
        bgr_matches = cv2.drawMatches(
            img1=gray_current, keypoints1=kp_curr,
            img2=gray_previous, keypoints2=kp_prev,
            matches1to2=good_matches, matchesMask=matchesMask, outImg=None)
        cv2.imshow("matches", bgr_matches)
        cv2.waitKey(10)

        # Combine homographies to get the transform from current to mosaic.
        H_current_mosaic = H_prev_mosaic @ H_current_prev

        # Warp current image to align it with reference image.
        w = bgr_mosaic.shape[1]
        h = bgr_mosaic.shape[0]
        bgr_current_warp = cv2.warpPerspective(bgr_current, H_current_mosaic, (w, h))

        # Fuse the two images.
        bgr_mosaic = fuse_color_images(bgr_mosaic, bgr_current_warp)

        cv2.imshow("Mosaic", bgr_mosaic)
        cv2.waitKey(0)

        bgr_previous = bgr_current
        H_prev_mosaic = H_current_mosaic

    # Write out final mosaic image.
    cv2.imwrite("mosaic.jpg", bgr_mosaic)

    print("All done, bye!")


# Fuse two color images.  Assume that zero indicates an unknown value.
# At pixels where both values are known, the output is the average of the two.
# At pixels where only one is known, the output uses that value.
def fuse_color_images(A, B):
    assert(A.ndim == 3 and B.ndim == 3)
    assert(A.shape == B.shape)

    # Allocate result image.
    C = np.zeros(A.shape, dtype=np.uint8)

    # Create masks for pixels that are not zero.
    A_mask = np.sum(A, axis=2) > 0
    B_mask = np.sum(B, axis=2) > 0

    # Compute regions of overlap.
    A_only = A_mask & ~B_mask
    B_only = B_mask & ~A_mask
    A_and_B = A_mask & B_mask

    C[A_only] = A[A_only]
    C[B_only] = B[B_only]
    C[A_and_B] = 0.5 * A[A_and_B] + 0.5 * B[A_and_B]

    return C

# Mouse callback function.  This will store the x,y locations of mouse clicks.
def get_xy(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        param.append((x, y))

if __name__ == "__main__":
    main()
