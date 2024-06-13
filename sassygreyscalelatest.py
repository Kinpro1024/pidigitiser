import cv2
import numpy as np
import os

def get_desktop_path():
    return os.path.join(os.path.join(os.environ['USERPROFILE']), 'Desktop')

def resize_image(image, scale_percent=70):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

def stitch_images(image1_path, image2_path, image3_path):
    # Load images in grayscale
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    image3 = cv2.imread(image3_path, cv2.IMREAD_GRAYSCALE)

    if image1 is None or image2 is None or image3 is None:
        print("Error: One or more images could not be loaded.")
        return None

    # Resize images to 50% of their original size
    image1 = resize_image(image1, 70)
    image2 = resize_image(image2, 70)
    image3 = resize_image(image3, 70)

    # Detect keypoints and descriptors
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
    keypoints3, descriptors3 = sift.detectAndCompute(image3, None)

    # Match descriptors
    matcher = cv2.BFMatcher()
    matches1_2 = matcher.knnMatch(descriptors1, descriptors2, k=2)
    matches2_3 = matcher.knnMatch(descriptors2, descriptors3, k=2)

    # Apply ratio test to select good matches
    good_matches1_2 = [m for m, n in matches1_2 if m.distance < 0.75 * n.distance]
    good_matches2_3 = [m for m, n in matches2_3 if m.distance < 0.75 * n.distance]

    # Homography estimation
    if len(good_matches1_2) > 4 and len(good_matches2_3) > 4:
        src_points1_2 = np.float32([keypoints1[m.queryIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
        dst_points1_2 = np.float32([keypoints2[m.trainIdx].pt for m in good_matches1_2]).reshape(-1, 1, 2)
        src_points2_3 = np.float32([keypoints2[m.queryIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)
        dst_points2_3 = np.float32([keypoints3[m.trainIdx].pt for m in good_matches2_3]).reshape(-1, 1, 2)

        M1_2, _ = cv2.findHomography(src_points1_2, dst_points1_2, cv2.RANSAC, 5.0)
        M2_3, _ = cv2.findHomography(src_points2_3, dst_points2_3, cv2.RANSAC, 5.0)

        # Create panorama images with sufficient size to hold the stitched images
        panorama1_2 = cv2.warpPerspective(image1, M1_2, (image1.shape[1] + image2.shape[1], image1.shape[0]))
        panorama1_2[0:image2.shape[0], 0:image2.shape[1]] = image2

        panorama_final = cv2.warpPerspective(panorama1_2, M2_3, (panorama1_2.shape[1] + image3.shape[1], panorama1_2.shape[0]))
        panorama_final[0:image3.shape[0], 0:image3.shape[1]] = image3

        # Release intermediate variables to free memory
        del image1, image2, image3
        del keypoints1, descriptors1, keypoints2, descriptors2, keypoints3, descriptors3
        del matches1_2, matches2_3, good_matches1_2, good_matches2_3
        del src_points1_2, dst_points1_2, src_points2_3, dst_points2_3

        return panorama_final
    else:
        print("Not enough matches are found - %d/%d or %d/%d" % (len(good_matches1_2), 4, len(good_matches2_3), 4))
        return None

def invert_colors(image):
    return cv2.bitwise_not(image)

# Paths to the input images
image1_path = 'image1.jpeg'
image2_path = 'image2.jpeg'
image3_path = 'image3.jpeg'

# Stitch images
panorama = stitch_images(image1_path, image2_path, image3_path)

if panorama is not None:
    # Invert the colors of the stitched image
    inverted_panorama = invert_colors(panorama)

    # Save the inverted stitched image to the desktop
    desktop_path = get_desktop_path()
    output_path = os.path.join(desktop_path, 'inverted_panorama.jpg')
    cv2.imwrite(output_path, inverted_panorama)
    print("Inverted Panorama saved to:", output_path)

    # Display the inverted stitched image
    cv2.imshow('Inverted Panorama', inverted_panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Panorama could not be created.")
