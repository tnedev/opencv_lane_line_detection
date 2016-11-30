from process_image import process_image
from PIL import Image
from moviepy.editor import VideoFileClip

output = 'out/solidYellowLeftOut.mp4'
clip1 = VideoFileClip("solidYellowLeft.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)

output = 'out/solidWhiteRightOut.mp4'
clip1 = VideoFileClip("solidWhiteRight.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)

output = 'out/challengeOut.mp4'
clip1 = VideoFileClip("challenge.mp4")
clip = clip1.fl_image(process_image)
clip.write_videofile(output, audio=False)

# import matplotlib.image as mpimg
#
# image = mpimg.imread('test_images/solidWhiteCurve.jpg')
# output = process_image(image)
# Image.fromarray(output).save('out/out.jpg')
