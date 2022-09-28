import os
import cv2

image_folder = '/mnt/disk2_hdd/mengya/surgical_blender/simulation_dataset_3DTool/video15/image_demo'
video_name = 'video15_demo_blood.mp4'

# images = ['seq_15_frame004.png','seq_15_frame006.png','seq_15_frame007.png','seq_15_frame008.png','seq_15_frame035.png','seq_15_frame037.png','seq_15_frame038.png','seq_15_frame044.png','seq_15_frame048.png','seq_15_frame049.png','seq_15_frame051.png','seq_15_frame070.png','seq_15_frame129.png','seq_15_frame130.png','seq_15_frame132.png','seq_15_frame133.png','seq_15_frame135.png','seq_15_frame136.png','seq_2_frame041.png','seq_2_frame043.png','seq_2_frame044.png','seq_2_frame064.png','seq_5_frame070.png','seq_5_frame121.png','seq_5_frame125.png','seq_9_frame001.png','seq_9_frame003.png','seq_9_frame004.png','seq_9_frame005.png','seq_9_frame008.png','seq_9_frame009.png','seq_9_frame011.png','seq_9_frame012.png','seq_9_frame022.png','seq_9_frame052.png','seq_9_frame086.png','seq_9_frame089.png','seq_9_frame093.png','seq_9_frame094.png','seq_9_frame098.png','seq_9_frame099.png','seq_9_frame100.png','seq_9_frame101.png','seq_9_frame110.png','seq_9_frame111.png','seq_15_frame004.png','seq_15_frame006.png','seq_15_frame007.png','seq_15_frame008.png','seq_15_frame035.png','seq_15_frame037.png','seq_15_frame038.png','seq_15_frame044.png','seq_15_frame048.png','seq_15_frame049.png','seq_15_frame051.png','seq_15_frame070.png','seq_15_frame129.png','seq_15_frame130.png','seq_15_frame132.png','seq_15_frame133.png','seq_15_frame135.png','seq_15_frame136.png','seq_2_frame041.png','seq_2_frame043.png','seq_2_frame044.png','seq_2_frame064.png','seq_5_frame070.png','seq_5_frame121.png','seq_5_frame125.png','seq_9_frame001.png','seq_9_frame003.png','seq_9_frame004.png','seq_9_frame005.png','seq_9_frame008.png','seq_9_frame009.png','seq_9_frame011.png','seq_9_frame012.png','seq_9_frame022.png','seq_9_frame052.png','seq_9_frame086.png','seq_9_frame089.png','seq_9_frame093.png','seq_9_frame094.png','seq_9_frame098.png','seq_9_frame099.png','seq_9_frame100.png','seq_9_frame101.png','seq_9_frame110.png','seq_9_frame111.png']


images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

# cv2.destroyAllWindows()
video.release()
