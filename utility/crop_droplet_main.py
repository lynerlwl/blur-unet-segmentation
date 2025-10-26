from crop_droplet_utils import crop_image_pillow

left=400
upper=383
right=left+54
lower=upper+36
crop_image_pillow(image_path="data/spray/train/img-0.5/img_rgb_b_0.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path="output/crop_blur.png") 



i = "A"
i = "B"

'''  1 '''

left=262
upper=398
right=left+25
lower=upper+28
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/1.png")

left=320 
upper=490
right=left+40
lower=upper+40
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-1-{n}.png")

for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-1-{n}.png")
# output A
left=325
upper=493
right=left+30
lower=upper+35
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-1-out.png")
# output B
left=306
upper=462
right=left+27
lower=upper+33
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-1-out.png")

''' 2 '''

left=379
upper=404
right=left+32
lower=upper+41
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/2.png")

left=472
upper=500
right=left+38
lower=upper+50
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-2-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-2-{n}.png")
# output
left=470
upper=502
right=left+39
lower=upper+49
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-2-out.png")
left=441
upper=470
right=left+36
lower=upper+48
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-2-out.png")

''' 3 '''

left=488
upper=419
right=left+22
lower=upper+34
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/3.png")

left=605
upper=517
right=left+28
lower=upper+45
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-3-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-3-{n}.png")
# output
left=605
upper=518
right=left+26
lower=upper+44
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-3-out.png")
left=567
upper=487
right=left+25
lower=upper+39
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-3-out.png")

''' 4 '''

left=295
upper=217
right=left+22
lower=upper+23
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/4.png")
# mid
left=364
upper=268
right=left+30
lower=upper+30
for n in range(5, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-4-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-4-{n}.png")
# output
left=366
upper=270
right=left+27
lower=upper+25
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-4-out.png")
left=343
upper=252
right=left+25
lower=upper+24
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-4-out.png")

''' 5 '''

left=369
upper=232
right=left+21
lower=upper+14
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/5.png")
# mid
left=458
upper=286
right=left+27
lower=upper+19
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-5-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-5-{n}.png")
# output
left=458
upper=290
right=left+24
lower=upper+14
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-5-out.png")
left=430
upper=270
right=left+22
lower=upper+15
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-5-out.png")

''' 6 '''

left=582
upper=115
right=left+31
lower=upper+16
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/6.png")
# mid
left=722
upper=142
right=left+40
lower=upper+20
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-6-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-6-{n}.png")
# output
left=723
upper=144
right=left+37
lower=upper+18
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-6-out.png")
left=679
upper=135
right=left+34
lower=upper+16
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-6-out.png")

''' 7 '''

left=642
upper=398
right=left+24
lower=upper+20
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/7.png")
# mid
left=797
upper=492
right=left+30
lower=upper+26
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-7-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-7-{n}.png")
# output
left=797
upper=494
right=left+28
lower=upper+24
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-7-out.png")
left=747
upper=463
right=left+26
lower=upper+22
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-7-out.png")

''' 8 '''

left=433
upper=410
right=left+21
lower=upper+28
crop_image_pillow(image_path="data/spray/test/f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/8.png")
# mid
left=537
upper=508
right=left+27
lower=upper+35
for n in range(1, 7):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-8-{n}.png")
for n in range(7, 11):
    crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/hot/{n}.png", 
                      left=left, 
                      upper=upper, 
                      right=right, 
                      lower=lower, 
                      output_path=f"output/conv_feature_map/spray/{i}-8-{n}.png")
# output
left=538
upper=511
right=left+24
lower=upper+32
crop_image_pillow(image_path="output/predicted/A-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/A-8-out.png")
left=505
upper=478
right=left+22
lower=upper+30
crop_image_pillow(image_path="output/predicted/B-f_01340.png", 
                  left=left, 
                  upper=upper, 
                  right=right, 
                  lower=lower, 
                  output_path=f"output/conv_feature_map/spray/B-8-out.png")


''' spray - crop single droplet for edge quantification '''

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=418.5, 
                  upper=73, 
                  right=445.5, 
                  lower=102.5, 
                  output_path=f"output/conv_feature_map/spray/1-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=368, 
                  upper=651, 
                  right=392, 
                  lower=679, 
                  output_path=f"output/conv_feature_map/spray/2-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=521, 
                  upper=616, 
                  right=536, 
                  lower=631, 
                  output_path=f"output/conv_feature_map/spray/3-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=770, 
                  upper=619, 
                  right=792, 
                  lower=640, 
                  output_path=f"output/conv_feature_map/spray/4-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=700, 
                  upper=77, 
                  right=718, 
                  lower=94, 
                  output_path=f"output/conv_feature_map/spray/5-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/spray-proposed{i}/f_01340/gray/1.png", 
                  left=1064, 
                  upper=161, 
                  right=1080, 
                  lower=176, 
                  output_path=f"output/conv_feature_map/spray/6-{i}.png")

''' wood '''

i = "A"
i = "B"

crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=42, 
                  upper=107, 
                  right=92, 
                  lower=157, 
                  output_path=f"output/conv_feature_map/wood/1-{i}.png")
                  
crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=24, 
                  upper=709, 
                  right=70, 
                  lower=763, 
                  output_path=f"output/conv_feature_map/wood/2-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=19, 
                  upper=950, 
                  right=59, 
                  lower=990, 
                  output_path=f"output/conv_feature_map/wood/3-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=140, 
                  upper=1000, 
                  right=195, 
                  lower=1060, 
                  output_path=f"output/conv_feature_map/wood/4-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=511, 
                  upper=1042, 
                  right=554, 
                  lower=1096, 
                  output_path=f"output/conv_feature_map/wood/5-{i}.png")

crop_image_pillow(image_path=f"output/conv_feature_map/wood-proposedB/5490/1-{i}.png", 
                  left=192, 
                  upper=130, 
                  right=236, 
                  lower=174, 
                  output_path=f"output/conv_feature_map/wood/6-{i}.png")
