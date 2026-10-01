import sys,glob
from PIL import Image, ImageDraw
d=sys.argv[1]; out=sys.argv[2]
fs=sorted(glob.glob(d+'/f_*.png'),key=lambda f:float(f.split('f_')[-1][:-4]))
w,h=640,360; cols=3; rows=(len(fs)+cols-1)//cols
sheet=Image.new('RGB',(w*cols,h*rows))
for i,f in enumerate(fs):
    im=Image.open(f).convert('RGB').resize((w,h)); dr=ImageDraw.Draw(im); dr.text((8,8),f.split('f_')[-1][:-4],fill=(255,255,0))
    sheet.paste(im,((i%cols)*w,(i//cols)*h))
sheet.save(out,quality=88)
