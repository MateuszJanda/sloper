from PIL import Image, ImageDraw, ImageFont
import os


img = Image.new('RGB', size=(400, 400))
draw = ImageDraw.Draw(img)

# font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', size=12)
# font = ImageFont.truetype('/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf', size=12)
font = ImageFont.truetype('/usr/share/fonts/truetype/ancient-scripts/Symbola_hint.ttf', size=24)
# font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size=12)
# font = ImageFont.truetype('DejaVuSansMono.ttf', size=12)
# font = ImageFont.load('ter-u32n_unicode.pcf')
# font = ImageFont.load('ter-u32n_unicode.pil')
# font = ImageFont.load('Symbola_hint.pil')
# font = ImageFont.load_default().font

color = 'rgb(255, 255, 255)' # black color

draw.text(xy=(50, 30), text='01234567890', fill=color, font=font)
draw.text(xy=(50, 50), text='Happy Birthday!', fill=color, font=font)
draw.text(xy=(50, 70), text='/\\/\\/\\/\\', fill=color, font=font)
draw.text(xy=(50, 90), text='⢧⢧⢧⢧⢧⢧⢧', fill=color, font=font)


img.save('output.png', 'PNG', optimize=True, quality=20)

os.system('display output.png')