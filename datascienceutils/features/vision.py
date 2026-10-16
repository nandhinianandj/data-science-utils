
def tesseract_ocr(imagefile, datadir):
    try:
        from PIL import Image
    except ImportError:
        import Image
    import pytesseract
    image = Image.open(os.path.join(datadir,imagefile))
    binary_image = image.convert('L')
    string = pytesseract.image_to_string(binary_image)
    return string

def easyocr(imagefile, datadir):
    import easyocr
    reader = easyocr.Reader(['en'])
    string = reader.readtext(os.path.join(datadir, fil))
    return string
