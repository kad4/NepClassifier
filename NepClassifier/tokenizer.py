import re


def tokenize(text):
    """ Tokenize the given text """

    # Removing unnecessary items
    remove_exp = re.compile("[\d]+")
    removed_text = remove_exp.sub("", text)

    # Extracting words from text
    # Splits complex combinations in single step
    extract_exp = re.compile("[\s।|!?.,:;%+\-–*/'‘’“\"()]+")
    words = extract_exp.split(removed_text)

    # Returns the non-empty items only
    return([word for word in words if word != ''])
