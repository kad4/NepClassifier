import re
import os

class NepStemmer():
    def __init__(self):
        self.stems_set = set()
        self.filter_set = set()
        
        self.suffixes = {
            1 : [
                    'छ', 'ा', 'न', 'ए', 'े', 'ी', 'ि', 'ै', 'ई', 'ओ', 'उ'
            ],
            2 : [
                    'ले', 'को', 'का', 'की', 'ँछ', 'छु', 'छे', 'छौ', 'ने',
                    'यो', 'दा', 'दै', 'नु', 'एँ',  'ेँ', 'छौ', 'तै', 'थे',
                    'औं', 'ौं','यौ', 'ला', 'ली', 'मा', 'उँ', 'ुँ', 'ेर',
                    'एर', 'ाइ', 'आइ', 'इन', 'कै', 'ता', 'दो', 'ाए', 'ना',
                    'ौँ', 'िक','ाइ','ाउ'
            ],
            3 : [
                    'हरू', 'लाई', 'बाट', 'एका', 'ँछु', 'इने', 'ँदै', 'छौँ',
                    'छस्', 'छन्', 'थेँ', 'एस्', 'ेस्', 'ओस्', 'ोस्','उन्',
                    'ुन्', 'एन्', 'ेन्', 'यौं', 'इस्', 'िस्','्नो', 'साथ',
                    'नन्', 'िया', 'झैँ', 'न्छ', 'ेका','एको', 'ेको',
                    'शील', 'सार', 'ालु', 'ईन्', 'ीन्', 'िलो', 'ाडी'
            ],
            4 : [
                    'छेस्', 'छ्यौ', 'ँछन्', 'छिन्', 'थ्यौ', 'थिन्',
                    'औंला', 'ौंला', 'लिन्', 'लान्','लास्','लिस्',
                    'होस्', 'माथी', 'तर्फ', 'मुनि', 'पर्छ', 'ियाँ',
                    'न्छौ', 'सम्म','ाएको', 'सुकै', 'यालु', 'डालु', 
                    'उँला', 'ुँला'
            ],
            5 : [
                    'थ्यौँ', 'स्थित', 'तुल्य', 'चाँहि', 'चाहीँ', 'मात्र',
                    'न्छन्', 'न्छस्', 'मध्ये'
            ],
            6 : [
                    'पालिका', 'अनुसार', 'न्छ्यौ', 'न्छेस्', 'न्छिन्',
                    'इन्जेल', 'िन्जेल', 'ुन्जेल', 'उन्जेल'
            ],
        }

        # Read stems and filter words from files
        self.read_stems()

    # Read stems from file
    def read_stems(self):
        # Reads the word stems
        base_path = os.path.dirname(__file__)

        stems_path = os.path.join(base_path, 'stems.txt')
        filter_path = os.path.join(base_path, 'filter.txt')

        with open(stems_path) as file:
            lines = file.readlines()

        # Constructing stems set
        for line in lines:
            new_line = line.replace('\n', '')
            stem = new_line.split('|')[0]
            self.stems_set.add(stem)

        # Reads filter words
        with open(filter_path) as file:
            filter_stems = file.read().split('\n')

        self.filter_set = set(filter_stems)

    # Removes suffix
    def remove_suffix(self, word):
        for L in 2,3,4,5,6:
            if len(word) > L + 1:
                for suf in self.suffixes[L]:
                    if word.endswith(suf):
                        return word[:-L]
            else:
                break
        return word

    # Tokenizes the given text
    def tokenize(self, text):
        # Removing unnecessary items
        remove_exp = re.compile("[\d]+")
        removed_text = remove_exp.sub("", text)

        # Extracting words from text
        # Splits complex combinations in single step
        extract_exp = re.compile("[\s।|!?.,:;%+\-–*/'‘’“\"()]+")
        words = extract_exp.split(removed_text)

        # Returns the non-empty items only
        return([word for word in words if word!=''])

    # Returns the stem
    def stem(self, word):
        word_stem = self.remove_suffix(word)
        if(word_stem in self.stems_set):
            return word_stem
        else:
            return word

    # Returns stems list
    def get_stems(self, text):
        # Obtain tokens of the text
        tokens = self.tokenize(text)

        return([self.stem(token) for token in tokens])

    # Returns known stems list
    def get_known_stems(self, text):
        # Obtain tokens of the text
        tokens = self.tokenize(text)

        # Obtain the stem list
        stems_list = [self.stem(token) for token in tokens]

        known_stem = [
            stem 
            for stem in stems_list 
            if stem in self.stems_set
            and stem not in self.filter_set
        ]

        return(known_stem)


        
