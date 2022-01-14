
class LZW:
    @staticmethod
    def compress(uncompressed: str) -> list:
        dictionary = {}
        word = ""
        result = []
        dict_size = 256

        for i in range(dict_size):
            dictionary[str(chr(i))] = i

        for index in range(len(uncompressed)):
            current_char = str(uncompressed[index])
            word_and_symbol = word + current_char

            if word_and_symbol in dictionary:
                word = word_and_symbol
            else:
                try:
                    result.append(dictionary[word])
                except:
                    print(index)
                    print(word)
                    print("-------------")
                dictionary[word_and_symbol] = dict_size
                dict_size += 1
                word = str(current_char)

        if word != "":
            result.append(dictionary[word])

        return result

    @staticmethod
    def decompress(compressed: list) -> str:
        dictionary = {}
        dict_size = 256

        for i in range(dict_size):
            dictionary[i] = str(chr(i))

        word = str(chr(compressed[0]))
        result = word

        for i in range(1, len(compressed)):
            temp = compressed[i]

            if temp in dictionary:
                entry = dictionary[temp]
            else:
                if temp == dict_size:
                    entry = word + str(word[0])
                else:
                    return None

            result += entry
            dictionary[dict_size] = word + str(entry[0])
            dict_size += 1
            word = entry

        return result
