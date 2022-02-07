class FIC:
    @staticmethod
    def compress(input_list_of_integers: list) -> list:
        result = []
        for number in input_list_of_integers:
            if number < (1 << 7):
                result.append((number).to_bytes(1, byteorder="little"))
            elif number < (1 << 14):
                result.append(((number & 0x7F) | 0x80).to_bytes(1, byteorder="little"))
                result.append((number >> 7).to_bytes(1, byteorder="little"))
            elif number < (1 << 21):
                result.append(((number & 0x7F) | 0x80).to_bytes(1, byteorder="little"))
                result.append(
                    (((number >> 7) & 0x7F) | 0x80).to_bytes(1, byteorder="little")
                )
                result.append((number >> 14).to_bytes(1, byteorder="little"))
            elif number < (1 << 28):
                result.append(((number & 0x7F) | 0x80).to_bytes(1, byteorder="little"))
                result.append((((number >> 7) & 0x7F) | 0x80)).to_bytes(
                    1, byteorder="little"
                )
                result.append(
                    (((number >> 14) & 0x7F) | 0x80).to_bytes(1, byteorder="little")
                )
                result.append((number >> 21).to_bytes(1, byteorder="little"))
            else:
                result.append(((number & 0x7F) | 0x80).to_bytes(1, byteorder="little"))
                result.append(
                    (((number >> 7) & 0x7F) | 0x80).to_bytes(1, byteorder="little")
                )
                result.append(
                    (((number >> 14) & 0x7F) | 0x80).to_bytes(1, byteorder="little")
                )
                result.append(
                    (((number >> 21) & 0x7F) | 0x80).to_bytes(1, byteorder="little")
                )
                result.append((number >> 28).to_bytes(1, byteorder="little"))

        return result

    @staticmethod
    def decompress(input_byte_array: list) -> list:
        result = []
        position = 0

        while len(input_byte_array) > position:
            byte_to_int = int.from_bytes(
                input_byte_array[position : position + 1],
                signed=True,
                byteorder="little",
            )
            position += 1
            temp_int = byte_to_int & 0x7F
            if byte_to_int >= 0:
                result.append(temp_int)
                continue
            byte_to_int = int.from_bytes(
                input_byte_array[position : position + 1],
                signed=True,
                byteorder="little",
            )
            position += 1
            temp_int |= (byte_to_int & 0x7F) << 7
            if byte_to_int >= 0:
                result.append(temp_int)
                continue
            byte_to_int = int.from_bytes(
                input_byte_array[position : position + 1],
                signed=True,
                byteorder="little",
            )
            position += 1
            temp_int |= (byte_to_int & 0x7F) << 14
            if byte_to_int >= 0:
                result.append(temp_int)
                continue
            byte_to_int = int.from_bytes(
                input_byte_array[position : position + 1],
                signed=True,
                byteorder="little",
            )
            position += 1
            temp_int |= (byte_to_int & 0x7F) << 21
            if byte_to_int >= 0:
                result.append(temp_int)
                continue
            byte_to_int = int.from_bytes(
                input_byte_array[position : position + 1],
                signed=True,
                byteorder="little",
            )
            position += 1
            temp_int |= byte_to_int << 28
            result.append(temp_int)

        return result
