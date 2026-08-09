import torch
from pathlib import Path

UNKNOWN_TOKEN = "<unk>"

character_to_id = {
	"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6,
	"h": 7, "i": 8, "j": 9, "k": 10, "l": 11, "m": 12,
	"n": 13, "o": 14, "p": 15, "q": 16, "r": 17, "s": 18,
	"t": 19, "u": 20, "v": 21, "w": 22, "x": 23, "y": 24,
	"z": 25,

	"A": 26, "B": 27, "C": 28, "D": 29, "E": 30, "F": 31,
	"G": 32, "H": 33, "I": 34, "J": 35, "K": 36, "L": 37,
	"M": 38, "N": 39, "O": 40, "P": 41, "Q": 42, "R": 43,
	"S": 44, "T": 45, "U": 46, "V": 47, "W": 48, "X": 49,
	"Y": 50, "Z": 51,

	"1": 52, "2": 53, "3": 54, "4": 55, "5": 56,
	"6": 57, "7": 58, "8": 59, "9": 60, "0": 61,

	".": 62, ",": 63, "?": 64, "!": 65, ":": 66, ";": 67,
	"'": 68, "&": 69, "@": 70, "#": 71, "$": 72, "%": 73,
	"*": 74, "+": 75, "-": 76, "_": 77, "(": 78, ")": 79,
	"[": 80, "]": 81, "{": 82, "}": 83, "<": 84, ">": 85,
	"=": 86, "~": 87, "/": 88, "`": 89, "|": 90, '"': 91,
	"\\": 92, " ": 93,

	UNKNOWN_TOKEN: 94,
	"\n": 95,
}

input_path = Path("wiki.train.tokens")
output_path = Path("wiki103_uint8.pt")

chunk_size = 1_000_000

encoded_chunks = []

with open(input_path, "r", encoding="utf-8") as file:
	while True:
		text_chunk = file.read(chunk_size)

		if not text_chunk:
			break

		encoded_chunk = torch.tensor(
			[
				character_to_id.get(character, character_to_id[UNKNOWN_TOKEN])
				for character in text_chunk
			],
			dtype=torch.uint8,
		)

		encoded_chunks.append(encoded_chunk)

		print(
			f"Processed {sum(chunk.numel() for chunk in encoded_chunks):,} characters"
		)

token_ids = torch.cat(encoded_chunks)

torch.save(token_ids, output_path)

print()
print(f"Saved to: {output_path}")
print(f"Number of tokens: {token_ids.numel():,}")
print(f"Data type: {token_ids.dtype}")
print(f"Approximate memory: {token_ids.numel() / 1024**2:.2f} MiB")