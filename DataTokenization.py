import numpy as np
from pathlib import Path
from datasets import load_dataset
from collections import Counter

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

	"°": 96, "£": 97, "€": 98, "©": 99, "®": 100, "™": 101, "^": 102,
	"½": 103, "²": 104, "μ": 105, "π": 106,

	"±": 107, "×": 108, "÷": 109, "≈": 110, "≠": 111,
	"≤": 112, "≥": 113, "√": 114, "∞": 115,
	"∑": 116, "∏": 117, "∫": 118, "∂": 119,

	"α": 120, "β": 121, "γ": 122, "δ": 123,
	"Δ": 124, "λ": 125, "θ": 126, "σ": 127, "Ω": 128,

	"→": 129, "←": 130, "↑": 131, "↓": 132, "↔": 133,

	"¹": 134, "³": 135,
	"⁰": 136, "⁴": 137, "⁵": 138, "⁶": 139, "⁷": 140, "⁸": 141, "⁹": 142,

	"₀": 143, "₁": 144, "₂": 145, "₃": 146, "₄": 147,
	"₅": 148, "₆": 149, "₇": 150, "₈": 151, "₉": 152,

	"′": 153, "″": 154,
	"¼": 155, "¾": 156, "⅓": 157, "⅔": 158,
	"⅛": 159, "⅜": 160, "⅝": 161, "⅞": 162,

	"∈": 163, "∉": 164, "⊂": 165, "⊆": 166, "⊃": 167, "⊇": 168,
	"∪": 169, "∩": 170, "∅": 171, "∝": 172, "∴": 173, "∵": 174,
	"∇": 175, "∥": 176, "⊥": 177, "≡": 178, "≅": 179,

	"ε": 180, "η": 181, "κ": 182, "ρ": 183, "τ": 184, "φ": 185,
	"χ": 186, "ψ": 187, "ω": 188,

	"℃": 189, "℉": 190,
	"¢": 191, "¥": 192,

	"§": 193, "¶": 194, "†": 195, "‡": 196,
	"…": 197, "•": 198,

	"é": 199, "á": 200, "í": 201, "ó": 202, "ú": 203,
	"ñ": 204, "ü": 205, "ö": 206, "ä": 207, "ç": 208,

	"É": 209, "Á": 210, "Í": 211, "Ó": 212, "Ú": 213,
	"Ñ": 214, "Ü": 215, "Ö": 216, "Ä": 217, "Ç": 218,

	"æ": 219, "·": 220,
}

replacement_map = {
	"’": "'", "‘": "'", "‚": "'", "‛": "'", "´": "'",
	"“": '"', "”": '"', "„": '"', "‟": '"', "«": '"', "»": '"',

	"‐": "-", "-": "-", "‒": "-", "–": "-", "—": "-", "−": "-",

	"\u00a0": " ", "\u2000": " ", "\u2001": " ", "\u2002": " ",
	"\u2003": " ", "\u2004": " ", "\u2005": " ", "\u2006": " ",
	"\u2007": " ", "\u2008": " ", "\u2009": " ", "\u200a": " ",
	"\u202f": " ", "\u205f": " ", "\u3000": " ",

	"\u200b": "", "\u200c": "", "\u200d": "", "\ufeff": "",

	"µ": "μ",
	"º": "°",

	"è": "e", "ê": "e", "ë": "e", "ē": "e",
	"È": "E", "Ê": "E", "Ë": "E", "Ē": "E",

	"à": "a", "â": "a", "ā": "a", "ã": "a", "å": "a",
	"À": "A", "Â": "A", "Ā": "A", "Ã": "A", "Å": "A",

	"ì": "i", "î": "i", "ï": "i", "ī": "i",
	"Ì": "I", "Î": "I", "Ï": "I", "Ī": "I",

	"ò": "o", "ô": "o", "ō": "o", "õ": "o",
	"Ò": "O", "Ô": "O", "Ō": "O", "Õ": "O",

	"ù": "u", "û": "u", "ū": "u",
	"Ù": "U", "Û": "U", "Ū": "U",

	"ſ": "s",
	"Æ": "AE",
}

output_path = Path("finewebedu_uint8.bin")

# Set to None for the entire sample.
max_characters = None

unknown_id = character_to_id[UNKNOWN_TOKEN]

# Build these once instead of rebuilding/looking things up for every document.
translation_table = str.maketrans(replacement_map)

# Unicode has code points from 0 through 0x10FFFF.
token_lookup = np.full(0x110000, unknown_id, dtype=np.uint8)

for character, token_id in character_to_id.items():
	if len(character) == 1:
		token_lookup[ord(character)] = token_id

dataset = load_dataset(
	"HuggingFaceFW/fineweb-edu",
	name="sample-10BT",
	split="train",
	streaming=True,
)

unknown_characters_list = Counter()

total_characters = 0
unknown_characters = 0
document_count = 0

with open(output_path, "wb") as output_file:

	for document in dataset:
		text = document["text"] + "\n\n"

		# Do all normalization in one string traversal.
		text = text.translate(translation_table)

		if max_characters is not None:
			remaining_characters = max_characters - total_characters

			if remaining_characters <= 0:
				break

			text = text[:remaining_characters]

		# Convert Unicode characters -> Unicode integer code points.
		codepoints = np.fromiter(
			map(ord, text),
			dtype=np.int32,
			count=len(text),
		)

		# Vectorized code-point -> token-ID conversion.
		encoded = token_lookup[codepoints]

		# Count unknowns.
		unknown_mask = encoded == unknown_id
		unknown_count = int(unknown_mask.sum())

		if unknown_count:
			unknown_characters += unknown_count

			# Only do the slower Python Counter work for actual unknown positions.
			unknown_codepoints = codepoints[unknown_mask]

			for codepoint in unknown_codepoints:
				unknown_characters_list[chr(int(codepoint))] += 1

		# NumPy uint8 buffer can be written directly as raw bytes.
		output_file.write(encoded.tobytes())

		total_characters += encoded.size
		document_count += 1

		if document_count % 1000 == 0:
			print(
				f"Documents: {document_count:,} | "
				f"Characters: {total_characters:,} | "
				f"Unknown rate: {unknown_characters / total_characters:.5%}"
			)

			if document_count % 10000 == 0:
				print(unknown_characters_list.most_common(30))

print()
print(f"Saved to: {output_path}")
print(f"Documents: {document_count:,}")
print(f"Characters/tokens: {total_characters:,}")
print(f"Unknown characters: {unknown_characters:,}")
print(f"Unknown rate: {unknown_characters / total_characters:.4%}")
print(f"Approximate size: {total_characters / 1024**3:.2f} GiB")