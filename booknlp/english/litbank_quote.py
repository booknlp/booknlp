import re
from collections import Counter

class QuoteTagger:
	

	def tag(self, toks):

		predictions=[]
		currentQuote=[]
		curStartTok=None
		lastPar=None

		quote_symbols=Counter()

		for tok in toks:
			if tok.text == "“" or tok.text == "”" or tok.text == "\"" or tok.text == "“":
				quote_symbols["DOUBLE_QUOTE"]+=1
			elif tok.text == "‘" or tok.text == "’" or tok.text == "'":
				quote_symbols["SINGLE_QUOTE"]+=1
			elif tok.text == "—":
				quote_symbols["DASH"]+=1


		quote_symbol="DOUBLE_QUOTE"
		if len(quote_symbols) > 0:
			quote_symbol=quote_symbols.most_common()[0][0]

		for tok in toks:

			w=tok.text

			for w_idx, w_char in enumerate(w):
				if w_char== "“" or w_char == "”" or w_char == "\"":
					w="DOUBLE_QUOTE"
				elif w_char == "‘" or w_char == "’" or w_char == "'":
					if w_idx == 0:
						suff=w[w_idx+1:]
						if suff != "s" and suff != "d" and suff != "ll" and suff != "ve":
							w="SINGLE_QUOTE"

			# start over at each new paragraph
			if tok.paragraph_id != lastPar and lastPar is not None:

				if len(currentQuote) > 0:
					predictions.append((curStartTok, tok.token_id-1))
				curStartTok=None
				currentQuote=[]

			if w == quote_symbol:

				if curStartTok is not None:

					if len(currentQuote) > 0:
						predictions.append((curStartTok, tok.token_id))
						currentQuote.append(tok.text)

					curStartTok=None
					currentQuote=[]
				else:
					curStartTok=tok.token_id

			
			if curStartTok is not None:
				currentQuote.append(tok.text)

			lastPar=tok.paragraph_id

		for start, end in predictions:
			for i in range(start, end+1):
				toks[i].inQuote=True

		return predictions



