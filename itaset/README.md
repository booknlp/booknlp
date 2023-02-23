# Books


In the folder books there are 87 works of Italian-language fiction not annotated. Some are collections of multiple works: among these collections the best works were chosen to be annotated.

There is also a metadata file that contains informations about these works. These informations are:

* id, the number that identifies the book/collection

* title

* author

* authoryearofbirth, the author's year of birth

* authoryearofdeath, the author's year of death

# ItaSet


ItaSet is an annotated dataset of 100 works of Italian-language fiction to support tasks in natural language processing and the computational humanities.

ItaSet currently contains annotations only for entities in a sample of ~2,000 words from each text.

Now the steps performed sequentially to obtain ItaSet are explained:

* All the texts were truncated firstly by a Python program to have 2,100 words. Then they were further reduced down to about ~2,000 words afterwards a human analysis.

* All the texts were "formatted" in the BRAT format, which assumes that the raw content of the text appears in one file and the annotations associated with it appear in another file/files: all the texts were organized eliminating carriage returns and separating words and punctuation marks with white spaces.