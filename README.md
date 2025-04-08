# BookNLP

A natural language processing pipeline for analyzing works of fiction, including entity detection, quotation attribution, and character relationship analysis.

## Prerequisites

- Python 3.9 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/booknlp.git
cd booknlp
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

4. Install the spaCy model:
```bash
python -m spacy download en_core_web_sm
```

## Usage

1. Make sure your virtual environment is activated:
```bash
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Run BookNLP on a text file:
```bash
./run_booknlp.py input_file.txt --output-dir output/directory
```

### Command Line Arguments

- `input_file`: The text file to process (required)
- `--output-dir`: Directory where output files will be saved (default: 'output')
- `--model`: Model size to use - 'big' or 'small' (default: 'small')

### Output Files

The pipeline generates several output files in the specified output directory:

1. `{book_id}.tokens`: Word-level information including:
   - Paragraph and sentence IDs
   - Word forms and lemmas
   - Part-of-speech tags
   - Dependency relations
   - Event annotations

2. `{book_id}.entities`: Named entity information including:
   - Entity types
   - Coreference IDs
   - Text spans

3. `{book_id}.quotes`: Quotation information including:
   - Quoted text
   - Speaker attribution
   - Coreference information

4. `{book_id}.supersense`: Semantic categories for words:
   - Verb categories
   - Noun categories

5. `{book_id}.event`: Event annotations including:
   - Event types
   - Participants
   - Temporal information

6. `{book_id}.book`: JSON file containing:
   - Character information
   - Relationships
   - Actions
   - Attributes

7. `{book_id}.book.html`: Interactive HTML visualization of the text with:
   - Entity annotations
   - Character relationships
   - Interactive features

## Example

```bash
# Process a text file named "emma.txt" with maximum accuracy
./run_booknlp.py emma.txt --output-dir output/emma --model big
```

