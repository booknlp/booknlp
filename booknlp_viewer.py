import streamlit as st
import os
import json
import pandas as pd

st.set_page_config(layout="wide")

st.title("BookNLP Output Viewer")

# --- Configuration ---
DEFAULT_OUTPUT_DIR = "output" # Default path where book-specific output folders reside

# --- Helper Functions ---
@st.cache_data
def load_book_data(book_output_path, book_id):
    """Loads the .book JSON data for a given book_id in the output directory."""
    # Construct filename using book_id
    book_json_path = os.path.join(book_output_path, f"{book_id}.book")
    data = None
    if os.path.exists(book_json_path):
        try:
            with open(book_json_path, 'r') as f:
                # Read the single line and parse it as JSON
                data = json.loads(f.readline())
            st.success(f"Successfully loaded: {book_json_path}")
        except json.JSONDecodeError:
            st.error(f"Error decoding JSON from {book_json_path}. Is the file format correct?")
        except Exception as e:
            st.error(f"Error loading {book_json_path}: {e}")
    else:
        st.warning(f"File not found: {book_json_path}")
    return data

@st.cache_data
def load_entities_data(book_output_path, book_id):
    """Loads the .entities TSV data for a given book_id."""
    entities_path = os.path.join(book_output_path, f"{book_id}.entities")
    df = None
    entity_cols = ["coref_id", "start_token", "end_token", "mention_type", "entity_type", "text"]
    if os.path.exists(entities_path):
        try:
            df = pd.read_csv(entities_path, sep='\t', header=None, names=entity_cols, quoting=3) # quoting=3 to handle potential quotes in text
            st.success(f"Successfully loaded: {entities_path}")
        except Exception as e:
            st.error(f"Error loading {entities_path}: {e}")
    else:
        st.warning(f"File not found: {entities_path}")
    return df

@st.cache_data
def load_quotes_data(book_output_path, book_id):
    """Loads the .quotes TSV data for a given book_id."""
    quotes_path = os.path.join(book_output_path, f"{book_id}.quotes")
    df = None
    quote_cols = [
        "quote_start_token",
        "quote_end_token",
        "mention_start_token",
        "mention_end_token",
        "mention_text",
        "speaker_coref_id",
        "quote_text"
    ]
    if os.path.exists(quotes_path):
        try:
            # Use quoting=3 (csv.QUOTE_NONE) as quotes can appear within the quote_text
            df = pd.read_csv(quotes_path, sep='\t', header=None, names=quote_cols, quoting=3, on_bad_lines='skip')
            st.success(f"Successfully loaded: {quotes_path}")
        except Exception as e:
            st.error(f"Error loading {quotes_path}: {e}")
    else:
        st.warning(f"File not found: {quotes_path}")
    return df

@st.cache_data
def load_supersense_data(book_output_path, book_id):
    """Loads the .supersense TSV data for a given book_id."""
    supersense_path = os.path.join(book_output_path, f"{book_id}.supersense")
    df = None
    supersense_cols = [
        "start_token",
        "end_token",
        "supersense_category"
    ]
    if os.path.exists(supersense_path):
        try:
            df = pd.read_csv(supersense_path, sep='\t', header=None, names=supersense_cols, quoting=3)
            st.success(f"Successfully loaded: {supersense_path}")
        except Exception as e:
            st.error(f"Error loading {supersense_path}: {e}")
    else:
        st.warning(f"File not found: {supersense_path}")
    return df

@st.cache_data(show_spinner="Loading tokens data (can be large)...")
def load_tokens_data(book_output_path, book_id):
    """Loads the .tokens TSV data for a given book_id."""
    tokens_path = os.path.join(book_output_path, f"{book_id}.tokens")
    df = None
    token_cols = [
        "paragraph_id",
        "sentence_id",
        "token_id_in_sentence",
        "token_id_in_document",
        "word",
        "lemma",
        "byte_onset",
        "byte_offset",
        "pos_tag",
        "dependency_relation",
        "syntactic_head_token_id",
        "event"
    ]
    if os.path.exists(tokens_path):
        try:
            # Use low_memory=False for potentially faster parsing of large files with mixed types
            # Specify dtypes where possible if known and consistent to save memory
            df = pd.read_csv(tokens_path, sep='\t', header=None, names=token_cols, quoting=3, low_memory=False)
            # Convert relevant columns to more memory-efficient types if possible
            # e.g., df['paragraph_id'] = df['paragraph_id'].astype('int32') # Be careful with potential NaNs if converting to int
            st.success(f"Successfully loaded: {tokens_path} ({len(df):,} tokens)")
        except Exception as e:
            st.error(f"Error loading {tokens_path}: {e}")
    else:
        st.warning(f"File not found: {tokens_path}")
    return df

# --- Sidebar for Selection ---
st.sidebar.header("Select Book Output")

# Input for the main output directory
output_base_dir = st.sidebar.text_input(
    "Enter the main output directory path:",
    value=DEFAULT_OUTPUT_DIR
)

if not os.path.isdir(output_base_dir):
    st.sidebar.error(f"Directory not found: {output_base_dir}")
    st.stop() # Stop execution if base directory is invalid

# Find potential book output directories (subdirectories)
try:
    book_dirs = [d for d in os.listdir(output_base_dir) if os.path.isdir(os.path.join(output_base_dir, d))]
    # Filter out hidden directories like .DS_Store if necessary (though os.path.isdir handles this)
    book_dirs = [d for d in book_dirs if not d.startswith('.')]

except FileNotFoundError:
    st.sidebar.error(f"Directory not found: {output_base_dir}")
    st.stop()
except Exception as e:
    st.sidebar.error(f"Error listing directories in {output_base_dir}: {e}")
    st.stop()

if not book_dirs:
    st.sidebar.warning(f"No subdirectories found in {output_base_dir}. Please run BookNLP first.")
    st.stop()

# Dropdown to select a specific book's output directory
selected_book_dir_name = st.sidebar.selectbox(
    "Choose a book output folder:",
    options=book_dirs
)

selected_book_path = os.path.join(output_base_dir, selected_book_dir_name)

# --- Find and Select Book ID within the directory ---
book_ids_found = []
try:
    files_in_dir = os.listdir(selected_book_path)
    for filename in files_in_dir:
        if filename.endswith(".book"):
            book_id = filename[:-len(".book")] # Remove .book extension
            book_ids_found.append(book_id)
except Exception as e:
    st.sidebar.error(f"Error scanning directory {selected_book_path}: {e}")
    st.stop()

if not book_ids_found:
    st.sidebar.warning(f"No .book files found in {selected_book_path}")
    st.stop()

# Dropdown to select the book_id
selected_book_id = st.sidebar.selectbox(
    "Choose a Book ID:",
    options=sorted(book_ids_found)
)

# --- Main Area ---
st.header(f"Exploring: {selected_book_dir_name} / {selected_book_id}")

# Load the data for the selected book using the selected book_id
book_data = load_book_data(selected_book_path, selected_book_id)
# Load entities data
entities_df = load_entities_data(selected_book_path, selected_book_id)
# Load quotes data
quotes_df = load_quotes_data(selected_book_path, selected_book_id)
# Load supersense data
supersense_df = load_supersense_data(selected_book_path, selected_book_id)
# Load tokens data (will be cached, not displayed directly)
tokens_df = load_tokens_data(selected_book_path, selected_book_id)

if book_data:
    # Optional: Keep the raw JSON preview if desired, or remove it
    # st.subheader("Raw Book Data (.book JSON)")
    # st.json(book_data, expanded=False) # Show collapsed by default

    # --- Characters Section ---
    if "characters" in book_data and book_data["characters"]:
        st.markdown("---")
        st.header("Characters")

        character_list = book_data.get("characters", []) # Get list of character dicts

        # Create a list of character names/IDs for the dropdown
        # Assuming each character dict has an 'id' or 'name' key we can use.
        # Let's use the 'id' for uniqueness, and maybe display name + id
        # We need to inspect the actual JSON structure to be sure.
        # For now, let's assume 'id' exists and is unique.
        # Let's also extract a 'name' if available for display purposes.
        # Trying to find a primary name - often the first proper mention.
        def get_display_name(char_info):
            primary_name = char_info.get("id", "Unknown ID")
            props = char_info.get("proper", [])
            if props:
                # Use the most frequent proper name as display name
                primary_name = sorted(props, key=lambda x: x['c'], reverse=True)[0]['n']
            return f"{primary_name} (ID: {char_info.get('id', 'N/A')})"

        character_options = {get_display_name(char): i for i, char in enumerate(character_list)} # Map display name to index

        if character_options:
            selected_char_display_name = st.selectbox(
                "Select a character:",
                options=list(character_options.keys())
            )

            if selected_char_display_name:
                selected_char_index = character_options[selected_char_display_name]
                selected_char_data = character_list[selected_char_index]

                st.subheader(f"Details for {selected_char_display_name}")

                # --- TEMPORARY DEBUG: Show raw character data ---
                # st.warning("Debugging Info: Raw data for selected character")
                # st.json(selected_char_data)
                # --- END TEMPORARY DEBUG ---

                # Display different aspects of the character data
                # Get counts by length of the lists
                agent_count = len(selected_char_data.get('agent', []))
                patient_count = len(selected_char_data.get('patient', []))
                poss_count = len(selected_char_data.get('poss', []))

                # Extract gender info using correct keys
                gender_info = selected_char_data.get('g', {})
                inferred_gender = gender_info.get('argmax', 'N/A')
                pronoun_ref_count = gender_info.get('total', 0)

                st.write(f"**Agent in {agent_count} actions**")
                st.write(f"**Patient in {patient_count} actions**")
                st.write(f"**Possessor of {poss_count} objects**")
                st.write(f"**Gender:** {inferred_gender} (based on {pronoun_ref_count} pronoun references)")

                # Access mentions via the 'mentions' key
                mentions_data = selected_char_data.get("mentions", {})

                with st.expander("Mentions"):
                    st.write("**Proper Names:**")
                    st.json(mentions_data.get("proper", []))
                    st.write("**Common Nouns:**")
                    st.json(mentions_data.get("common", []))
                    st.write("**Pronouns:**")
                    st.json(mentions_data.get("pronoun", []))

                # Optionally display other fields if they exist (using len for counts if they are lists)
                mod_count = len(selected_char_data.get('mod', []))
                if mod_count > 0:
                     st.write(f"**Modifiers Applied: {mod_count}**")
                     with st.expander("Modifiers List"):
                         st.json(selected_char_data.get('mod', []))

        else:
            st.info("No characters found in the .book data structure.")

    else:
        st.info("No 'characters' data found in the loaded .book JSON file.")

    # --- Entities Section ---
    if entities_df is not None:
        st.markdown("---")
        st.header("Entities")

        tab1, tab2 = st.tabs(["All Mentions", "Grouped by Entity ID"])

        with tab1:
            st.subheader("All Mentions")
            # Get unique entity types for filtering
            all_entity_types = sorted(entities_df['entity_type'].unique())

            # Multiselect filter for entity types
            selected_types = st.multiselect(
                "Filter by entity type:",
                options=all_entity_types,
                default=all_entity_types, # Select all by default
                key="entity_type_filter_all" # Unique key for this widget
            )

            if selected_types:
                filtered_df = entities_df[entities_df['entity_type'].isin(selected_types)]
                # Hide the default DataFrame index
                st.dataframe(filtered_df, hide_index=True)
            else:
                st.info("Select at least one entity type to display.")

        with tab2:
            st.subheader("Grouped by Entity ID")
            try:
                # Group by coref_id and aggregate
                grouped_entities = entities_df.groupby('coref_id').agg(
                    entity_type=('entity_type', 'first'),
                    total_mentions=('coref_id', 'size'),
                    mentions=('text', lambda x: list(x.unique()))
                ).reset_index()

                # Sort by total mentions descending
                grouped_entities = grouped_entities.sort_values(by='total_mentions', ascending=False)

                # --- Replace loop with Dropdown Selection ---
                # Create display names for the dropdown (e.g., ID - Type (Count))
                def get_entity_display_name(row):
                    # Attempt to get a primary name if it's a person entity
                    # This requires linking back to character data, complex for now.
                    # Default to ID - Type (Count)
                    return f"{row.coref_id} - {row.entity_type} ({row.total_mentions} mentions)"

                grouped_entities['display_name'] = grouped_entities.apply(get_entity_display_name, axis=1)
                # Remove the potentially problematic index-based mapping
                # entity_options = pd.Series(grouped_entities.index, index=grouped_entities.display_name).to_dict()

                selected_entity_display_name = st.selectbox(
                    "Select an Entity ID to view details:",
                    options=grouped_entities.display_name.tolist(), # Use the list of display names directly
                    key="entity_group_selector"
                )

                if selected_entity_display_name:
                    # Filter the dataframe directly based on the selected display name
                    selected_entity_data = grouped_entities[grouped_entities['display_name'] == selected_entity_display_name]
                    # Ensure we have exactly one match and get the first row
                    if not selected_entity_data.empty:
                        selected_entity_data = selected_entity_data.iloc[0]

                        st.markdown("---")
                        st.write(f"**Coref ID:** {selected_entity_data.coref_id}")
                        st.write(f"**Entity Type:** {selected_entity_data.entity_type}")
                        st.write(f"**Total Mentions:** {selected_entity_data.total_mentions}")
                        with st.expander("View Unique Mentions"):
                            # Display list of mentions (can be long)
                            st.json(selected_entity_data.mentions)
                    else:
                        st.error("Could not find data for the selected entity.")

            except Exception as e:
                st.error(f"Error processing grouped entities: {e}")

    else:
        st.info("Could not load entities data (.entities file). Check sidebar warnings/errors.")

    # --- Quotes Section ---
    if quotes_df is not None:
        st.markdown("---")
        st.header("Quotes")

        # Display the quotes
        # Consider adding filtering by speaker_coref_id later
        st.dataframe(quotes_df, hide_index=True)

    else:
        st.info("Could not load quotes data (.quotes file). Check sidebar warnings/errors.")

    # --- Supersenses Section ---
    if supersense_df is not None:
        st.markdown("---")
        st.header("Supersenses")

        all_supersense_types = sorted(supersense_df['supersense_category'].astype(str).unique())

        selected_supersenses = st.multiselect(
            "Filter by supersense category:",
            options=all_supersense_types,
            default=[], # Start with none selected for brevity
            key="supersense_filter"
        )

        if selected_supersenses:
            filtered_supersense_df = supersense_df[supersense_df['supersense_category'].isin(selected_supersenses)]
            st.dataframe(filtered_supersense_df, hide_index=True)
        else:
            st.info("Select supersense categories to display.")
    else:
        st.info("Could not load supersenses data (.supersense file). Check sidebar warnings/errors.")

    # --- Events Section ---
    if tokens_df is not None:
        st.markdown("---")
        st.header("Events (Realis)")
        try:
            # Ensure the event column exists and handle potential NaN values gracefully
            if 'event' in tokens_df.columns:
                # Filter tokens that are part of an event (value is not 'O' and not NaN)
                event_tokens = tokens_df[tokens_df['event'].notna() & (tokens_df['event'] != 'O')]

                if not event_tokens.empty:
                    # Get unique sentence identifiers that contain events
                    sentences_with_events = event_tokens[['paragraph_id', 'sentence_id']].drop_duplicates().sort_values(by=['paragraph_id', 'sentence_id'])

                    # --- Add robustness: Filter out non-numeric IDs before iterating ---
                    sentences_with_events['paragraph_id'] = pd.to_numeric(sentences_with_events['paragraph_id'], errors='coerce')
                    sentences_with_events['sentence_id'] = pd.to_numeric(sentences_with_events['sentence_id'], errors='coerce')
                    sentences_with_events = sentences_with_events.dropna(subset=['paragraph_id', 'sentence_id'])
                    # --- End robustness addition ---

                    st.write(f"Found {len(sentences_with_events):,} sentences containing asserted realis events.")

                    # Display sentence by sentence (consider adding pagination/search later if too many)
                    # Group original tokens by sentence ONCE for efficient lookup
                    sentence_groups = tokens_df.groupby(['paragraph_id', 'sentence_id'])

                    # Use an expander to make the list collapsible if long
                    with st.expander("View Sentences with Events", expanded=False):
                        for idx, row in sentences_with_events.iterrows():
                            para_id = int(row['paragraph_id'])
                            sent_id = int(row['sentence_id'])
                            try:
                                # Get all tokens for this sentence
                                sentence_tokens = sentence_groups.get_group((para_id, sent_id))
                                # Reconstruct sentence text
                                sentence_text = ' '.join(sentence_tokens['word'].astype(str).tolist())
                                # Get unique non-'O' events within this sentence
                                events_in_sentence = list(sentence_tokens[sentence_tokens['event'].notna() & (sentence_tokens['event'] != 'O')]['event'].unique())

                                st.markdown(f"**Sentence ({para_id}-{sent_id}):** {sentence_text}")
                                st.markdown(f"**Events:** `{events_in_sentence}`")
                                st.markdown("---")
                            except KeyError:
                                st.warning(f"Could not retrieve tokens for sentence {para_id}-{sent_id}. Skipping.")
                            except Exception as inner_e:
                                st.error(f"Error processing sentence {para_id}-{sent_id}: {inner_e}")
                else:
                    st.info("No asserted realis event annotations (non-'O') found in the .tokens file.")
            else:
                st.warning("'event' column not found in the loaded .tokens file.")
        except Exception as e:
            st.error(f"Error processing event data: {e}")
    else:
        st.warning("Token data (.tokens file) must be loaded to display events.")

    # --- Note about Tokens Data ---
    if tokens_df is not None:
        st.markdown("---")
        st.info(f"Detailed token data ({len(tokens_df):,} tokens) loaded successfully. \n" 
                 "This data is used for context lookups and future features but not displayed directly due to size.")
    else:
        st.warning("Token data (.tokens file) could not be loaded.")

    # --- Annotated Text Section (Embed HTML) ---
    st.markdown("---")
    st.header("Annotated Text (HTML Output)")

    html_file_path = os.path.join(selected_book_path, f"{selected_book_id}.book.html")

    if os.path.exists(html_file_path):
        try:
            with open(html_file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            st.components.v1.html(html_content, height=800, scrolling=True)
            st.success(f"Successfully loaded: {html_file_path}")
        except Exception as e:
            st.error(f"Error reading or displaying {html_file_path}: {e}")
    else:
        st.warning(f"File not found: {html_file_path}. Cannot display annotated text.")

    # --- Update Next Steps ---
    st.markdown("---")
    st.success("All planned sections implemented!")
    st.markdown("Future Enhancements:")
    st.markdown("- Add search/filtering to Entities/Quotes.")
    st.markdown("- Group Entities by coref_id.")
    st.markdown("- Link Speaker IDs in Quotes to Characters.")
    st.markdown("- Link PER Entities to Characters.")

else:
    st.info("Could not load book data. Check the sidebar warnings/errors.") 