"""
Compatibility patches for BookNLP.

This module contains patches to handle compatibility issues with different
versions of dependencies, particularly transformers 4.x+.
"""


def remove_position_ids_from_state_dict(state_dict):
	"""
	Remove 'bert.embeddings.position_ids' from a state dict if present.
	
	This fixes compatibility with transformers 4.x+ where position_ids
	is no longer a buffer in the BERT embeddings, causing:
	RuntimeError: Error(s) in loading state_dict for Tagger: 
		Unexpected key(s) in state_dict: "bert.embeddings.position_ids"
	
	Args:
		state_dict: PyTorch state dictionary loaded from a model file
		
	Returns:
		The state dict with position_ids removed if it was present
	"""
	key = "bert.embeddings.position_ids"
	if key in state_dict:
		del state_dict[key]
	return state_dict
