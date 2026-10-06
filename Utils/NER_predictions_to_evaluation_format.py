#!/usr/bin/env python
# coding: utf-8

# ### Convert predicted entities in the evaluation format

# Define the imports

# In[12]:


import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # repository root


# Define paths to the prediction files

# In[ ]:


PATH_NER_PREDICTIONS = ROOT / "Predictions/predicted_entities.json"


# Define output path

# In[ ]:


PATH_OUTPUT_NER_PREDICTIONS = ROOT / "Predictions/predicted_entities_eval_format.json"


# Load the input files into dictionary variables

# In[15]:


with open(PATH_NER_PREDICTIONS, 'r', encoding='utf-8') as file:
	ner_predictions = json.load(file)


# #### Define the functions to process NER predictions

# Merge consecutive NER predictions 

# In[16]:


def merge_consecutive_predictions(data):
    """
    Parse and merge consecutive predicted entities in the input dictionary where start and end indices are sequential.
    """
    print(f"Merging consecutive NER predictions...")

    # Process each document
    for pmid, doc in data.items():
        merged_entities = []
        current_entity = None

        for entity in doc.get("pred_entities", []):
            if current_entity is None:
                # Start a new entity
                current_entity = entity
            else:
                # Check if the current entity should be merged with the previous one
                if (
                    current_entity["end_idx"] + 1 == entity["start_idx"] and
                    current_entity["entity_label"] == entity["entity_label"]
                ):
                    # Merge entities by extending the current entity
                    current_entity["end_idx"] = entity["end_idx"]
                    current_entity["text_span"] += " " + entity["text_span"]
                    current_entity["score"] = min(current_entity["score"], entity["score"])
                elif(
                    current_entity["end_idx"] == entity["start_idx"] and
                    current_entity["entity_label"] == entity["entity_label"]
                ):
                    # Merge entities by extending the current entity
                    current_entity["end_idx"] = entity["end_idx"]
                    current_entity["text_span"] += entity["text_span"]
                    current_entity["score"] = min(current_entity["score"], entity["score"])
                else:
                    # Append the completed entity and start a new one
                    merged_entities.append(current_entity)
                    current_entity = entity

        # Append the last entity if any
        if current_entity is not None:
            merged_entities.append(current_entity)

        # Replace the original entities with the merged ones
        doc["pred_entities"] = merged_entities


# In[17]:


merge_consecutive_predictions(ner_predictions)


# Adjust the indices of predicted entities to reflect the ground truth format

# In[18]:


def adjust_predicted_indices(data):
    """
    Adjust the indices of predicted entities in the abstract by subtracting the length of the title
    from both the start and end indices, and decreasing the end index by 1.
    """
    print("Adjusting indices for NER predictions...")
    # Process each document
    for pmid, doc in data.items():
        title_length = len(doc.get("title", ""))  # Calculate the length of the title

        for entity in doc.get("pred_entities", []):
            entity["end_idx"] -= 1  # Adjust the end index to be exclusive

            if entity["tag"] == "a":  # Process only entities from the abstract
                entity["start_idx"] -= title_length + 1
                entity["end_idx"] -= title_length + 1


# In[19]:


adjust_predicted_indices(ner_predictions)


# Convert predicted entities to ground truth format

# In[20]:


def migrate_to_ground_truth_format(articles):
    return_dict = {}

    for pmid, article in articles.items():
        return_dict[pmid] = {}
        return_dict[pmid]['metadata'] = {}
        return_dict[pmid]['entities'] = []
        return_dict[pmid]['relations'] = []

        return_dict[pmid]['metadata']['title'] = article['title']
        return_dict[pmid]['metadata']['author'] = article['author']
        return_dict[pmid]['metadata']['journal'] = article['journal']
        return_dict[pmid]['metadata']['year'] = article['year']
        return_dict[pmid]['metadata']['abstract'] = article['abstract']
        return_dict[pmid]['metadata']['annotator'] = 'distant'

        for entity in article['pred_entities']:
            ent_dict = {
                "start_idx": entity['start_idx'],
                "end_idx": entity['end_idx'],
                "location": 'title' if entity['tag'] == 't' else 'abstract',
                "text_span": entity['text_span'],
                "label": entity['entity_label']
            }
            return_dict[pmid]['entities'].append(ent_dict)

    return return_dict


# In[21]:


predictions = migrate_to_ground_truth_format(ner_predictions)


# In[22]:


with open(PATH_OUTPUT_NER_PREDICTIONS, 'w', encoding='utf-8') as file:
    json.dump(predictions, file, indent=2)

