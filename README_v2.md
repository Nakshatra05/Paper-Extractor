# Paper Net Data Extractor

A Python utility for extracting structured data about different net types from research papers on mosquito control.

## Features

- Extract structured data for multiple net types from a single paper
- Support for either OpenAI or LlamaIndex as LLM backends
- Pydantic schema validation and error handling
- Returns data as a pandas DataFrame with one row per net type

## Installation

```bash
pip install -r requirements_v2.txt
```

## Usage

```python
from paper_extractor_v2 import extract_net_data

# Define schema
schema = {
    "Study_type": "str",
    "Net_type": "str",
    "Insecticide": "str",
    "Num_washes": "int",
    "Source": "str",
    "Mosquito_spp": "str",
    "Site": "str",
    "Start_date": "str",
    "End_date": "str",
    "Time_elapsed": "float"
}

# Define net types to extract data for
net_types = ["Interceptor G2", "Interceptor", "untreated net"]

# Full paper text
paper_text = """
[Full text of the paper would go here]
"""

# Extract data
df = extract_net_data(
    paper_text,
    schema,
    net_types,
    llm_service="openai",
    api_key="your_api_key"
)

print(df)
```

## Schema

The script extracts the following fields for each net type:

- **Study_type**: One of: Hut trial, lab based bioassay, or village trial
- **Net_type**: The type or brand name of the insecticide-treated net
- **Insecticide**: The insecticide active ingredient in the net
- **Num_washes**: Number of times the net was washed
- **Source**: Mosquito source: 'Wild' or 'Lab'
- **Mosquito_spp**: Mosquito species tested with the net
- **Site**: Specific geographic information
- **Start_date**: Start date in YYYY-MM format
- **End_date**: End date in YYYY-MM format
- **Time_elapsed**: Duration in months

## Approach and Implementation

The implementation approach focused on extracting data for multiple net types from a single research paper:

1. **Multi-Object Extraction**: The system is designed to extract data for multiple net types at once, producing separate structured data for each type.

2. **Pydantic Validation**: Using Pydantic models for data validation ensures type safety and consistent output structure.

3. **Missing Data Handling**: The script automatically adds empty rows for any requested net types that weren't found in the LLM extraction.

4. **LLM Flexibility**: Support for both OpenAI and LlamaIndex provides options for different LLM integrations.

## Challenges Faced

Several challenges were encountered during development:

1. **Data Relationship Complexity**: Extracting data specific to each net type required carefully parsing relationships between data points that might be spread throughout the paper.

2. **JSON List Output**: LLMs sometimes struggle to output valid JSON lists, requiring robust parsing of the response.

3. **Contextual Understanding**: Net wash counts and other numeric data might be presented in different contexts (e.g., "unwashed" vs. "0 washes").

4. **Information Consistency**: The same information might be described differently across the paper, requiring the LLM to synthesize consistent data.

## Suggested Improvement

**Chain-of-Thought Extraction**: Break the extraction process into multiple steps:

1. First identify all net types mentioned in the paper
2. For each net type, extract key descriptive paragraphs
3. Then extract structured data from those focused paragraphs

This approach would improve extraction accuracy by allowing the LLM to first isolate relevant information about each net type before attempting to structure it. The multi-step approach would reduce confusion between different net types and their properties, especially in complex papers discussing multiple nets with similar characteristics. 