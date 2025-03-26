# Paper Extractor

A Python utility for extracting structured data from research paper abstracts using LLMs.

## Features

- Extract structured data from paper abstracts using OpenAI or LlamaIndex
- Support for direct extraction from DOI references
- Schema validation and type conversion
- Error handling for LLM outputs
- Returns data as pandas DataFrame

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Extract data from an abstract:

```python
from paper_extractor import extract_data_from_abstract

# Example abstract text
abstract = """
Background: Long-lasting insecticidal nets (LLINs) are the primary malaria prevention approach globally. However, 
insecticide resistance in vectors threatens the efficacy of insecticidal interventions, including LLINs. 
Interceptor® G2 is a new LLIN that contains a mixture of two insecticides: alpha-cypermethrin and chlorfenapyr.

Methods: This study was conducted in Northeast Tanzania between December 2017 to January 2018...
"""

# Extract data using OpenAI
df = extract_data_from_abstract(
    abstract, 
    llm_service="openai", 
    api_key="your_openai_api_key"
)

# Or use LlamaIndex
df = extract_data_from_abstract(
    abstract, 
    llm_service="llamaindex", 
    api_key="your_api_key"
)

print(df)
```

### Extract data directly from a DOI:

```python
from paper_extractor import extract_data_from_doi

# Extract data from a DOI
df = extract_data_from_doi(
    "10.1186/s12936-019-2973-x", 
    llm_service="openai", 
    api_key="your_openai_api_key"
)

print(df)
```

## Schema

The script extracts the following fields from research paper abstracts:

- **Pub_year**: Publication year (integer)
- **Journal**: Name of journal (string)
- **Study_type**: Hut trial, lab based bioassay, or village trial (string)
- **Net_type**: Names of LLINs tested, comma-separated if multiple (string)
- **Source**: Whether mosquitoes were from the wild or lab - 'Wild' or 'Lab' (string)
- **Country**: Country where the study was conducted (string)
- **Site**: Specific geographic information (string)
- **Start_date**: Study start date in YYYY-MM format (string)
- **End_date**: Study end date in YYYY-MM format (string)
- **Time_elapsed**: Time elapsed in months (float)

## Approach and Implementation

The implementation approach focused on creating a flexible, robust system for extracting structured data from academic paper abstracts:

1. **Dual LLM Integration**: Support for both OpenAI and LlamaIndex provides flexibility, allowing users to choose their preferred LLM service.

2. **Structured Output Format**: Carefully crafted prompts instruct the LLM to produce responses in valid JSON format to ensure consistent parsing.

3. **Schema Validation**: A rigorous validation process converts extracted values to the appropriate data types and handles missing or invalid values.

4. **Modular Design**: The codebase separates concerns into discrete functions for fetching abstracts, extracting data, and validating output.

5. **DOI Integration**: Added capability to directly fetch abstracts from DOI references, eliminating the need for manual copying.

## Challenges Faced

Several challenges were encountered during development:

1. **LLM Output Variability**: LLMs occasionally generate responses that don't strictly adhere to the requested format, requiring robust parsing logic to extract valid JSON.

2. **HTML Parsing Complexity**: Different publishers format their paper pages differently, making it difficult to create a universal abstract extraction method from DOIs.

3. **Date Extraction**: Dates in abstracts are often presented in various formats, requiring additional logic to standardize to the YYYY-MM format.

4. **Inferring Time Elapsed**: Calculating the time elapsed between dates often requires contextual understanding, as this information might not be explicitly stated.

5. **Type Conversion Edge Cases**: Converting extracted text to specific data types (especially numerics) requires handling a variety of edge cases and formats.

## Suggested Improvements

Several enhancements could make the extraction more robust and efficient:

1. **Few-Shot Learning**: Include examples of correct extractions in the prompt to guide the LLM toward more accurate outputs. This would significantly improve extraction accuracy by demonstrating the expected format and reasoning.

2. **Custom NER Model**: Train a specialized Named Entity Recognition model for scientific papers to pre-process abstracts and identify key entities before LLM extraction.

3. **Cross-Validation**: Implement a multi-LLM approach where extractions from different models are compared and reconciled for higher confidence.

4. **Structured Reasoning**: Break down the extraction process into steps, asking the LLM to first identify relevant sections before extraction.

5. **Caching Mechanism**: Implement a caching system for DOI fetching and LLM calls to improve efficiency for repeated queries.

6. **Enhanced Publisher Integration**: Develop dedicated parsers for major academic publishers to improve DOI-based abstract retrieval. 